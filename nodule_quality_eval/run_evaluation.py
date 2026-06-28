"""End-to-end nodule morphology quality evaluation.

Pipeline for every input triplet (T0, T1_real, T1_gen):
  1. Segment T1_real with MedSAM 2 (centered bbox prompt).
  2. Segment T1_gen with MedSAM 2 (bbox derived from the T1_real mask + margin).
  3. Segment T0 with Otsu thresholding inside the dilated T1_real mask
     (anchoring on T1_real avoids MedSAM grabbing chest wall on noisy
     baselines where the nodule is barely visible).
  4. Extract 8 shape descriptors + 93 IBSI radiomic features (PyRadiomics)
     per ROI.
  5. Write per-sample metrics, long-form radiomics, and trajectory
     consistency CSVs.

Caches every binary mask, every feature JSON, and a red-outline QC PNG
under `cache/` so subsequent runs reuse them.

Usage:
    # image input (default):
    python run_evaluation.py --input-dir path/to/images \
                             --cache-dir cache --out-dir results

    # legacy .npy input (idx0=T0, idx1=T1_gen, idx2=T1_real):
    python run_evaluation.py --input-dir path/to/npy --input-format npy

See README.md for the expected directory layout.
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from skimage.segmentation import find_boundaries

from load_triplets import (
    iter_image_triplets,
    iter_npy_triplets,
    parse_label,
)
from segmentation import MedSAM2Segmenter, segment_from_prior


# ----------------------------------------------------------------------
def get_metrics_module(backend):
    if backend == "pyradiomics":
        from metrics import (area_metrics, intensity_metrics,
                             radiomic_features, shape_features)
    elif backend == "skimage":
        from metrics_skimage import (area_metrics, intensity_metrics,
                                     radiomic_features, shape_features)
    else:
        raise ValueError(f"unknown metrics backend: {backend}")
    return area_metrics, intensity_metrics, radiomic_features, shape_features


def cache_paths(cache_dir, timepoint, sample_id, seed=None):
    if timepoint == "T1_gen":
        base = os.path.join(cache_dir, "T1_gen", seed or "default")
    else:
        base = os.path.join(cache_dir, timepoint)
    return (
        os.path.join(base, "masks", f"{sample_id}.npy"),
        os.path.join(base, "features", f"{sample_id}.json"),
        os.path.join(base, "overlays", f"{sample_id}.png"),
    )


def save_overlay(image_2d, mask_2d, out_path, scale=1):
    """Write a PNG with the mask outline drawn in red over the grayscale patch."""
    rgb = np.repeat(image_2d[:, :, None], 3, axis=-1).astype(np.uint8)
    boundary = find_boundaries(mask_2d, mode="outer")
    rgb[boundary] = [255, 0, 0]
    img = Image.fromarray(rgb)
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def ensure_mask(seg, image, mask_path, overlay_path, box=None, prior_mask=None):
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
    else:
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        if prior_mask is not None:
            mask = segment_from_prior(image, prior_mask)
        else:
            mask = seg.segment(image, box=box)
        np.save(mask_path, mask)
    if not os.path.exists(overlay_path):
        save_overlay(image, mask, overlay_path)
    return mask


def mask_to_box(mask, image_shape, margin=5, min_pixels=5):
    if mask.sum() < min_pixels:
        return None
    H, W = image_shape
    ys, xs = np.where(mask)
    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, W)
    y1 = min(int(ys.max()) + margin + 1, H)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def ensure_features(image, mask, cache_path, shape_fn, radiomic_fn):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    feats = {
        "shape": shape_fn(image, mask),
        "radiomic": radiomic_fn(image, mask),
    }
    with open(cache_path, "w") as f:
        json.dump(feats, f)
    return feats


# ----------------------------------------------------------------------
def process_one(seg, seed, sample_id, T0, T1_real, T1_gen,
                cache_dir, area_metrics, intensity_metrics,
                shape_features, radiomic_features):
    label = parse_label(sample_id)

    mp_T0, fp_T0, op_T0 = cache_paths(cache_dir, "T0", sample_id)
    mp_T1r, fp_T1r, op_T1r = cache_paths(cache_dir, "T1_real", sample_id)
    mp_T1g, fp_T1g, op_T1g = cache_paths(cache_dir, "T1_gen", sample_id, seed=seed)

    m_T1r = ensure_mask(seg, T1_real, mp_T1r, op_T1r)
    anchor_box = mask_to_box(m_T1r, T1_real.shape, margin=5)
    m_T1g = ensure_mask(seg, T1_gen, mp_T1g, op_T1g, box=anchor_box)
    m_T0 = ensure_mask(seg, T0, mp_T0, op_T0, prior_mask=m_T1r)

    f_T0 = ensure_features(T0, m_T0, fp_T0, shape_features, radiomic_features)
    f_T1r = ensure_features(T1_real, m_T1r, fp_T1r, shape_features, radiomic_features)
    f_T1g = ensure_features(T1_gen, m_T1g, fp_T1g, shape_features, radiomic_features)

    row = {"sample": sample_id, "label": label, "seed": seed}
    row.update(area_metrics(m_T0, m_T1r, m_T1g))
    row.update(intensity_metrics(T0, T1_real, T1_gen, m_T0, m_T1r, m_T1g))

    sh_r = f_T1r["shape"]
    sh_g = f_T1g["shape"]
    for k, v in sh_r.items():
        row[f"{k}_T1r"] = v
        row[f"{k}_T1g"] = sh_g.get(k, np.nan)
        try:
            row[f"{k}_abs_err"] = abs(v - sh_g.get(k, np.nan))
        except TypeError:
            row[f"{k}_abs_err"] = np.nan

    return row, f_T0["radiomic"], f_T1r["radiomic"], f_T1g["radiomic"]


# ----------------------------------------------------------------------
def save_csv(path, rows, leading_cols):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    keys = leading_cols + [k for k in keys if k not in leading_cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_radiomics_long(path, ids, labels, seeds, T0s, T1rs, T1gs):
    keys = sorted(set().union(*[set(d.keys())
                                for d in T0s + T1rs + T1gs if d]))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "label", "seed", "timepoint"] + keys)
        for sid, lab, seed, t0, t1r, t1g in zip(
            ids, labels, seeds, T0s, T1rs, T1gs
        ):
            for tag, fd in [("T0", t0), ("T1_real", t1r), ("T1_gen", t1g)]:
                w.writerow([sid, lab, seed, tag]
                           + [fd.get(k, np.nan) for k in keys])


def trajectory_consistency(path, T0s, T1rs, T1gs):
    """Write per-feature trajectory agreement + the global cosine.

    Returns True if the CSV was written, False if the run is degenerate and
    nothing was written. The common feature set is an *intersection* across
    every ROI, so a single ROI whose mask was too small for radiomic
    extraction (an empty feature dict) collapses it to nothing -- which would
    make this file all-NaN with no per-feature rows. When that happens we skip
    the file and let the caller direct the user to analysis/progression.py,
    which derives the same trajectory metrics after dropping those samples.
    """
    if not T0s:
        return False
    common = set(T0s[0].keys())
    for d in T0s + T1rs + T1gs:
        common &= set(d.keys())
    keys = sorted(common)
    if not keys:
        return False

    delta_real = np.array(
        [[t1r[k] - t0[k] for k in keys] for t0, t1r in zip(T0s, T1rs)],
        dtype=np.float64,
    )
    delta_gen = np.array(
        [[t1g[k] - t0[k] for k in keys] for t0, t1g in zip(T0s, T1gs)],
        dtype=np.float64,
    )

    cosines = []
    for dr, dg in zip(delta_real, delta_gen):
        if np.linalg.norm(dr) > 0 and np.linalg.norm(dg) > 0:
            cosines.append(1.0 - float(cosine(dr, dg)))
        else:
            cosines.append(np.nan)

    rows = []
    for j, k in enumerate(keys):
        try:
            rho, p = spearmanr(delta_real[:, j], delta_gen[:, j])
        except Exception:
            rho, p = np.nan, np.nan
        mae = float(np.mean(np.abs(delta_real[:, j] - delta_gen[:, j])))
        rows.append({"feature": k, "spearman_rho": rho,
                     "p_value": p, "mae": mae})

    rows.sort(
        key=lambda r: -(r["spearman_rho"]
                        if not np.isnan(r["spearman_rho"]) else -2.0)
    )

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["overall_cosine_similarity_mean",
                    float(np.nanmean(cosines))])
        w.writerow(["overall_cosine_similarity_std",
                    float(np.nanstd(cosines))])
        w.writerow(["n_samples", len(cosines)])
        w.writerow([])
        w.writerow(["feature", "spearman_rho", "p_value", "mae"])
        for r in rows:
            w.writerow([r["feature"], r["spearman_rho"],
                        r["p_value"], r["mae"]])
    return True


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True,
                    help="Root directory of input triplets "
                         "(see README for layout).")
    ap.add_argument("--input-format", choices=["image", "npy"],
                    default="image",
                    help="image: PNG/JPG/TIF per timepoint per sample folder; "
                         "npy: stacked (3, H, W) array per sample.")
    ap.add_argument("--metrics-backend", choices=["pyradiomics", "skimage"],
                    default="pyradiomics",
                    help="Feature extractor. 'pyradiomics' = full 93-feature "
                         "IBSI panel; 'skimage' = lightweight first-order + "
                         "GLCM (no pyradiomics dependency).")
    ap.add_argument("--cache-dir", default="cache",
                    help="Directory to cache masks, features, and overlays.")
    ap.add_argument("--out-dir", default="results",
                    help="Directory to write CSV outputs.")
    ap.add_argument("--out-tag", default="all",
                    help="Suffix appended to output CSV filenames.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    area_metrics, intensity_metrics, radiomic_features, shape_features = \
        get_metrics_module(args.metrics_backend)

    print(f"Loading MedSAM 2 (auto-downloads checkpoint on first run) ...")
    seg = MedSAM2Segmenter()

    if args.input_format == "image":
        triplet_iter = iter_image_triplets(args.input_dir)
    else:
        triplet_iter = iter_npy_triplets(args.input_dir)

    rows = []
    ids, labels_list, seeds_list = [], [], []
    rT0s, rT1rs, rT1gs = [], [], []

    n = 0
    for seed, sample_id, T0, T1_real, T1_gen in triplet_iter:
        try:
            row, rT0, rT1r, rT1g = process_one(
                seg, seed, sample_id, T0, T1_real, T1_gen,
                args.cache_dir, area_metrics, intensity_metrics,
                shape_features, radiomic_features,
            )
        except Exception as e:
            print(f"  [skip] {seed}/{sample_id}: {e}")
            continue
        rows.append(row)
        ids.append(row["sample"])
        labels_list.append(row["label"])
        seeds_list.append(seed)
        rT0s.append(rT0)
        rT1rs.append(rT1r)
        rT1gs.append(rT1g)
        n += 1
        if n % 25 == 0:
            print(f"  processed {n}")

    print(f"\nTotal processed: {n}")

    tag = args.out_tag
    summary_path = os.path.join(args.out_dir, f"per_sample_summary_{tag}.csv")
    save_csv(summary_path, rows, leading_cols=["sample", "label", "seed"])
    print(f"Wrote {summary_path}")

    if any(rT0s):
        radiomics_path = os.path.join(
            args.out_dir, f"radiomic_features_per_sample_{tag}.csv")
        save_radiomics_long(
            radiomics_path,
            ids, labels_list, seeds_list, rT0s, rT1rs, rT1gs,
        )
        print(f"Wrote {radiomics_path}")

        traj_path = os.path.join(
            args.out_dir, f"trajectory_consistency_{tag}.csv")
        if trajectory_consistency(traj_path, rT0s, rT1rs, rT1gs):
            print(f"Wrote {traj_path}")
        else:
            n_empty = sum(1 for d in rT0s + rT1rs + rT1gs if not d)
            print(
                f"Skipped {os.path.basename(traj_path)}: {n_empty} ROI(s) had a "
                f"mask too small for radiomic extraction, which empties the "
                f"cross-ROI feature intersection. Read the trajectory-consistency "
                f"metrics from analysis/progression.py instead -- it drops those "
                f"samples and computes the trajectory cosine over the rest."
            )


if __name__ == "__main__":
    main()
