"""
Malignancy linear probe on frozen VAE latents -- NO augmentation, year-split test AUROC.

Differences from linear_probe_latents.py / probe_malignancy_pooled.py
---------------------------------------------------------------------
1. NO augmentation -- but the 64x64 centre crop still has to happen.

   `augment=False` alone would be WRONG: `augment_images` is what calls
   `image_translation(dims=64)`, and that centre crop is the only thing turning the
   128x128 patch into the 64x64 the VAE was trained on. With plain `augment=False` the
   loader hands back 128x128, the encoder (fully convolutional) happily produces
   (4,16,16), and the probe silently runs on a 1024-d latent at a resolution the VAE
   never saw. There is a runtime assert on the 256-d latent to catch exactly that.

   Going through `augment_images(rotate=False, translate=False)` would give the right
   answer but is a bad way to get it: that function ignores its `num_workers` argument
   and forks `cpu_count()-1` workers (63 on these nodes, whatever the cgroup allocation
   is), each buffering a `len(args)//num_workers` chunk of 128x128 images. On the pooled
   9498-image train set that OOM-killed a 16 GB job -- for work that is a pure slice.

   So: build with `augment=False`, then centre-crop the whole stacked array in one
   vectorised numpy slice. `centre_crop` reproduces `image_translation`'s offset formula
   exactly, and `build()` asserts equality against `image_translation` on a real image
   before using it.

   Train therefore drops from 9000 -> 1800 NLST rows (900 subjects x 2 years).

2. Inner CV is grouped by subject. Both years of a subject are still in train, so a plain
   StratifiedKFold would put year 1 and year 2 of the same patient on opposite sides of
   the inner split and pick C optimistically. StratifiedGroupKFold on the subject index
   fixes that. (With augmentation on, the same leak was 5x worse -- 5 rotations per row.)

3. Test AUROC is reported three ways, which is the point of this script:
       all    900 rows
       year1  450 rows  (the `cond_img` half, first-year scan)
       year2  450 rows  (the `gt_img` half, second-year scan)
   dataset.py builds `images = concat([cond_imgs, gt_imgs])` and
   `labels = [label...] * 2`, so the first half is year 1, the second half is year 2, in
   the same subject order, with identical labels. That is asserted at runtime.

   CIs are bootstrapped over SUBJECTS (450 clusters), not rows: the two year rows of a
   patient share a label and their scores correlate ~0.8, so a row bootstrap on the
   900-row column understates the width by ~25%.

FIT SETS (this is the "NLST only or everything?" question, answered both ways)
  nlst    NLST train rows only                                  -> ~1800 rows
  pooled  NLST + LUNA25 + DLCS + LUNA16, the VAE's own pool     -> ~9.5k rows
The extra datasets are gated to split=='train' in dataset.py, so the test column is the
identical NLST 900 either way and every delta is paired.

Run from the repo root on a GPU node:
    python probe_malignancy_noaug.py --out_dir results/vae_ablation/probe_noaug
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold as _GroupCV
    _GROUP_CV_NAME = "StratifiedGroupKFold"
except ImportError:                                  # sklearn < 0.24
    from sklearn.model_selection import GroupKFold as _GroupCV
    _GROUP_CV_NAME = "GroupKFold"

from dataset import Lung_Pair_DS
from utils import image_translation
from linear_probe_latents import (
    SEED, N_BOOT, CS, BASE_PATH, PROBE_FEATURES,
    set_seeds, load_encoder, extract, reduce_latent,
    m_auroc, m_auprc, ci, fmt, render_plain, render_latex,
)

# THE change: centre crop only, no rotated copies, and no augment_images pool.
NOAUG_CFG = dict(augment=False, centre_crop_dims=64)
CROP_DIMS = 64

REDUCTIONS = ["flatten", "pooled"]
FIT_SETS = ["nlst", "pooled"]
METRICS = [("auroc", m_auroc), ("auprc", m_auprc)]
SLICES = ["all", "year1", "year2"]

DEFAULT_CKPTS = {
    "align": "checkpoints/vae-abl-align-full-bs32-lr5.0e-05-s23/"
             "epoch=34-val/total_loss=0.614.ckpt",
    "pred": "checkpoints/vae-abl-pred-full-bs32-lr2.0e-05-s23/"
            "epoch=08-val/total_loss=0.212.ckpt",
    "published_full": "checkpoints/autoencoder/v1.5-autoencoder-contrastive-0211.ckpt",
    "published_none": "checkpoints/autoencoder/v1.5-autoencoder-original-0211.ckpt",
}


def centre_crop(arr, dims=CROP_DIMS):
    """Same offsets as utils.image_translation, applied to a whole (N, C, H, W) stack."""
    h, w = arr.shape[-2], arr.shape[-1]
    ho, wo = (h - dims) // 2, (w - dims) // 2
    return arr[..., ho:ho + dims, wo:wo + dims]


def build(split, pooled):
    ds = Lung_Pair_DS(
        split=split, ch=3, model_type="autoencoder",
        selected_features=PROBE_FEATURES, base_path=BASE_PATH,
        has_Luna25=pooled, has_DLCS=pooled, has_Luna16=pooled,
        augment=False,
    )
    images = np.asarray(ds.images)
    assert images.shape[-1] == images.shape[-2], (
        f"expected square per-image patches after the cond/gt split, got {images.shape}")
    cropped = centre_crop(images)

    # the crop this script does must be byte-identical to the one the rest of the
    # codebase does via augment_images -> image_translation(dims=64)
    ref, _ = image_translation(images[0], dims=CROP_DIMS)
    assert np.array_equal(cropped[0], ref), "centre_crop disagrees with image_translation"
    ref_last, _ = image_translation(images[-1], dims=CROP_DIMS)
    assert np.array_equal(cropped[-1], ref_last)

    ds.images = np.ascontiguousarray(cropped)
    ds.features = np.asarray(ds.features, dtype=np.float64)
    ds.labels = np.asarray(ds.labels, dtype=np.float64)
    assert len(ds.images) == len(ds.features) == len(ds.labels)
    return ds


def nlst_mask(F):
    """NLST rows carry a full feature record; add_Luna25/DLCS/Luna16 rows are all-NaN."""
    return ~np.all(np.isnan(F), axis=1)


def subject_groups(n_rows, n_nlst):
    """
    Group id per train row, for the grouped inner CV.

    NLST rows are [year1 x N, year2 x N] in the same subject order, so row i and row i+N
    are the same patient and must share a group. Every extra-dataset row is its own group.
    """
    N = n_nlst // 2
    g = np.arange(n_rows)
    g[:N] = np.arange(N)
    g[N:n_nlst] = np.arange(N)
    return g


def fit_logreg_grouped(Ztr, ytr, groups, Zte, seed):
    """LogisticRegression, C chosen by grouped stratified CV on the train rows."""
    scaler = StandardScaler().fit(Ztr)
    Xtr, Xte = scaler.transform(Ztr), scaler.transform(Zte)
    y = ytr.astype(int)
    counts = np.bincount(y)
    folds = max(2, int(min(5, counts[counts > 0].min())))
    cv = (_GroupCV(n_splits=folds, shuffle=True, random_state=seed)
          if _GROUP_CV_NAME == "StratifiedGroupKFold" else _GroupCV(n_splits=folds))
    base = LogisticRegression(max_iter=5000, class_weight="balanced")
    gs = GridSearchCV(base, {"C": CS}, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(Xtr, y, groups=groups)
    return gs.best_estimator_.predict_proba(Xte)[:, 1], float(gs.best_params_["C"])


def subject_boot(n_subj, seed, n=N_BOOT):
    """Resample the 450 test SUBJECTS, not the 900 rows."""
    return np.random.default_rng(seed).integers(0, n_subj, (n, n_subj))


def boot_ci(y, s, fn, subj_idx, which, n_subj):
    """which in {'all','year1','year2'}; rows i and i+n_subj are the same subject."""
    out = np.empty(len(subj_idx))
    for i, ix in enumerate(subj_idx):
        if which == "all":
            rows = np.concatenate([ix, ix + n_subj])
        elif which == "year1":
            rows = ix
        else:
            rows = ix + n_subj
        try:
            out[i] = fn(y[rows], s[rows])
        except ValueError:
            out[i] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=None, help="name=path ...")
    ap.add_argument("--reference", default="published_none")
    ap.add_argument("--out_dir", default="results/vae_ablation/probe_noaug")
    args = ap.parse_args()

    ckpts = DEFAULT_CKPTS.copy()
    if args.ckpts:
        ckpts = dict(spec.split("=", 1) for spec in args.ckpts)
    for name, path in ckpts.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name}: {path}")
    if args.reference and args.reference not in ckpts:
        raise ValueError(f"--reference {args.reference} not among {list(ckpts)}")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seeds(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[seeds] SEED={SEED}   [device] {device}")
    print(f"[aug] {NOAUG_CFG}  -> 64x64 centre crop only, 1 row per image, no pool")
    print(f"[cv] inner CV = {_GROUP_CV_NAME} grouped by subject")

    print("\n[data] pooled train (NLST + LUNA25 + DLCS + LUNA16), no augmentation ...")
    train_ds = build("train", pooled=True)
    print("[data] test (NLST, centre crop) ...")
    test_ds = build("test", pooled=False)
    print(f"[data] train rows={len(train_ds)}  test rows={len(test_ds)}")

    meta = {
        "seed": SEED, "device": device, "n_bootstrap": N_BOOT,
        "checkpoints": ckpts, "reference": args.reference, "base_path": BASE_PATH,
        "augmentation": NOAUG_CFG,
        "augmentation_note": (
            "no augmentation: augment=False, then a vectorised 64x64 centre crop that is "
            "asserted byte-identical to utils.image_translation(dims=64). Skipping the "
            "crop entirely would leave 128x128 images -> a 1024-d latent at a resolution "
            "the VAE never trained on. augment_images() is bypassed because it forks "
            "cpu_count()-1 workers regardless of its num_workers argument."),
        "inner_cv": f"{_GROUP_CV_NAME} grouped by subject (both years share a group)",
        "bootstrap": "resamples the 450 test SUBJECTS, not the 900 rows",
        "fit_sets": {
            "nlst": "NLST train rows only",
            "pooled": "NLST + LUNA25 + DLCS + LUNA16 (the VAE's own training pool)",
        },
        "test_slices": {
            "all": "900 rows", "year1": "450 cond_img rows", "year2": "450 gt_img rows"},
    }

    # ---- latents, once per checkpoint ----
    store, n_nlst = {}, None
    for name, path in ckpts.items():
        print(f"\n[latent] {name}: {path}")
        model = load_encoder(path, device)
        Ztr, Ftr, Ytr = extract(model, train_ds, device)
        Zte, Fte, Yte = extract(model, test_ds, device)
        del model
        torch.cuda.empty_cache()
        assert Ztr.shape[1] == 256, (
            f"latent is {Ztr.shape[1]}-d, expected 256. The centre crop did not happen "
            f"-- check that augment=True.")
        m = nlst_mask(Ftr)
        store[name] = (Ztr, Ytr, m, Zte, Yte)
        if n_nlst is None:
            n_nlst = int(m.sum())
            print(f"[composition] train rows={len(Ytr)}  NLST={n_nlst}  "
                  f"extra={len(Ytr) - n_nlst}")

    _, Ytr0, m_nlst0, _, Yte0 = store[next(iter(ckpts))]
    groups = subject_groups(len(Ytr0), n_nlst)

    # ---- test layout: [year1 x 450, year2 x 450], identical labels ----
    n_te = len(Yte0)
    assert n_te % 2 == 0
    n_subj = n_te // 2
    yte = Yte0
    assert np.array_equal(yte[:n_subj], yte[n_subj:]), (
        "test halves do not carry identical labels -- the year1/year2 split assumption "
        "in this script is wrong for this dataset build")
    sl = {"all": np.arange(n_te),
          "year1": np.arange(n_subj),
          "year2": np.arange(n_subj, n_te)}
    meta["n_test_subjects"] = n_subj
    meta["n_test_rows"] = n_te
    meta["prevalence_test"] = float(np.mean(yte))
    meta["prevalence_train_nlst"] = float(np.mean(Ytr0[m_nlst0]))
    meta["prevalence_train_pooled"] = float(np.mean(Ytr0))
    print(f"\n[test] {n_te} rows = {n_subj} subjects x 2 years, "
          f"prevalence={meta['prevalence_test']:.4f}")

    b_idx = subject_boot(n_subj, SEED)
    metrics, boots = {"meta": meta}, {}

    for red in REDUCTIONS:
        print(f"\n{'=' * 88}\nREDUCTION: {red}\n{'=' * 88}")
        for name in ckpts:
            Ztr, Ytr, m_nlst, Zte, _ = store[name]
            Rtr, Rte = reduce_latent(Ztr, red), reduce_latent(Zte, red)
            for fs in FIT_SETS:
                sel = (m_nlst if fs == "nlst" else np.ones(len(Ytr), bool)) & ~np.isnan(Ytr)
                s, hp = fit_logreg_grouped(Rtr[sel], Ytr[sel], groups[sel], Rte, SEED)

                entry = {"n_train": int(sel.sum()), "C": hp,
                         "prevalence_train": float(np.mean(Ytr[sel]))}
                for which in SLICES:
                    rows = sl[which]
                    entry[which] = {}
                    for mname, fn in METRICS:
                        dist = boot_ci(yte, s, fn, b_idx, which, n_subj)
                        lo, hi = ci(dist)
                        entry[which][mname] = {
                            "value": float(fn(yte[rows], s[rows])), "ci95": [lo, hi]}
                        boots[(name, red, fs, which, mname)] = dist
                metrics.setdefault(name, {}).setdefault(red, {})[fs] = entry

                print(f"  {name:<16} fit={fs:<7} n={entry['n_train']:>6} C={hp:<9.4g} "
                      + "  ".join(f"{w}={entry[w]['auroc']['value']:.3f}" for w in SLICES))

                with open(os.path.join(args.out_dir,
                                       f"predictions_{name}_{red}_{fs}.csv"),
                          "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["idx", "year", "y_true", "score"])
                    for i in range(n_te):
                        w.writerow([i, 1 if i < n_subj else 2,
                                    f"{yte[i]:.6g}", f"{s[i]:.6g}"])

    # ---- paired deltas vs the reference arm (shared subject resamples) ----
    if args.reference:
        for red in REDUCTIONS:
            for name in ckpts:
                if name == args.reference:
                    continue
                for fs in FIT_SETS:
                    for which in SLICES:
                        for mname, _ in METRICS:
                            d = (boots[(name, red, fs, which, mname)]
                                 - boots[(args.reference, red, fs, which, mname)])
                            lo, hi = ci(d)
                            pt = (metrics[name][red][fs][which][mname]["value"]
                                  - metrics[args.reference][red][fs][which][mname]["value"])
                            metrics[name][red][fs][which].setdefault("delta", {})[mname] = {
                                "value": pt, "ci95": [lo, hi],
                                "excludes_zero": bool(lo > 0 or hi < 0)}

    # ---- tables ----
    cols = ["Model", "Fit set", "n train", "AUROC all (900)",
            "AUROC year1 (450)", "AUROC year2 (450)"]
    for red in REDUCTIONS:
        rows = []
        for name in ckpts:
            for fs in FIT_SETS:
                e = metrics[name][red][fs]
                rows.append([name, fs, str(e["n_train"])]
                            + [fmt(e[w]["auroc"]["value"], *e[w]["auroc"]["ci95"])
                               for w in SLICES])
        plain = render_plain(cols, rows)
        with open(os.path.join(args.out_dir, f"table_{red}.txt"), "w") as f:
            f.write(plain + "\n")
        with open(os.path.join(args.out_dir, f"table_{red}.tex"), "w") as f:
            f.write(render_latex(cols, rows, f"no augmentation, reduction={red}") + "\n")
        print(f"\n{'=' * 88}\nMALIGNANCY AUROC - no augmentation - reduction={red}\n")
        print(plain)

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[saved] -> {args.out_dir}/")
    print("  CIs are subject-level bootstraps (450 clusters), not row bootstraps.")


def selfcheck():
    """Crop, group assignment and year1/year2 slicing, without touching the GPU."""
    rng = np.random.default_rng(0)
    img = rng.random((4, 3, 128, 128)).astype(np.float32)
    got = centre_crop(img)
    assert got.shape == (4, 3, 64, 64), got.shape
    for i in range(4):
        ref, _ = image_translation(img[i], dims=64)
        assert np.array_equal(got[i], ref), f"centre_crop != image_translation at {i}"

    g = subject_groups(n_rows=10, n_nlst=6)
    assert list(g) == [0, 1, 2, 0, 1, 2, 6, 7, 8, 9], list(g)

    y = np.array([0, 1, 1, 0, 1, 1], float)          # 3 subjects x 2 years
    s = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.6])
    n_subj = 3
    assert np.array_equal(y[:n_subj], y[n_subj:])
    assert m_auroc(y[:n_subj], s[:n_subj]) == 1.0
    assert m_auroc(y[n_subj:], s[n_subj:]) == 1.0
    assert m_auroc(y, s) == 1.0

    idx = subject_boot(n_subj, SEED, n=5)
    d = boot_ci(y, s, m_auroc, idx, "year2", n_subj)
    assert len(d) == 5 and np.all(np.isnan(d) | (d <= 1.0))
    print("selfcheck ok")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        main()
