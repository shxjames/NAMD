"""Compact all-20-seed grids for the NAMD_3 case studies.

Two figures:

  success_grid.png  Cond X^(1) | GT X^(2) | all 20 generation seeds.
                    Minimal captions, no error maps.

  failure_grid.png  Same layout plus a mean-error map, restricted to the four
                    hand-picked cases, and with the ViT's malignancy
                    probability printed under every single seed so it is
                    visible how wrong each generation makes the classifier.

Run after select_cases.py — it reads that script's npz and CSVs.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from select_cases import (DEFAULT_NAMD_DIR, load_images, sigmoid,
                          youden_threshold)

# Hand-picked failures: two malignant then two benign, so the label split reads
# straight off the figure. #230 and #202 are perceptual collapses; #203 and #357
# are generator-induced classifier flips -- for #357 the ViT is 99.8% sure the
# real follow-up is malignant, and the generator smooths the spiculated mass
# into a smaller round blob (185 -> 133 px) that drags the score to 0.40.
#
# Only two malignant samples in the cohort flip this way (the other is #233,
# where the ViT is far less sure of the real image at p=0.686), so this row has
# essentially no alternatives.
FAILURE_SAMPLES = [230, 357, 202, 203]

CORRECT_C = "#1a7f37"
WRONG_C = "#c8102e"

# MedSAM2 masks already segmented by the nodule eval — reused, not recomputed.
MASK_CACHE = ("/scratch/liyues_root/liyues/shxjames/NAMD/nodule_quality_eval/"
              "cache_NAMD_3")
SELF_C = "#31d6ff"      # this tile's own nodule
REF_C = "#ffc233"       # the real follow-up's nodule, for comparison


def load_mask(kind, sample, label, seed=None):
    """kind: 'T0' | 'T1_real' | 'T1_gen'. Returns a bool [H, W] or None."""
    stem = f"sample_{int(sample)}_{int(label)}.npy"
    if kind == "T1_gen":
        path = os.path.join(MASK_CACHE, "T1_gen", f"boot_{seed}", "masks", stem)
    else:
        path = os.path.join(MASK_CACHE, kind, "masks", stem)
    return np.load(path) if os.path.exists(path) else None


def contour(ax, mask, color, ls="solid", lw=0.7):
    if mask is None or not mask.any():
        return
    ax.contour(mask.astype(float), levels=[0.5], colors=[color],
               linewidths=lw, linestyles=ls)


def row_label(row, with_probs):
    txt = "malignant" if row.label == 1 else "benign"
    if with_probs:
        txt += f"\np(mal) {row.p_gen_mean:.2f}$\\pm${row.p_gen_std:.2f}"
    return txt


def draw_grid(df, t0, real, gen, seed_ids, id_to_pos,
              out_path, title, show_error=False, show_probs=False, probs=None,
              contours=False, outline_sources=False):
    """show_probs puts the across-seed ViT malignancy probability in the row
    caption and, when `probs` [S, N] is given, prints each seed's own
    probability under its tile -- green where that seed's call matches the
    ground-truth label, red where it does not."""
    n_rows = len(df)
    n_cols = 2 + len(seed_ids) + (1 if show_error else 0)
    per_tile = show_probs and probs is not None

    tile = 0.68 if per_tile else 0.60
    label_w = 1.20 if show_probs else 0.80
    fig_w = label_w + n_cols * tile
    head_h = 0.42
    row_h = tile + (0.20 if per_tile else 0.04)
    fig_h = head_h + n_rows * row_h + 0.30

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=label_w / fig_w, right=0.998,
        bottom=0.02, top=1 - head_h / fig_h,
        wspace=0.06, hspace=0.30 if per_tile else 0.06)

    for r, (_, row) in enumerate(df.iterrows()):
        j = id_to_pos[int(row["sample"])]
        gen_j = gen[:, j]
        if contours:
            s, lb = int(row["sample"]), int(row.label)
            m_t0 = load_mask("T0", s, lb)
            m_gt = load_mask("T1_real", s, lb)
            m_gen = [load_mask("T1_gen", s, lb, seed=sd) for sd in seed_ids]

        tiles = [t0[j], real[j]] + [gen_j[s] for s in range(len(seed_ids))]
        heads = (["Cond $X^{(1)}$", "GT $X^{(2)}$"]
                 + [f"gen {i + 1}" for i in range(len(seed_ids))])

        for c, (img, head) in enumerate(zip(tiles, heads)):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(head, fontsize=7.5, pad=3)
            if contours:
                if c >= 2:
                    # The real follow-up's outline sits under each generation so
                    # over- and under-growth read off the tile directly.
                    contour(ax, m_gt, REF_C, ls="dashed", lw=0.55)
                    contour(ax, m_gen[c - 2], SELF_C)
                elif outline_sources:
                    contour(ax, m_t0 if c == 0 else m_gt,
                            SELF_C if c == 0 else REF_C)
            if per_tile and c >= 2:
                p = probs[c - 2, j]
                ok = (p >= 0.5) == (row.label == 1)
                ax.text(0.5, -0.06, f"{p:.2f}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=6.2,
                        color=CORRECT_C if ok else WRONG_C,
                        fontweight="normal" if ok else "bold")

        if show_error:
            ax = fig.add_subplot(gs[r, n_cols - 1])
            err = np.abs(gen_j.astype(np.float32).mean(0) - real[j].astype(np.float32))
            ax.imshow(err, cmap="magma", vmin=0, vmax=max(err.max(), 1e-6))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title("$|\\overline{gen}-GT|$", fontsize=7.5, pad=3)

        # Row caption, in the left margin.
        first = fig.add_subplot(gs[r, 0])
        first.set_visible(False)
        y = (first.get_position().y0 + first.get_position().y1) / 2
        fig.text((label_w - 0.10) / fig_w, y, row_label(row, show_probs),
                 ha="right", va="center", fontsize=7.2, family="monospace")

    fig.suptitle(title, fontsize=11, y=1 - 0.06 * (0.42 / fig_h) - 0.004)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({n_rows} x {n_cols})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--namd_dir", default=DEFAULT_NAMD_DIR)
    ap.add_argument("--dir", default="results/NAMD_3_cases",
                    help="output dir of select_cases.py; grids land here too")
    args = ap.parse_args()

    npz = np.load(os.path.join(args.dir, "per_sample_seed_metrics.npz"))
    thr = youden_threshold(npz["labels"], npz["logit_real"][0])
    probs = sigmoid(npz["logit_gen"] - thr)          # [S, N] p(malignant)

    full = pd.read_csv(os.path.join(args.dir, "per_sample_metrics.csv"))
    succ = pd.read_csv(os.path.join(args.dir, "success_cases.csv"))

    fail = full.set_index("sample").loc[FAILURE_SAMPLES].reset_index()
    pos = {int(s): i for i, s in enumerate(npz["sample_ids"])}
    sel = [pos[s] for s in FAILURE_SAMPLES]
    fail["p_gen_mean"] = probs[:, sel].mean(0)
    fail["p_gen_std"] = probs[:, sel].std(0)

    t0, real, gen, seed_ids, ids = load_images(args.namd_dir)
    id_to_pos = {int(s): i for i, s in enumerate(ids)}
    assert np.array_equal(ids, npz["sample_ids"]), "image order != metric order"

    # Plain, then a MedSAM2-contoured variant of each; the plain versions stay
    # the default figures.
    note = ("   (cyan = generated nodule outline; "
            "dashed amber = real follow-up outline)")
    for cont, suffix in ((False, ""), (True, "_contours")):
        draw_grid(succ, t0, real, gen, seed_ids, id_to_pos,
                  os.path.join(args.dir, f"success_grid{suffix}.png"),
                  "NAMD generations" + (note if cont else ""),
                  contours=cont, outline_sources=True)

        draw_grid(fail, t0, real, gen, seed_ids, id_to_pos,
                  os.path.join(args.dir, f"failure_grid{suffix}.png"),
                  "Representative failure cases of NAMD" + (note if cont else ""),
                  show_error=True, show_probs=True, probs=probs, contours=cont)


if __name__ == "__main__":
    main()
