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
# straight off the figure. Within each pair the first is a perceptual collapse
# and the second a classifier flip.
FAILURE_SAMPLES = [230, 233, 202, 203]

CORRECT_C = "#1a7f37"
WRONG_C = "#c8102e"


def row_label(row, with_probs):
    tag = "malignant" if row.label == 1 else "benign"
    txt = f"#{int(row['sample'])}  {tag}\nLPIPS {row.lpips_mean:.3f}"
    if with_probs:
        # The Cond / GT probabilities are printed under their own tiles, so the
        # caption only carries the across-seed aggregate.
        txt += (f"\nViT ok {int(row.seeds_correct)}/20"
                f"\n$p_{{mal}}$ gen {row.p_gen_mean:.2f}$\\pm${row.p_gen_std:.2f}")
    return txt


def draw_grid(df, t0, real, gen, seed_ids, lpips, probs, id_to_pos,
              out_path, title, show_error=False, show_probs=False,
              p_cond=None, p_gt=None):
    """probs: [S, N] ViT malignancy probability for the generated images.

    p_cond / p_gt: [N] the same probability for the conditioning image and the
    real follow-up, annotated under their columns so every tile in the figure
    carries a directly comparable number.
    """
    n_rows = len(df)
    n_cols = 2 + len(seed_ids) + (1 if show_error else 0)

    tile = 0.72 if show_probs else 0.60
    label_w = 1.45
    fig_w = label_w + n_cols * tile
    head_h = 0.42
    row_h = tile + (0.20 if show_probs else 0.04)
    fig_h = head_h + n_rows * row_h + 0.30

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=label_w / fig_w, right=0.998,
        bottom=0.02, top=1 - head_h / fig_h,
        wspace=0.06, hspace=0.30 if show_probs else 0.06)

    for r, (_, row) in enumerate(df.iterrows()):
        j = id_to_pos[int(row["sample"])]
        gen_j = gen[:, j]

        tiles = [t0[j], real[j]] + [gen_j[s] for s in range(len(seed_ids))]
        heads = ["Cond $X^{(1)}$", "GT $X^{(2)}$"] + [f"s{s}" for s in seed_ids]

        for c, (img, head) in enumerate(zip(tiles, heads)):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(head, fontsize=7.5, pad=3)
            if show_probs:
                if c == 0:
                    p = None if p_cond is None else p_cond[j]
                elif c == 1:
                    p = None if p_gt is None else p_gt[j]
                else:
                    p = probs[c - 2, j]
                if p is not None:
                    ok = (p >= 0.5) == (row.label == 1)
                    ax.text(0.5, -0.06, f"{p:.2f}", transform=ax.transAxes,
                            ha="center", va="top", fontsize=6.2,
                            color=CORRECT_C if ok else WRONG_C,
                            fontweight="bold" if not ok else "normal")

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
    # T0 and T1_real are identical across seeds, so row 0 is the whole story.
    p_cond = sigmoid(npz["logit_t0"][0] - thr)       # [N]
    p_gt = sigmoid(npz["logit_real"][0] - thr)       # [N]

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

    draw_grid(succ, t0, real, gen, seed_ids, npz["lpips"], probs, id_to_pos,
              os.path.join(args.dir, "success_grid.png"),
              "NAMD_3 — faithful generations, all 20 seeds")

    draw_grid(fail, t0, real, gen, seed_ids, npz["lpips"], probs, id_to_pos,
              os.path.join(args.dir, "failure_grid.png"),
              "NAMD_3 — failure cases, all 20 seeds "
              "(number under every tile = ViT $p$(malignant); red = wrong call)",
              show_error=True, show_probs=True, p_cond=p_cond, p_gt=p_gt)


if __name__ == "__main__":
    main()
