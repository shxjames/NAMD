#!/usr/bin/env python3
"""Score every nodule_quality_eval results directory on the paper metrics.

The upstream analysis scripts (analysis/progression.py, analysis/compare_to_t0.py)
hardcode `results/...csv` and take no arguments, so they can only score one arm
at a time. This reproduces their computations verbatim and applies them across
all `results_*` directories, so arms can be put side by side.

Metrics (all reproduced exactly from the upstream scripts):
  trajectory_cosine  mean per-sample cosine between Delta_real and Delta_gen in
                     the z-scored 93-D radiomic space (z-scored on the real
                     deltas). Higher is better; 1.0 is a perfect trajectory.
  r_area             Pearson r between area_T1r and area_T1g. Higher better.
  r_HU               Pearson r between mean_T1r and mean_T1g. Higher better.

    python compare_arms.py [--results-root DIR] [--only SUBSTR]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

DEFAULT_ROOT = "/scratch/liyues_root/liyues/shxjames/NAMD/nodule_quality_eval"


def trajectory_cosine(radiomics_csv):
    """Verbatim reproduction of analysis/progression.py."""
    rdf = pd.read_csv(radiomics_csv)
    keys = [c for c in rdf.columns if c.startswith("original_")]
    if not keys:
        return np.nan, 0

    pivot = rdf.pivot_table(index=["sample", "label", "seed"],
                            columns="timepoint", values=keys)
    pivot.columns = [f"{f}__{t}" for f, t in pivot.columns]
    pivot = pivot.reset_index().dropna()
    if pivot.empty:
        return np.nan, 0

    T0 = pivot[[f"{k}__T0" for k in keys]].values
    Tr = pivot[[f"{k}__T1_real" for k in keys]].values
    Tg = pivot[[f"{k}__T1_gen" for k in keys]].values

    dr, dg = Tr - T0, Tg - T0
    mu = dr.mean(axis=0, keepdims=True)
    sd = dr.std(axis=0, keepdims=True) + 1e-8
    drz, dgz = (dr - mu) / sd, (dg - mu) / sd

    cs = (drz * dgz).sum(axis=1) / (
        (np.linalg.norm(drz, axis=1) + 1e-12) * (np.linalg.norm(dgz, axis=1) + 1e-12)
    )
    return float(cs.mean()), len(cs)


def correlations(summary_csv):
    """Verbatim reproduction of analysis/compare_to_t0.py."""
    df = pd.read_csv(summary_csv)

    def pearson(real, pred):
        d = pd.DataFrame({"r": real, "p": pred}).dropna()
        if len(d) < 3:
            return np.nan
        return pearsonr(d["r"], d["p"])[0]

    return (pearson(df["area_T1r"], df["area_T1g"]),
            pearson(df["mean_T1r"], df["mean_T1g"]),
            df["seed"].nunique(),
            df["sample"].nunique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=DEFAULT_ROOT)
    ap.add_argument("--only", default=None, help="Only arms whose name contains this")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.results_root, "results_*/"))):
        arm = os.path.basename(d.rstrip("/")).replace("results_", "")
        if args.only and args.only not in arm:
            continue

        summary = glob.glob(os.path.join(d, "per_sample_summary_*.csv"))
        radiomics = glob.glob(os.path.join(d, "radiomic_features_per_sample_*.csv"))
        if not summary:
            continue

        r_area, r_hu, nseed, nsamp = correlations(summary[0])
        cos, ncos = trajectory_cosine(radiomics[0]) if radiomics else (np.nan, 0)
        rows.append((arm, nseed, nsamp, cos, ncos, r_area, r_hu))

    if not rows:
        print("no results directories found")
        return

    w = max(len(r[0]) for r in rows) + 2
    print(f"{'arm':<{w}}{'seeds':>6}{'samples':>9}{'traj_cos':>10}{'n_cos':>8}{'r_area':>9}{'r_HU':>8}")
    print("-" * (w + 50))
    # Rank by the headline paper metric.
    for arm, nseed, nsamp, cos, ncos, r_area, r_hu in sorted(
        rows, key=lambda r: (-(r[3] if r[3] == r[3] else -9))
    ):
        c = f"{cos:.4f}" if cos == cos else "-"
        a = f"{r_area:.3f}" if r_area == r_area else "-"
        h = f"{r_hu:.3f}" if r_hu == r_hu else "-"
        print(f"{arm:<{w}}{nseed:>6}{nsamp:>9}{c:>10}{ncos:>8}{a:>9}{h:>8}")

    print("\ntraj_cos / r_area / r_HU: higher is better.")
    print("Arms with differing seed counts are not directly comparable.")


if __name__ == "__main__":
    main()
