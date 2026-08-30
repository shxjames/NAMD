"""Print the four paper nodule-property metrics for one arm's nodule_quality_eval
output dir: r_area, r_HU (compare_to_t0), sphericity MAE (stratified), and
radiomic trajectory cosine (progression). Same math as the scripts in
nodule_quality_eval/analysis/, but path-parameterized so multiple arms coexist.

Usage: python scripts/nodule_report.py <results_dir> [tag]
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

results_dir = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "all"

df = pd.read_csv(f"{results_dir}/per_sample_summary_{tag}.csv")


def pearson(real, pred):
    d = pd.DataFrame({"r": real, "p": pred}).dropna()
    return pearsonr(d["r"], d["p"])[0]


print(f"n_rows = {len(df)}")
print(f"r_area = {pearson(df['area_T1r'], df['area_T1g']):.3f}")
print(f"r_HU   = {pearson(df['mean_T1r'], df['mean_T1g']):.3f}")
mae = (df["shape_Sphericity_T1r"] - df["shape_Sphericity_T1g"]).abs().mean()
print(f"sphericity_MAE = {mae:.4f}")

rdf = pd.read_csv(f"{results_dir}/radiomic_features_per_sample_{tag}.csv")
keys = [c for c in rdf.columns if c.startswith("original_")]
pivot = rdf.pivot_table(index=["sample", "label", "seed"],
                        columns="timepoint", values=keys)
pivot.columns = [f"{f}__{t}" for f, t in pivot.columns]
pivot = pivot.reset_index().dropna()

T0 = pivot[[f"{k}__T0" for k in keys]].values
Tr = pivot[[f"{k}__T1_real" for k in keys]].values
Tg = pivot[[f"{k}__T1_gen" for k in keys]].values

dr = Tr - T0
dg = Tg - T0
mu = dr.mean(axis=0, keepdims=True)
sd = dr.std(axis=0, keepdims=True) + 1e-8
drz = (dr - mu) / sd
dgz = (dg - mu) / sd
cs = (drz * dgz).sum(axis=1) / ((np.linalg.norm(drz, axis=1) + 1e-12)
                                * (np.linalg.norm(dgz, axis=1) + 1e-12))
print(f"trajectory_cosine = {cs.mean():.4f}  (n={len(cs)})")
