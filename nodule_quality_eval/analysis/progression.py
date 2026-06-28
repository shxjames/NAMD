"""Paper metric: radiomic trajectory cosine.

Mean per-sample cosine similarity between Delta_real = T1_real - T0 and
Delta_gen = T1_gen - T0 in the z-scored 93-D radiomic feature space, over all
samples. Z-normalization uses the real-delta mean/std so the cosine is
meaningful. Samples with any missing feature (a mask too small for radiomic
extraction) are dropped.
"""
import pandas as pd
import numpy as np

rdf = pd.read_csv('results/radiomic_features_per_sample_all.csv')
keys = [c for c in rdf.columns if c.startswith('original_')]

pivot = rdf.pivot_table(index=['sample', 'label', 'seed'],
                        columns='timepoint', values=keys)
pivot.columns = [f'{f}__{t}' for f, t in pivot.columns]
pivot = pivot.reset_index().dropna()

T0 = pivot[[f'{k}__T0' for k in keys]].values
Tr = pivot[[f'{k}__T1_real' for k in keys]].values
Tg = pivot[[f'{k}__T1_gen' for k in keys]].values

dr = Tr - T0
dg = Tg - T0
mu = dr.mean(axis=0, keepdims=True)
sd = dr.std(axis=0, keepdims=True) + 1e-8
drz = (dr - mu) / sd
dgz = (dg - mu) / sd
cs = (drz * dgz).sum(axis=1) / ((np.linalg.norm(drz, axis=1) + 1e-12)
                                * (np.linalg.norm(dgz, axis=1) + 1e-12))

print(f"trajectory_cosine = {cs.mean():.4f}")
