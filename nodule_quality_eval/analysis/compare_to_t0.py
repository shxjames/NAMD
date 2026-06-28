"""Paper metrics: r_area and r_HU.

Pearson r between the real and generated follow-up, over all samples:
  r_area : area_T1r   vs area_T1g
  r_HU   : mean_T1r   vs mean_T1g
"""
import pandas as pd
from scipy.stats import pearsonr

df = pd.read_csv('results/per_sample_summary_all.csv')


def pearson(real, pred):
    d = pd.DataFrame({'r': real, 'p': pred}).dropna()
    return pearsonr(d['r'], d['p'])[0]


print(f"r_area = {pearson(df['area_T1r'], df['area_T1g']):.3f}")
print(f"r_HU   = {pearson(df['mean_T1r'], df['mean_T1g']):.3f}")
