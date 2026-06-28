"""Paper metric: sphericity MAE.

Mean absolute error of PyRadiomics shape2D sphericity between the real and
generated follow-up, over all samples.
"""
import pandas as pd

df = pd.read_csv('results/per_sample_summary_all.csv')

mae = (df['shape_Sphericity_T1r'] - df['shape_Sphericity_T1g']).abs().mean()
print(f"sphericity_MAE = {mae:.4f}")
