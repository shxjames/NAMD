# Nodule Quality Evaluation

> Image-content quality evaluation for generated follow-up lung nodule patches.

For every triplet `(T0, T1_real, T1_gen)` of 2D nodule-centered patches, the
pipeline segments each timepoint, extracts 8 shape + 93 IBSI radiomic features,
and reports how well the generated follow-up matches the real one on **size**,
**HU intensity**, **boundary regularity**, and **radiomic trajectory**.

Segmentation is anchored on `T1_real`: MedSAM 2 segments `T1_real` from a
centered box, `T1_gen` from a box derived from the `T1_real` mask, and `T0` by
Otsu thresholding inside the dilated `T1_real` mask (avoids grabbing chest wall
on faint baselines).

---

## Layout

```
nodule_quality_eval/
├── run_evaluation.py      # main entry point (segment + extract + per-sample CSVs)
├── load_triplets.py       # image / npy loaders
├── segmentation.py        # MedSAM 2 wrapper + post-processing
├── metrics.py             # 8 shape + 93 IBSI features (PyRadiomics)
├── metrics_skimage.py     # lightweight first-order + GLCM fallback (no PyRadiomics)
└── analysis/              # post-hoc comparisons (read the CSVs, print metrics)
    ├── compare_to_t0.py
    ├── progression.py
    └── stratified.py
```

---

## Install

```bash
pip install -r requirements.txt
pip install git+https://github.com/AIM-Harvard/pyradiomics.git

git clone https://github.com/bowang-lab/MedSAM2.git
cd MedSAM2 && pip install -e ".[dev]"
```

The MedSAM 2 checkpoint (~750 MB) auto-downloads from HuggingFace
(`wanglab/MedSAM2`) on first run and caches under `checkpoints/`. If PyRadiomics
won't install, pass `--metrics-backend skimage` (first-order + GLCM, no IBSI
panel — note the `analysis/` scripts expect the PyRadiomics columns).

---

## Input

**Image** (`--input-format image`, default) — one folder per sample, three
grayscale images (`.png/.jpg/.tif/...`):

```
input/<sample_id>/{T0,T1_real,T1_gen}.png        # flat
input/<seed>/<sample_id>/{T0,T1_real,T1_gen}.png # multi-seed
```

**Stacked npy** (`--input-format npy`) — one `(3, H, W)` uint8 array per sample,
`data[0]=T0`, `data[1]=T1_gen`, `data[2]=T1_real`:

```
input/<seed>/<sample_id>.npy      # multi-seed
input/<sample_id>.npy             # flat
```

Append the binary label as the last token of the sample folder/file name for
stratified metrics: `sample_001_0` (benign), `sample_397_1` (malignant).
Unlabeled samples get `label = -1` and are still processed.

---

## Run

```bash
# images (default)
python run_evaluation.py --input-dir input --cache-dir cache --out-dir results --out-tag run1

# stacked npy (use this for outputs from eval_all.py)
python run_evaluation.py --input-dir input_npy --input-format npy --out-tag run1

# no PyRadiomics
python run_evaluation.py --input-dir input --metrics-backend skimage
```

The first run segments and extracts features (slow); every mask, feature JSON,
and red-outline QC PNG is cached under `cache/`, so re-runs are near-instant.

---

## Outputs (`--out-dir`)

| File | Row | Description |
|------|-----|-------------|
| `per_sample_summary_<tag>.csv` | (sample, seed) | area / intensity / shape summary (~55 cols) |
| `radiomic_features_per_sample_<tag>.csv` | (sample, seed, timepoint) | full 93 radiomic features in long form |
| `trajectory_consistency_<tag>.csv` | feature + header block | per-feature Spearman ρ / p / MAE for `Δ_real` vs `Δ_gen`, plus global trajectory cosine |

`trajectory_consistency_<tag>.csv` is built from the feature set common to
**every** ROI, so it is written only when all masks were large enough for
radiomic extraction. If any mask is too small (common for faint `T0`), the file
is skipped with a notice — read the trajectory metrics from
`analysis/progression.py` instead, which drops those samples.

Cache layout: `cache/{T0,T1_real}/{masks,features,overlays}/<id>.*` and
`cache/T1_gen/<seed>/{masks,features,overlays}/<id>.*`.

---

## Post-hoc analyses

The `analysis/` scripts print the four reported metrics from the per-sample
CSVs that `run_evaluation.py` writes. Run them after the CSVs exist (they assume
`--out-tag all`; edit the paths at the top otherwise).

```bash
python analysis/compare_to_t0.py   # r_area, r_HU  (Pearson r, T1_real vs T1_gen)
python analysis/stratified.py      # sphericity MAE
python analysis/progression.py     # radiomic trajectory cosine
```

---

## Reproducing the paper numbers

450 unique nodules × 20 seeds = 9,000 triplets, 64×64 grayscale patches,
label `0` = benign (samples 0–396), label `1` = malignant (samples 397–449).
Place the data in either layout, run `run_evaluation.py`, then the three
analysis scripts.
