# HemExp on NLST (NAMD task)

HemExp's formulation — *baseline image + tabular clinical variables + a binary
oracle flag → follow-up image*, as a latent diffusion model predicting a latent
**residual** — transfers to the NAMD/NLST nodule-progression task with no change
to the diffusion architecture. Only the data path and the two dimensions that
were hard-coded to the hemorrhage problem were touched.

## Mapping

| HemExp (hemorrhage) | NLST (this arm) |
| --- | --- |
| baseline CT slice, NIfTI | year-1 nodule patch (left half of the 256×128 jpg) |
| follow-up CT slice | year-2 patch (right half) — the prediction target |
| `iph` / `ivh` mask channels | none: `n_aux == 0`, AE is single-channel |
| 7 clinical variables (CSV) | 13 first-year biomarker + EHR fields (`demo/*.npy`) |
| `expansion_flag` (oracle 0/1) | malignancy label |
| 2.5D context slices (`context_mm_step`) | `context_slices: 1` — patches are single 2D slices |
| 384×384 → 96×96 latent | 128×128 → 32×32 latent |

## Conditioning vector

**First-year features only.** `nlst_dataset.py` keeps only the first half of each
`demo/*.npy` and asserts the label is identical across both years (it is, for all
1457 subjects), so the flag carries no follow-up information either.

Default selection (`clinical_dim: 13`):

- **Imaging biomarkers** — `SCT_PRE_ATT`, `SCT_EPI_LOC`, `SCT_LONG_DIA`, `SCT_PERP_DIA`, `SCT_MARGINS`
- **EHR / demographics** — `age`, `diagemph`, `gender`, `famfather`, `fammother`, `fambrother`, `famsister`, `famchild`

Excluded by default: `unique_ids` and `year` (identifiers, not clinical — and
`unique_ids` ≈ 1e5 would dominate the first `Linear`), and `label`, which reaches
the model through the flag embedding rather than the clinical vector.

For literal parity with NAMD's `LatentDiffusionCond_EHR` (`EHREncoder(in_dim=16)`,
which feeds all 16 raw values including the identifiers), set:

```json
"excluded_features": [], "normalize_clinical": false   // data
"clinical_dim": 16                                     // model
```

Features are z-scored using **train-split** statistics only, reused verbatim for
val/test.

## Running

Three stages. The autoencoder must be retrained — HemExp's released weights are
3-channel 384px hemorrhage CT and do not transfer.

```bash
conda env create -f environment.yml && conda activate hemexp_demo

# 1. first stage: MultiHeadAEKL, in_channels=1, head_names=("ct",), 128px
python train_autoencoder_nlst.py --config CONFIGS/config_ae_nlst.json

# 2. latent statistics for THIS autoencoder -> paste into config_nlst.json's "ae"
python compute_latent_stats.py \
    --ae-ckpt output/autoencoder_nlst/checkpoints/checkpoint_step_20000.pth

# 3. diffusion
python train_diffusion.py CONFIGS/config_nlst.json

# 4. inference (test split)
python sample_nlst.py --config CONFIGS/config_nlst.json \
    --ckpt results/<stamp>__config_nlst/model-20.pt --split test
```

Counterfactuals, HemExp's main clinical affordance:

```bash
python sample_nlst.py ... --flag-override 1   # generate every subject as malignant
python sample_nlst.py ... --cond-scale 0.0    # clinical conditioning off (learned null)
python sample_nlst.py ... --cond-scale 2.0    # amplified clinical effect
```

## What changed

New files: `nlst_dataset.py`, `train_autoencoder_nlst.py`, `compute_latent_stats.py`,
`sample_nlst.py`, `CONFIGS/config_nlst.json`, `CONFIGS/config_ae_nlst.json`.

`denoising_diffusion_pytorch.py`, kept backward compatible with the hemorrhage path:

- `Unet(clinical_dim=…)` — was a hard-coded `nn.Linear(7, …)`; now validated against
  the incoming vector so a mis-sized selection fails loudly instead of silently.
- `MultiHeadAEKL(head_names=…)` — variable number of 1×1 output heads. The
  3-head state dict is byte-identical, so existing checkpoints still load.
- Segmentation channels generalised to an "aux" tensor `[B, n_aux, K, H, W]` that
  may be empty. Both datasets now return the same 6-tuple; Dice / volume metrics
  and the seg PNG panels are skipped when `n_aux == 0`.
- `dataset_flags()` / `_balanced_val_indices()` replace the direct
  `metadata_expansion` lookups, so the balanced sampler and the validation
  selection work off either dataset.
- `assert len(self.ds) >= 100` → one batch (NLST train is 900 pairs, not CT slices).

`sampling.py` is untouched and still serves the hemorrhage path.

## Notes before a real run

- **Scale.** 900 train pairs, one sample per patient — against HemExp's per-slice
  corpus across many patients. `steps: 20000` at batch 16 is already ~350 epochs;
  watch for overfitting and consider stronger augmentation or fewer steps.
- **Class balance.** `ExpansionBalancedSampler` forces 50/50 batches, oversampling
  the 175 positives roughly 4×. An epoch is `2 × 725` samples.
- **Oracle flag.** Training conditions on the ground-truth malignancy label. That
  is intended for counterfactual simulation, but it makes any downstream
  "synthetic data improves a classifier" claim circular unless the generated set
  is built with an override or a predicted probability instead.
- **Latent stats.** The placeholders in `config_nlst.json` are identity whitening.
  Training will run with them, but badly scaled — run step 2 first.
