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

### On SLURM (the way these were actually run)

```bash
AE=$(sbatch --parsable slurm/hemexp_nlst_ae.sh)
sbatch --dependency=afterok:$AE slurm/hemexp_nlst_diffusion.sh
```

Stage 2 recomputes the latent statistics for the stage-1 checkpoint and writes
them into `CONFIGS/config_nlst.json` itself, so there is no manual step between
the jobs. Outputs land in `$RUNS = NAMD/HemExp_NLST_runs/`.

### By hand

```bash
conda activate /scratch/liyues_root/liyues/shxjames/envs/HEMEXP
RUNS=/scratch/liyues_root/liyues/shxjames/NAMD/HemExp_NLST_runs

# 1. first stage: MultiHeadAEKL, in_channels=1, head_names=("ct",), 128px
python train_autoencoder_nlst.py --config CONFIGS/config_ae_nlst.json

# 2. latent statistics for THIS autoencoder, written into the diffusion config
python compute_latent_stats.py \
    --ae-ckpt $RUNS/autoencoder/checkpoints/checkpoint_best.pth \
    --update-config CONFIGS/config_nlst.json

# 3. diffusion
python train_diffusion.py CONFIGS/config_nlst.json

# 4. inference -- model-best.pt is the lowest val/loss checkpoint
python sample_nlst.py --config CONFIGS/config_nlst.json \
    --ckpt $RUNS/diffusion/<stamp>__config_nlst/model-best.pt --split test
```

The env is `envs/HEMEXP` — the NAMD env cloned, plus `monai==1.5.2`,
`ema-pytorch`, and `nibabel`. Building `environment.yml` fresh also works but
re-pins the whole torch/CUDA stack. Note the entrypoints prepend the sibling
`denoising-diffusion-pytorch/` to `sys.path` so the adapted fork wins over any
pip-installed copy; don't rely on `pip install -e` pointing at the right
checkout.

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

## Training budget and loss tracking

Both stages are configured in **epochs**, not raw steps — an epoch's length is
resolved once the sampler exists and is printed at startup.

| | steps/epoch @ batch 16 | default | total steps |
| --- | --- | --- | --- |
| autoencoder | 112 (1800 images) | `total_epochs: 150` | 16800 |
| diffusion | 91 (balanced sampler) | `epochs: 150` | 13650 |

Set `total_steps` / `steps` instead if you want to pin the raw count; the epoch
field takes precedence when both are present.

Logged to wandb every step: `train/loss` and `epoch`. Every epoch:

- **diffusion** — `val/loss`, the *training objective* evaluated on a plain
  sequential pass of the val split. It is made deterministic on purpose:
  timesteps and noise come from a fixed seed (`val_loss_seed`) and
  classifier-free dropout is disabled, so successive points differ only because
  the weights changed. Without that, resampling noise swamps the signal and the
  curve cannot be read point to point. Best-so-far is written to `model-best.pt`.
- **autoencoder** — `val/recon`, `val/kl`, `val/loss`; best to `checkpoint_best.pth`.

The expensive sampling-based image metrics (SSIM, and Dice/volume on the
hemorrhage path) still need a full reverse diffusion per batch, so they stay on
the slower `sample_metrics_every` cadence — 25 epochs by default — while
checkpoints are written every 5.

Note the balanced sampler oversamples the 175 positives roughly 4×, so a
diffusion epoch is `2 × 725 = 1450` samples rather than 900.

## Notes before a real run
- **Oracle flag.** Training conditions on the ground-truth malignancy label. That
  is intended for counterfactual simulation, but it makes any downstream
  "synthetic data improves a classifier" claim circular unless the generated set
  is built with an override or a predicted probability instead.
- **Latent stats.** The placeholders in `config_nlst.json` are identity whitening.
  Training will run with them, but badly scaled — run step 2 first.
