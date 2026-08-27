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

### SUPERSEDED — Results (20 seeds, 128px model, evaluation cropped, 2026-08-26)

> **These numbers are not a valid comparison.** The model was trained at
> 128x128 and only its OUTPUT was cropped to 64x64, so it still conditioned on
> ~4x the context the NAMD arms get. Retraining at 64x64 reverses the ranking:
> traj_cos 0.3407 vs ctclip's 0.3645, losing 0/20 seeds. See the RESULTS
> section at the top. Kept only to document the progression.

`python compare_arms.py` — paper metrics, higher is better:

| arm | traj_cos | n_cos | r_area | r_HU |
| --- | --- | --- | --- | --- |
| **hemexp_final_c64** | **0.3745** | 8573 | 0.689 | 0.729 |
| ctclip_e69 | 0.3644 | 8564 | 0.680 | 0.732 |
| ctclip_e38 | 0.3435 | 8529 | 0.647 | 0.693 |
| alignvae_last | 0.3406 | 8539 | 0.613 | 0.645 |
| pred_clip_e36 | 0.3246 | 8512 | 0.588 | 0.630 |
| hemexp_best_c64 | 0.2923 | 8569 | 0.584 | 0.622 |
| ifn_0624 | 0.2432 | 8473 | 0.711 | 0.468 |

`n_cos` (samples surviving radiomic extraction) now matches the other arms,
which is what makes the row comparable at all — see the field-of-view section.

Two conclusions:

1. **Use the final checkpoint, not the val-loss minimum.** `hemexp_final_c64`
   0.3745 vs `hemexp_best_c64` 0.2923, a 28% relative gap, even though the
   final checkpoint's val loss is 35% HIGHER (0.172 vs 0.127). Selecting on
   val/loss would have cost most of the arm's performance.
2. ~~The arm is competitive on trajectory cosine~~ **WRONG** — this was the
   field-of-view advantage. At 64x64 parity HemExp scores 0.3407 and loses to
   ctclip_e69 on every one of 20 seeds (p = 1.2e-07).

**Caveat that qualifies conclusion 2.** The model conditions on and generates
the full 128x128 field; only evaluation is cropped to 64x64. The NAMD arms
condition on 64x64 only, so HemExp had roughly 4x the conditioning context.
That asymmetry favours HemExp and is not separable from method quality without
retraining on 64x64 crops. Treat the ranking as promising, not settled.

### FIELD OF VIEW: read this before comparing arms

Every arm already in `nodule_quality_eval` was built and evaluated on **64x64**
images. This arm generates **128x128**. They are not the same task, and the
difference is a field of view, not a resolution:

    their_image == our_patch[32:96, 32:96]      # exact, 0 difference

`NLST_with_second_large_cleaned` is a LARGER field-of-view version of the data
those arms were trained on. In their framing the nodule fills the frame; in the
full 128x128 patch it occupies about a quarter of it. Consequences:

- Areas, diameters and every other pixel-unit metric are on different scales.
- MedSAM 2 segments the wider frame far less reliably. On identical real T0
  images it failed on 1060 of ours versus 280 of theirs, so the metrics are
  also averaged over different surviving subsets.

Putting a raw 128x128 result next to their table is meaningless. Use
`crop_triplets.py` to re-frame exported triplets to the common 64x64 window
first:

```bash
python crop_triplets.py --in-dir $RUNS/eval_triplets_final \
                        --out-dir $RUNS/eval_triplets_final_c64
```

That still is not a like-for-like reproduction: the model GENERATED the full
128x128 field and only evaluation is re-framed. Training on 64x64 crops would
be the strict version.

### Comparing against the other NAMD arms

`sample_nlst.py --export-triplets` writes the layout `nodule_quality_eval`
consumes — one `(3, H, W)` uint8 `.npy` per sample per seed under
`seed_<seed>/sample_<idx>_<label>.npy`, channels `[T0, T1_gen, T1_real]` —
the same format `eval_all.py:export_eval_triplets` produces for every other arm:

```bash
python sample_nlst.py --config CONFIGS/config_nlst.json \
    --ckpt $RUNS/diffusion/<stamp>__config_nlst/model-best.pt \
    --split test --seeds 0 1 2 3 4 \
    --export-triplets $RUNS/eval_triplets
```

Note the intensity convention differs from `eval_all.py`: that pipeline holds
images in `[-1, 1]` and converts with `(x+1)/2`, while this one is `[0, 1]`
throughout (HemExp windows CT to `[0, 1]`), so the export uses a plain `x*255`.
Reusing eval_all's formula here would compress every image into the upper half
of the range without erroring.

When reporting downstream classifier numbers, use `avg_score`, not `avg_pred`.

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

## RESULTS — read this section first (2026-08-27)

**The valid comparison is the PARITY column.** Every earlier HemExp number was
measured with an advantage the NAMD arms did not have, and all of them
disappear once matched. Metrics: higher is better except LPIPS / FID / sphere MAE.

| metric | FLAGGED (invalid) | NO-FLAG 128px | **PARITY final** | NAMD_3 / ctclip |
| --- | --- | --- | --- | --- |
| ROC-AUC gt *(ceiling)* | 0.8136 | 0.8136 | 0.8136 | 0.8190 |
| ROC-AUC avg_score | **0.9009** ⚠ above ceiling | 0.7511 | 0.7634 ± 0.021 | **0.8052** |
| AUPRC avg_score | **0.5403** ⚠ above ceiling | 0.3132 | 0.3044 ± 0.030 | **0.3469** |
| LPIPS ↓ | 0.1901 | 0.1868 | **0.1692** | 0.2229 |
| FID ↓ | 67.24 | 67.36 | 88.92 | **82.97** |
| HaarPSI ↑ | 0.3281 | 0.3297 | **0.3276** | 0.3248 |
| traj_cos ↑ | 0.3745 | 0.3681 | 0.3407 | **0.3644** |
| r_area ↑ | 0.689 | 0.686 | 0.643 | **0.680** |
| r_HU ↑ | 0.729 | 0.714 | 0.708 | **0.732** |
| sphere MAE ↓ | 0.1072 | 0.1076 | 0.1043 | 0.1040 |

### Conclusion: at parity HemExp does NOT beat the existing arms

Paired over the same 20 seeds and 450 subjects (ctclip_e69):

| metric | HemExp parity | ctclip | diff | p | seeds won |
| --- | --- | --- | --- | --- | --- |
| traj_cos | 0.3408 ± 0.008 | 0.3645 ± 0.011 | −0.0236 | 1.2e-07 | **0/20** |
| r_area | 0.6434 ± 0.031 | 0.6800 ± 0.017 | −0.0365 | 5.3e-04 | 2/20 |
| r_HU | 0.7083 ± 0.021 | 0.7326 ± 0.013 | −0.0243 | 1.8e-04 | 3/20 |
| sphere MAE | 0.1043 ± 0.004 | 0.1040 ± 0.004 | +0.0004 | 0.73 | 9/20 |

HemExp loses the paper's headline metric on **every** seed. It wins LPIPS
(0.169 vs 0.223) and ties HaarPSI — locally sharper patches — but tracks real
nodule progression less faithfully, and is worse on classification and FID.

An earlier version of this file reported HemExp leading traj_cos at 0.3745.
That was the field-of-view advantage, not the method.

### The three defects, and what each was worth

1. **Oracle label leak.** HemExp's expansion flag maps to the malignancy label.
   Diagnostic: model AUROC 0.9009 ABOVE the 0.8136 real-image ceiling — a
   generator cannot beat the data it imitates, so that is leakage, not skill.
   Worth ~0.15 AUROC. Fixed by `use_expansion_flag_input: false`.
2. **Field of view.** Trained at 128x128 vs the arms' 64x64 (~4x context).
   Worth ~0.027 traj_cos — enough to invert the ranking.
3. **Label-balanced batches.** 50/50 vs the natural ~19%.

### Checkpoint selection: use `final`, never `model-best.pt`

Confirmed in all three configurations. At parity: traj_cos 0.3407 (final) vs
0.2957 (best), FID 88.9 vs 127.3, LPIPS 0.169 vs 0.197 — despite `best` having
the lower val loss. Do not early-stop the diffusion stage on `val/loss`.

## Completed run (2026-08-25)

All under `$RUNS = NAMD/HemExp_NLST_runs/`.

| stage | job | elapsed | output |
| --- | --- | --- | --- |
| autoencoder | 27524301 | 27 min | `autoencoder/checkpoints/checkpoint_{best,last}.pth` |
| diffusion | 27524366 | 37 min | `diffusion/20260825_213513__config_nlst/model-{best,1..30}.pt` |
| sampling | 27524455 | 21 min | `eval_triplets_{best,final}/`, `samples_{best,final}/` |

Test-split generation: 450 subjects x 5 seeds = 2250 triplets per checkpoint,
verified readable by `nodule_quality_eval.load_triplets.iter_npy_triplets`
(labels parse to 1191 benign / 159 malignant per 3 seeds, matching the split).

Next step — not yet run, needs MedSAM 2 + pyradiomics:

```bash
python nodule_quality_eval/run_evaluation.py \
    --input-dir $RUNS/eval_triplets_best --input-format npy \
    --cache-dir cache_hemexp_best --out-tag hemexp_best
# and again with eval_triplets_final / hemexp_final
```

**Disk**: the run directory is ~52 GB, of which 48 GB is 31 diffusion
checkpoints at 1.3 GB each (model + optimizer + EMA). Only `model-best.pt` and
`model-30.pt` are needed once the arm is evaluated; the intermediates are
prunable. `save_every` is 5 epochs, which is finer than this arm needs.

## Observed run behaviour (150 epochs each)

**Stage 1 (autoencoder)** converged cleanly: val recon 5.5e-3 at epoch 1 to
3.1e-4 by epoch 114, flat thereafter. Perceptual loss at epoch 58 caused no
visible disruption. Adversarial at epoch 115 raised val recon to a stable
~4.1e-4 — the intended fidelity-for-sharpness trade, not a regression.
`checkpoint_best.pth` holds epoch 114; `checkpoint_last.pth` the finetuned model.

**Stage 2 (diffusion)** val loss bottomed at **0.1275 around epoch 36** and rose
steadily afterwards (0.1349 by epoch 63). The evaluation is deterministic, so
that is a real trend, not sampling noise — 900 training pairs, one per patient,
is a small corpus and the model starts fitting it.

Do NOT conclude from this alone that ~36 epochs is the right budget. On this
project `val/loss` has repeatedly been a poor checkpoint selector for the *UNet*
stages, with late checkpoints outperforming the val-loss minimum; the VAE stage
is the opposite, which is why stage 1 uses lowest-val and it works. Evaluate the
arm at **both** `model-best.pt` (the val minimum) and the final checkpoint, and
let the generation metrics decide rather than the loss curve.

## Notes before a real run
- **Oracle flag.** Training conditions on the ground-truth malignancy label. That
  is intended for counterfactual simulation, but it makes any downstream
  "synthetic data improves a classifier" claim circular unless the generated set
  is built with an override or a predicted probability instead.
- **Latent stats.** The placeholders in `config_nlst.json` are identity whitening.
  Training will run with them, but badly scaled — run step 2 first.
