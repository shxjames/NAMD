# NAMD on SD2.1 — plan and run log

The NAMD two-stage recipe with the SD1.5 backbone replaced by SD2.1-base.
Independent of the SD2.1 *baseline* (`worktree-sd21-baseline`); that arm is the
comparison point, not an input to this one.

## Branch

`worktree-sd21-namd`, forked from **`NAMD-eos` @ 18d4d12** — deliberately not
from `NAMD-medgemma`.

`NAMD-eos` is `main` plus exactly three things this arm wants:

| Commit | What |
|---|---|
| `36e35bc` | Terminate conditioning prompts with EOS |
| `18d4d12` | run.py `--set` / `--save_top_k` / torch.load shim / `ckpt_path=` resume, and the `log_images` N clamp |

`NAMD-medgemma` would have brought `strict_load`, `expand_concat_channels`,
`train_contexts` and the empty-prompt stage 1 — machinery built to force an
*exact* stage-1 → stage-2 load. This arm wants the opposite: the original NAMD
handoff, where stage 1 keeps the backbone's own text tower and the conditional
stage swaps in the LLM. `grep -c` for those three names on this branch returns 0.

The `sd21/` package is grafted verbatim from `worktree-sd21-baseline`; no file
in `sd21/` or `ldm/` is modified. Configs reach the SD2.1 code via `sd21.*`
targets.

## Design

| | stage 1 (uncond) | stage 2 (cond) |
|---|---|---|
| class | `LatentDiffusionCustom` | `LatentDiffusionCond_LLM` |
| text tower | OpenCLIP ViT-H (SD2.1's own) | MedGemma 1.5 4B, nf4 |
| `context_dim` | 1024 | 2560 |
| `conditioning_key` | crossattn | hybrid |
| `in_channels` | 4 | 8 (prior-year concat) |
| lr / bs / epochs | 1e-4 / 16 / 50 | 2e-5 / 8 / 100 |
| VAE | contrastive-0211, SF 0.519096 | same, arrives in the stage-1 ckpt |

**First stage.** `checkpoints/autoencoder/v1.5-autoencoder-contrastive-0211.ckpt`
— the NAMD aligned VAE (L_align + L_pred, feature 1.0 / label 0.2). Scale factor
**0.519096**, corroborated by `configs/ldm/v1.5-conditional-unet-ctclip.yaml`,
which pairs that exact value with the UNet trained on this VAE. It is repeated in
both stage configs because `scale_factor` is a config value, not a state_dict
entry — no checkpoint load can catch a mismatch.

The VAE's own training config used `resolution: 64, dropout: 0.2` against the
`256 / 0.0` in the LDM configs. Neither changes the parameter set:
`attn_resolutions` is empty so `resolution` never adds a block, and dropout has
no parameters and the VAE is frozen in eval here.

## What the stage-2 load does

`LatentDiffusionCond_LLM.init_from_ckpt` walks the stage-1 state_dict against the
UNet and appends every shape-mismatched key to `ignore_keys` before loading, so
the divergences are dropped silently rather than raising. Dry-run
(`scripts/sd21_namd/` meta-device check, before any training):

```
stage-1 UNet keys: 687   stage-2 UNet keys: 687
  transfer cleanly : 654
  dropped (shape)  : 33     1 input conv (4->8) + 16 attn2.to_k + 16 attn2.to_v
  absent in stage 1: 0
contexts transfers? True
fraction of stage-2 UNet weights initialised from stage 1: 92.9%
```

So **stage 1 pretrains the backbone, not the conditioning pathway** — self-attn,
resblocks and time embedding carry over; all 32 cross-attention matrices start
fresh at stage 2. That is inherent to a CLIP→LLM handoff and is exactly what the
v1.5 NAMD arm does, which is what keeps the comparison honest.

`contexts` transfers because stage 1 leaves `txt_embed_dim` at the sd21 default
of 2560, so the soft prompts are already the right shape. They are dead weight in
stage 1 (the CLIP path never reads them) but they survive the load instead of
being re-initialised.

Stage 2 also sets `ignore_keys: ["cond_stage_model", "model_ema"]`. The shape
filter only checks against the UNet, so a `cond_stage_model` whose shapes moved
would reach `load_state_dict` and raise on size mismatch. Dropping those keys is
free — `FrozenOpenCLIPHFEmbedder` reloads from HF in `__init__`, and nothing
reads it (conditioning goes through `self.text_encoder`).

## LR schedules

`LambdaWarmUpCosineScheduler2` takes **absolute step counts**, and this repo has
hit both failure modes:

- too small — `find_in_interval` returns `None` past the last cycle and the
  caller indexes with it. Crashed the SD2.1 baseline uncond run at step 100k.
- too large — `t = (n - warmup) / (cycle_length - warmup)` stays ~0, so the LR
  sits flat at `base_lr` and never decays. The baseline's `1e7` did exactly this.

Both launchers therefore derive `steps/epoch` from the run's real
`len(train_dataloader())` and set `cycle_lengths = total + one epoch`. The probe
must run on the compute node — the dataset's worker pool OOMs on a login node.

Values baked into the configs are the expected ones (~2983 steps/ep at bs16;
~460 at bs8) and are overridden at launch.

## Running it

```bash
sbatch scripts/sd21_namd/uncond.sbatch

# then, per the select-late rule, seed stage 2 from stage 1's last.ckpt
sbatch --export=ALL,UNCOND_CKPT=checkpoints/sd21-namd-uncond-bs16-lr1.0e-04/last.ckpt \
       scripts/sd21_namd/cond.sbatch
```

Both request `--mem=100g`. 80g is **not** enough margin: stage 1 completed with
MaxRSS 83,851,016 K against an 80 G request — 99.96% of the limit, surviving by
~34 MB. The pooled Luna25 + DLCS + Luna16 + NLST loader at bs16 is far heavier
than the 50.3 GB peak measured on earlier arms. `save_top_k` is 3; `run.py` sets
`save_last=True` unconditionally, so `last.ckpt` is always there for the
select-late rule.

Expect roughly 21 h stage 1 and 7–9 h stage 2, extrapolating from the SD1.5
MedGemma arm, plus ~155 GB of checkpoints.

## Run log

| Stage | Result |
|---|---|
| configs + launchers | UNet build and stage-1→2 transfer verified on meta device before submitting |
| stage 1 (uncond) | **COMPLETED** — job 27510622, 15:39:24 on vvh-l40s (lh1200), 18.8 min/epoch, 0 errors |
| stage 2 smoke | **COMPLETED** — job 27514619, 1 epoch in 11:36, val/loss 0.281, 0 errors |
| stage 2 (cond) | **COMPLETED** — job 27514675, 7:37:25, 4.6 min/epoch, 0 errors |

### Stage 2 final (job 27514675 — constant LR 2e-5, bs16, 100 epochs)

Superseded job 27514632 (bs8, cosine schedule), cancelled at epoch ~19 when the
recipe changed to a flat LR.

Handoff: `checkpoints/sd21-namd-cond-bs16-lr2.0e-05/last.ckpt` (epoch 99, 15.6 GB).
Lowest-val reference: `epoch=38-val/loss=0.229.ckpt`.

| epoch | train/loss | val/loss |
|---|---|---|
| 5 | — | 0.275 |
| 14–32 | — | 0.253 (plateau) |
| **38** | — | **0.229**  ← best |
| 45 | — | 0.230 |
| 69 | 0.192 | 0.279 |
| 79 | 0.164 | 0.374 |
| 89 | 0.141 | 0.365 |
| **99** | **0.124** | **0.471** |

The `lr` column in `logs/sd21-namd-cond-bs16-lr2.0e-05/loss_log.csv` reads
`0.00002000` at every epoch — the flat LR confirmed end to end.

val/loss bottomed at epoch 38 and then climbed steadily to 0.471 while train/loss
kept falling to 0.124. **This is expected, not a failure.** The SD2.1 baseline's
conditional stage did the same thing (val/loss 0.36–0.42 by epochs 50–69) and its
ep70 `last.ckpt` still beat the lowest-val ep21 checkpoint on AUROC, AUPRC, LPIPS
and all four nodule-property metrics. Hence the select-late handoff. Keeping
`epoch=38` gives the same late-vs-lowest contrast for this arm.

Dropping the warmup along with the schedule produced **no opening loss spike** —
val/loss descended smoothly from epoch 1 despite the freshly-initialised
cross-attention taking full-rate updates from step 1.

**Memory, again: 100g is marginal at bs16.** MaxRSS 101,954,424 K against the
100 G request — 97% of the limit. Stage 1 at bs16 hit 99.96% of 80 G. Anything
rerun at this batch size wants ~120g.

## Eval — eval_all.py, last.ckpt (job 27516100, 10:23)

20 seeds, guidance 1.0, 50 steps, batch 64, test split (450 subjects).
`results/sd21-namd-cond-bs16-lr2.0e-05/eval_all_1.0_last.json`
(copied aside immediately — eval_all.py names its JSON by guidance only, so the
next run at guidance 1.0 overwrites `eval_all_1.0.json`).

| arm / ckpt | AUROC per-seed | AUPRC per-seed | LPIPS | FID |
|---|---|---|---|---|
| **NAMD-on-SD2.1, ep99 (last)** | **0.783 ± 0.017** | **0.352 ± 0.040** | 0.218 | **89.6 ± 0.7** |
| SD2.1 baseline, ep70 (last) | 0.787 ± 0.013 | 0.351 ± 0.031 | 0.189 | 105.7 ± 0.9 |
| SD2.1 baseline, ep21 (lowest val) | 0.727 ± 0.029 | 0.280 ± 0.034 | 0.211 | 88.0 ± 0.9 |
| SD1.5 CLIP + PRED-VAE, ep40 | 0.764 | 0.293 | 0.208 | 77.3 |

Reference points from this run: `auc_gt` (real images) **0.8136**, `auc_cond`
(prior-year image) **0.7417**. So the arm recovers most of the gap between the
prior-year scan and the real follow-up.

**Report `avg_score`, never `avg_pred`.** This run's `auc_model_avg_pred` is
0.8234, which is *above* the 0.8136 real-image ceiling — it averages predictions
over 20 seeds, so it is a 20-sample ensemble rather than a single generated
image, and it is not a valid headline number. Same for
`auprc_model_avg_pred` (0.4023 vs 0.3896 gt).

Reading it: against the SD2.1 baseline's own late checkpoint, AUROC (0.783 vs
0.787) and AUPRC (0.352 vs 0.351) are statistically indistinguishable — the
AUROC gap is a quarter of one standard deviation. FID is clearly better
(89.6 vs 105.7); LPIPS is clearly worse (0.218 vs 0.189). Against the SD1.5 CLIP
arm, AUROC and AUPRC are both better.

Also emitted: predictive variance across seeds (mean 0.0784), and
`error_vs_variance_pearson_r` = **0.393** — the model's seed-to-seed disagreement
correlates with its actual error, which is the useful calibration signal.

Triplets: `nodule_quality_eval/triplets/sd21-namd-cond-bs16-lr2.0e-05-last/`,
20 seed dirs, 612 MB — ready for the nodule-property eval.

### Stage 2 detail

Probe derives **450 steps/epoch** at bs8 → total 45000, warmup 2250, cycle 45450.

The handoff landed exactly as the meta-device dry-run predicted:

```
Restored from checkpoints/sd21-namd-uncond-bs16-lr1.0e-04/last.ckpt
  with 421 missing and 0 unexpected keys
```

421 = 388 `cond_stage_model` (dropped via `ignore_keys`) + the 33 shape-mismatched
UNet keys (1 input conv 4→8, 16 `attn2.to_k`, 16 `attn2.to_v`). `first_stage_model`
shows 0 missing, confirming the VAE rode in on the stage-1 checkpoint.

**Two failures got here, both worth recording:**

1. Job 27514617, FAILED in 36 s. `steps_per_epoch.py` called `dm.setup()` with no
   retriever. The conditional dataset builds a prompt per subject and needs one
   (`run.py` passes the model). Every subject raised, the dataset's per-subject
   `except ... continue` swallowed all of them, and `np.stack` got an empty list —
   surfacing only as `need at least one array to stack`. Fixed with a tokenizer
   stub plus a loud empty-split check. Stage 1 was immune: `model_type:
   unconditional` never constructs prompts.
2. Job 27514618, FAILED at 2:27, at the first optimizer step.
   `AttributeError: 'NoneType' object has no attribute 'group_separator'` —
   Lightning's `_track_grad_norm` reads `trainer.logger.group_separator`
   unconditionally, so `track_grad_norm: 2` cannot survive a logger-less run. It
   came from the v1.5 template, whose runs used wandb. `worktree-sd21-baseline`
   had already fixed this in `96df2c6`; forking from `NAMD-eos` meant not
   inheriting it.

Both only surfaced once training actually started, so stage 2 now smoke-tests
first: `EPOCHS=1 PROJECT=sd21-namd-cond-smoke` writes to its own checkpoint dir,
costs ~12 min, and would have caught either.

### Stage 1 detail (job 27510622, 2026-08-20 → 08-21)

Probe derived **2969 steps/epoch** (warmup 4000, cycle 151419) against a 2983
estimate. Load was clean: the aligned VAE restored with 0 missing / 0 unexpected,
and the SD2.1 init's 638 missing keys were exactly the ignored ones —
388 `cond_stage_model` + 248 `first_stage_model` + `model.diffusion_model.contexts`
+ `triplet_loss_weight`. **No UNet weight was missing**, so the SD2.1 backbone
loaded in full.

| epoch | val/loss |
|---|---|
| 1 | 0.268 |
| 9 | 0.247 |
| 18 | 0.245 |
| 29 | 0.248 |
| **45** | **0.234**  ← best |
| 48, 49 | 0.243 |

val/loss sat in a 0.245–0.249 band from epoch 9 to 39, then fell to 0.234 at
epoch 45 as the cosine decayed toward `f_min`. **That late gain is only available
because `cycle_lengths` was sized to the real step count** — the baseline's
unreachable `1e7` pins the LR flat at `f_max`, so it can never decay.

Handoff to stage 2 (select-late rule):
`checkpoints/sd21-namd-uncond-bs16-lr1.0e-04/last.ckpt` (epoch 49, 12.1 GB).
Lowest-val reference: `epoch=45-val/loss=0.234.ckpt`.

**val/loss is not comparable to the SD2.1 baseline's.** This arm scales latents
by 0.519096 and the baseline by 0.580121, so the noise-prediction target lives on
a different scale. The arms compare only through the downstream evals.
