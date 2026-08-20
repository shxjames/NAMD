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

Both request `--mem=80g`: these LDM stages peaked at 50.3 GB RSS and OOM-killed
at 48g. `save_top_k` is 3; `run.py` sets `save_last=True` unconditionally, so
`last.ckpt` is always there for the select-late rule.

Expect roughly 21 h stage 1 and 7–9 h stage 2, extrapolating from the SD1.5
MedGemma arm, plus ~155 GB of checkpoints.

## Run log

| Stage | Result |
|---|---|
| configs + launchers | written, UNet build and stage-1→2 transfer verified on meta device; **nothing submitted** |
| stage 1 (uncond) | not started |
| stage 2 (cond) | not started |
