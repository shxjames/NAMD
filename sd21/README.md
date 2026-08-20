# sd21/ — SD2.1-base backbone port

Self-contained fork of the SD1.5 pipeline pieces; **nothing in `ldm/` is
modified**. Configs reach this code via `target: sd21.*`.

> **On this branch (`worktree-sd21-namd`)**: the port and weights notes below
> apply unchanged, but the "Run log", "Eval" and "Nodule-property eval" sections
> are the **SD2.1 baseline's** results, grafted verbatim from
> `worktree-sd21-baseline`. They are not this arm's numbers. For the NAMD-on-SD2.1
> arm see [`SD21_NAMD_PLAN.md`](../SD21_NAMD_PLAN.md).

| File | Forked from | Delta |
|---|---|---|
| `attention.py` | `ldm/modules/attention.py` | `SpatialTransformer(use_linear=...)` — SD2.x Linear proj_in/out instead of 1x1 Conv (ported from Stability-AI/generative-models `sgm/modules/attention.py`; the original `Stability-AI/stablediffusion` repo was deleted) |
| `openaimodel.py` | `ldm/modules/diffusionmodules/openaimodel.py` | `use_linear_in_transformer` kwarg threaded to the 3 SpatialTransformer sites; imports sd21.attention |
| `ddpm_clip.py` | `ldm/models/diffusion/ddpm_clip.py` | text tower = OpenCLIP ViT-H via HF mirror (1024-d), penultimate-layer features |
| `encoders.py` | new (subclasses `FrozenCLIPEmbedder`) | uncond-stage encoder, same tower/layer |
| `check_sd21_load.py` | new | load audit — run before any training |

## Weights (official repos are gone)

GitHub `Stability-AI/stablediffusion` and HF `stabilityai/stable-diffusion-2-1-base`
were deleted. Weights come from mirrors; sha256 verified identical on two
independent mirrors (`ckpt/stable-diffusion-2-1-base`,
`LanguageMachines/stable-diffusion-2-1-base`) and matching the historically
documented official hash:

    v2-1_512-ema-pruned.ckpt  (5,214,865,159 bytes, eps-prediction "2.1-base")
    sha256 88ecb782561455673c4b78d05093494b9c539fc6bfc08f3a9a4a0dd7b0b10f36

Text tower: `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` (HF-format OpenCLIP ViT-H,
still online) — the same weights SD2.1 froze; no `open_clip` package needed.
The 768 "v"-prediction SD2.1 checkpoint is deliberately NOT used: 2.1-base is
eps-prediction, so `ldm/models/diffusion/ddpm.py` needs no v-parameterization.

## Pipeline

```bash
# 0. audit the port (CPU, ~1 min; -m so the repo root is on sys.path)
python -m sd21.check_sd21_load

# 1. extract the SD2.1 VAE init
python preprocess/preprocess.py --ckpt_path preprocess/v2-1_512-ema-pruned.ckpt \
    --type autoencoder --output_path preprocess/v2-1-autokl.ckpt

# 2. VAE fine-tune — reuse the ablation template, only the init differs.
#    Baseline arm (plain SD-VAE fine-tune):
python run.py --config configs/autoencoder/ablation-template.yaml --train ... \
    --set model.params.ckpt_path=preprocess/v2-1-autokl.ckpt \
          model.params.feature_loss_weight=0.0 model.params.label_loss_weight=0.0
#    Method arm (NAMD aligned VAE): same with weights 1.0 / 0.2.

# 3. scale factor for the new VAE, then fill YOUR_SCALE_FACTOR in both configs
python eval_autoencoder.py ...

# 4. unconditional backbone
python run.py --config configs/ldm/v2.1-unconditional-unet.yaml --train ...

# 5. conditional backbone (fill YOUR_PATH_TO_UNCOND_CKPT)
python run.py --config configs/ldm/v2.1-conditional-unet-clip.yaml --train ...
```

Note: the uncond config pins `context_length: 0, n_prompts: 1` (the v1.5
uncond config left the UNet defaults 8/2, creating soft-prompt params the
conditional arm then dropped on load; pinning 0/1 keeps the SD2.1 uncond →
cond handoff clean).

## Run log (this branch)

| Stage | Result |
|---|---|
| VAE screen (6 cfg x 12 ep, seed 23) | winner bs32 / lr 5e-5, val/total_loss 0.135 (range 0.135-0.138) |
| VAE full (200 ep) | best epoch 155, val/total_loss 0.123: `checkpoints/sd21-vae-full-bs32-lr5e-5/epoch=155-val/total_loss=0.123.ckpt` |
| scale factor | 0.580121 (latent std 1.7238, 101 train batches, posterior.sample()) |
| uncond screen (6 cfg x 8 ep, seed 23) | winner bs16 / lr 5e-5, val/loss 0.264 (range 0.264-0.285; bs16/lr2e-4 worst) |
| uncond screen (6 cfg x 8 ep) | winner bs16 / lr 5e-5, val/loss 0.264 @ep6 (bs32 configs plateau at ep1 ~0.266) |
| uncond full (50 ep = 33 + 17 warm-start resume after scheduler-overflow crash) | handoff `checkpoints/sd21-uncond-full-bs16-lr5e-5-resume/last.ckpt` (epoch 50, select-late rule); lowest val/loss 0.248 at resume-epoch 15 (`epoch=15-val/loss=0.248.ckpt`) as reference |
| uncond full (50 ep = 33 + 17 warm-start after scheduler crash) | best val/loss 0.248 @ overall ep48: `checkpoints/sd21-uncond-full-bs16-lr5e-5-resume/epoch=15-val/loss=0.248.ckpt`; `…-resume/last.ckpt` = ep49 |
| cond screen (6 cfg x 8 ep) | seeded from `…-resume/last.ckpt` (user's array 27498487) |
| cond screen result | winner bs32 / lr 1e-4, val/loss 0.247 @ep5 (next best 0.257; bs32 > bs16, higher lr better throughout) |
| cond screen (6 cfg x 8 ep, seed 23, from uncond last.ckpt) | winner bs32 / lr 1e-4, val/loss 0.247 (range 0.247-0.281; larger bs + higher lr both help) |
| cond full (70 ep, bs32 / lr 1e-4) | best val/loss 0.220 @ep21 (ModelCheckpoint metric): `checkpoints/sd21-cond-full-bs32-lr1e-4/epoch=21-val/loss=0.220.ckpt`; `last.ckpt` = ep69. Overfits after ~ep21 (val/loss climbs to 0.36-0.42 by ep50-69, per `logs/sd21-cond-full-bs32-lr1e-4/loss_log.csv`; that CSV's own val/loss min is ep19 = 0.207 but only the ep21 top-1 checkpoint was kept). |

### Final SD2.1 baseline checkpoints
- VAE:    `checkpoints/sd21-vae-full-bs32-lr5e-5/epoch=155-val/total_loss=0.123.ckpt` (scale factor 0.580121)
- uncond: `checkpoints/sd21-uncond-full-bs16-lr5e-5-resume/epoch=15-val/loss=0.248.ckpt` (overall ep48; `…-resume/last.ckpt` = ep49 was used to seed the cond stage)
- cond:   `checkpoints/sd21-cond-full-bs32-lr1e-4/epoch=21-val/loss=0.220.ckpt`
| cond full (70 ep, bs32 / lr 1e-4) | handoff `checkpoints/sd21-cond-full-bs32-lr1e-4/last.ckpt` (epoch 70, select-late rule); lowest val/loss 0.220 at epoch 21 (`epoch=21-val/loss=0.220.ckpt`) as reference |

Every checkpoint above lives under the main checkout's `checkpoints/` (the
worktree's `checkpoints/` is a symlink to it). Selection: VAE = lowest val;
UNet stages = `last.ckpt` primary, lowest-val ckpt secondary (user rule).

## Eval (eval_all.py, 20 seeds, guidance 1.0, LPIPS+FID; JSONs in results/sd21-cond-full-bs32-lr1e-4/)

| ckpt | AUROC per-seed | AUROC avg-pred | AUPRC per-seed | LPIPS | FID |
|---|---|---|---|---|---|
| epoch 21 (lowest val/loss 0.220) | 0.727 +- 0.029 | 0.798 | 0.280 +- 0.034 | 0.211 | 88.0 +- 0.9 |
| epoch 70 (last.ckpt, val/loss 0.364) | 0.787 +- 0.013 | 0.818 | 0.351 +- 0.031 | 0.189 | 105.7 +- 0.9 |

Reference SD1.5 CLIP arm (PRED-VAE, ep 40): AUROC 0.764, AUPRC 0.293, LPIPS 0.208, FID 77.3.
Note eval_all.py names its JSON by guidance only, so the e70 run overwrote e21's;
e21 was rebuilt from its stdout via scripts/sd21/rebuild_eval_json.py.

## Nodule-property eval (MedSAM2 seg + PyRadiomics, 20 seeds x 450 samples = 9000 rows)

`scripts/sd21/nodule_eval.sbatch` -> `nodule_quality_eval/results_sd21_{e21,e70}/`

| arm / ckpt | r_area | r_HU | sphericity MAE | trajectory cosine |
|---|---|---|---|---|
| **SD2.1 CLIP, ep 21** (lowest val) | 0.571 | 0.592 | 0.1195 | 0.3104 (n=8514) |
| **SD2.1 CLIP, ep 70** (last.ckpt) | 0.688 | 0.750 | 0.1042 | 0.3598 (n=8558) |
| SD1.5 CLIP + PRED-VAE, ep 36 | 0.588 | 0.630 | 0.1201 | 0.3246 |
| SD1.5 CT-CLIP, ep 38 | 0.647 | 0.693 | 0.1065 | 0.3435 |
| SD1.5 CT-CLIP, ep 69 | 0.680 | 0.732 | 0.1040 | 0.3644 |
| IFN baseline (0624) | 0.711 | 0.468 | 0.0966 | 0.2432 |
| IFN + EHR (0812) | 0.703 | 0.535 | 0.0965 | 0.2503 |

Higher is better except sphericity MAE. Same select-late pattern as the ViT
metrics: the ep-70 checkpoint beats ep-21 on all four despite a 1.75x higher
val/loss.
