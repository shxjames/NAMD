# sd21/ — SD2.1-base backbone port

Self-contained fork of the SD1.5 pipeline pieces; **nothing in `ldm/` is
modified**. Configs reach this code via `target: sd21.*`.

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
