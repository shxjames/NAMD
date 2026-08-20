"""Load-audit for the SD2.1 port. Run BEFORE any training:

    python sd21/check_sd21_load.py

Asserts every UNet tensor in the SD2.1-base checkpoint loads into
sd21.openaimodel.UNetModel with an exactly matching shape — a botched
use_linear port shows up here as shape-mismatched proj_in/proj_out keys that
init_from_ckpt would otherwise silently drop to random init. Then runs one
CPU forward at the 8x8 latent size used for 64x64 patches.
"""
import argparse
import sys

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="preprocess/v2-1_512-ema-pruned.ckpt")
    ap.add_argument("--config", default="configs/ldm/v2.1-unconditional-unet.yaml")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    unet_cfg = cfg.model.params.unet_config
    unet_cfg.params.use_checkpoint = False  # plain forward for the CPU smoke test
    unet = instantiate_from_config(unet_cfg)

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
    prefix = "model.diffusion_model."
    ckpt_unet = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    model_sd = unet.state_dict()
    mismatched = [(k, tuple(v.shape), tuple(model_sd[k].shape))
                  for k, v in ckpt_unet.items()
                  if k in model_sd and model_sd[k].shape != v.shape]
    unexpected = [k for k in ckpt_unet if k not in model_sd]
    missing = [k for k in model_sd if k not in ckpt_unet]

    print(f"ckpt UNet tensors                 : {len(ckpt_unet)}")
    print(f"shape mismatches                  : {len(mismatched)}")
    for k, a, b in mismatched[:20]:
        print(f"  {k}: ckpt{a} vs model{b}")
    print(f"unexpected (in ckpt, not in model): {len(unexpected)} {unexpected[:10]}")
    print(f"missing (in model, not in ckpt)   : {len(missing)} {missing[:10]}")

    if mismatched or unexpected or missing:
        print("AUDIT FAILED")
        sys.exit(1)

    unet.load_state_dict(ckpt_unet)
    unet.eval()
    with torch.no_grad():
        x = torch.randn(1, unet_cfg.params.in_channels, 8, 8)
        t = torch.tensor([500])
        ctx = torch.randn(1, 77, unet_cfg.params.context_dim)
        out = unet(x, t, context=ctx)
    print(f"forward OK: out {tuple(out.shape)}")
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
