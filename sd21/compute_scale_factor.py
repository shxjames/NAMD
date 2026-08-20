"""Vectorized scale-factor computation for a first-stage VAE checkpoint.

Same protocol as eval_autoencoder.py (train loader, first 101 batches,
posterior.sample()) but reduced on-GPU instead of a Python .item() loop per
latent element, so it takes minutes, not hours.

    python -m sd21.compute_scale_factor \
        --config checkpoints/<project>/config.yaml --ckpt <best.ckpt>
"""
import argparse

import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--max_batches', type=int, default=101)
    parser.add_argument('--seed', type=int, default=23)
    args = parser.parse_args()

    seed_everything(args.seed)
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(args.ckpt)
    model.eval().to('cuda')

    ds = instantiate_from_config(config.data)
    ds.setup()

    n, s, s2 = 0, 0.0, 0.0
    for batch_idx, (img, *_) in enumerate(ds.train_dataloader()):
        img = img.to('cuda').to(model.dtype)
        with torch.no_grad():
            z = model.encode(img).sample()
        n += z.numel()
        s += z.sum().double().item()
        s2 += (z.double() ** 2).sum().item()
        if batch_idx >= args.max_batches - 1:
            break

    mean = s / n
    std = (s2 / n - mean ** 2) ** 0.5
    print(f"n={n} mean={mean:.6f} std={std:.6f}")
    print(f"SCALE_FACTOR={1.0 / std:.6f}")


if __name__ == "__main__":
    main()
