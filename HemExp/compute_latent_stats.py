#!/usr/bin/env python3
"""Per-channel latent mean/std for a trained NLST first stage.

The diffusion config's `ae.mean` / `ae.std` whiten the AE latents before the
UNet sees them (norm_latent / unnorm_latent). They must be recomputed whenever
the autoencoder is retrained -- reusing HemExp's hemorrhage statistics against
an NLST autoencoder would mis-scale every latent.

Statistics are taken over the TRAIN split only, on un-augmented images.

    python compute_latent_stats.py \
        --ae-ckpt output/autoencoder_nlst/checkpoints/checkpoint_step_20000.pth \
        --base-path ../NLST_with_second_large_cleaned
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
# The vendored (adapted) fork must win over any pip-installed
# denoising-diffusion-pytorch, including an editable install pointing at a
# different checkout. Path order is the only thing that reliably guarantees it.
sys.path.insert(0, os.path.join(_HERE, "denoising-diffusion-pytorch"))
sys.path.insert(0, _HERE)

from nlst_dataset import NLSTAutoencoderDataset
from train_autoencoder_nlst import MultiHeadAEKL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-ckpt", required=True)
    parser.add_argument("--base-path", default="../NLST_with_second_large_cleaned")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--channels", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default=None, help="Optional path to write the stats as JSON")
    parser.add_argument("--update-config", default=None,
                        help="Diffusion config to write ae.mean / ae.std into, in place")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiHeadAEKL(
        spatial_dims=2,
        in_channels=1,
        channels=tuple(args.channels),
        latent_channels=args.latent_channels,
        head_names=("ct",),
    ).to(device)

    ckpt = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = NLSTAutoencoderDataset(base_path=args.base_path, split=args.split, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Encoding {len(ds)} images from split '{args.split}'")

    # Streaming per-channel sums: the latent set is small, but this keeps the
    # script independent of how many images the split holds.
    n = 0
    total = None
    total_sq = None

    with torch.no_grad():
        for batch in loader:
            z = model.encode_stage_2_inputs(batch["image"].to(device))  # [B, C, h, w]
            flat = z.permute(1, 0, 2, 3).reshape(z.shape[1], -1).double()

            if total is None:
                total = flat.sum(dim=1)
                total_sq = (flat ** 2).sum(dim=1)
            else:
                total += flat.sum(dim=1)
                total_sq += (flat ** 2).sum(dim=1)
            n += flat.shape[1]

    mean = total / n
    var = (total_sq / n) - mean ** 2
    std = var.clamp_min(1e-12).sqrt()

    stats = {"mean": [float(v) for v in mean.cpu()], "std": [float(v) for v in std.cpu()]}

    print(json.dumps(stats, indent=2))
    print("\nPaste into the diffusion config under \"ae\":")
    print(f'  "mean": {stats["mean"]},')
    print(f'  "std" : {stats["std"]}')

    if args.out:
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nWrote {args.out}")

    if args.update_config:
        with open(args.update_config) as f:
            cfg = json.load(f)
        n = int(cfg["ae"]["latent_channels"])
        if len(stats["mean"]) != n:
            raise ValueError(
                f"computed {len(stats['mean'])} latent channels but the config declares {n}"
            )
        cfg["ae"]["mean"] = stats["mean"]
        cfg["ae"]["std"] = stats["std"]
        cfg["ae"]["_comment_stats"] = f"Written by compute_latent_stats.py from {args.ae_ckpt}"
        with open(args.update_config, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Updated ae.mean / ae.std in {args.update_config}")


if __name__ == "__main__":
    main()
