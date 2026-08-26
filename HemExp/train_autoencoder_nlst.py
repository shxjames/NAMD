#!/usr/bin/env python3
"""Train HemExp's MultiHeadAEKL first stage on NLST nodule patches.

Same architecture and loss recipe as train_autoencoder.py -- MONAI AutoencoderKL
backbone, KL + perceptual + patch-adversarial -- with the segmentation terms
removed, since NLST carries no masks:

    HemExp AE                     NLST AE (this script)
    in_channels=3 (ct, iph, ivh)  in_channels=1 (nodule patch)
    heads: ct / iph / ivh         heads: ct
    recon + dice + bce            recon only
    384 x 384                     128 x 128

Each paired jpg contributes two independent training images (year-1 and
year-2): the VAE models the marginal distribution of nodule patches, not the
pairing, so both halves are valid samples.

    python train_autoencoder_nlst.py --config CONFIGS/config_ae_nlst.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
torch.backends.cudnn.benchmark = True

try:
    from monai.networks.layers import Act
    from monai.losses import PatchAdversarialLoss
    from monai.networks.nets import AutoencoderKL, PatchDiscriminator
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "MONAI is not installed in this environment. Run from the env built by environment.yml."
    ) from e

try:
    from monai.losses import PerceptualLoss

    PERCEPTUAL_AVAILABLE = True
except Exception:
    PERCEPTUAL_AVAILABLE = False

import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from nlst_dataset import NLSTAutoencoderDataset

DEFAULTS = {
    "base_path": "../NLST_with_second_large_cleaned",
    "train_split": "train",
    "val_split": "val",
    "image_size": 128,
    "latent_channels": 4,
    "channels": [64, 128, 256],
    "num_res_blocks": 2,
    "norm_num_groups": 32,
    "batch_size": 16,
    "accumulate_grad_batches": 1,
    "total_epochs": 150,
    "total_steps": None,
    "lr_g": 5e-5,
    "lr_d": 5e-6,
    "kl_weight": 1e-8,
    "perceptual_weight": 0.001,
    "adversarial_weight": 0.005,
    "warmup_steps_percept": 2500,
    "warmup_steps_adv": 5000,
    "augment_flip": True,
    "augment_rotation": False,
    "rotation_range_deg": 0.0,
    "mixed_precision": "no",
    "num_workers": 8,
    "log_interval": 100,
    "save_interval": 1000,
    "val_interval": None,
    "output_dir": "output/autoencoder_nlst",
    "resume_checkpoint": None,
    "seed": 123,
    "wandb_enabled": True,
    "wandb_project": "hemexp-nlst-autoencoder",
}


class MultiHeadAEKL(nn.Module):
    """Mirror of denoising_diffusion_pytorch.MultiHeadAEKL.

    Duplicated here (as in HemExp's original train_autoencoder.py) so the
    first stage trains without importing the diffusion package. The state dict
    is key-compatible with the diffusion-side class.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        feature_channels: int = 256,
        latent_channels: int = 4,
        num_res_blocks: int = 2,
        channels: tuple = (64, 128, 256),
        norm_num_groups: int = 32,
        attention_levels: tuple = (False, False, False),
        head_names: tuple = ("ct",),
    ):
        super().__init__()
        self.head_names = tuple(head_names)
        self.backbone = AutoencoderKL(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_channels,
            channels=tuple(channels),
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            attention_levels=tuple(attention_levels),
        )
        for name in self.head_names:
            setattr(self, f"{name}_head", nn.Conv2d(feature_channels, 1, 1))

    def _apply_heads(self, y):
        return torch.cat([getattr(self, f"{n}_head")(y) for n in self.head_names], dim=1)

    def forward(self, x):
        y, z_mu, z_sigma = self.backbone(x)
        return self._apply_heads(y), z_mu, z_sigma

    @torch.no_grad()
    def encode_stage_2_inputs(self, x):
        z_mu, z_sigma = self.backbone.encode(x)
        return self.backbone.sampling(z_mu, z_sigma)

    @torch.no_grad()
    def decode(self, z):
        return self._apply_heads(self.backbone.decode(z))


def compute_kl(z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
    """KL(q||N(0,1)) for MONAI's (mu, sigma) parameterisation."""
    term = z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + 1e-8) - 1
    return 0.5 * torch.sum(term, dim=list(range(1, len(z_mu.shape)))).mean()


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path=None):
    cfg = dict(DEFAULTS)
    if path:
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


@torch.no_grad()
def log_reconstructions(model, loader, device, step, output_dir, max_images=8):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    batch = next(iter(loader))
    images = batch["image"][:max_images].to(device)
    recon, _, _ = model(images)
    panel = torch.cat([images, recon[:, 0:1].clamp(0, 1)], dim=0)
    save_image(panel, os.path.join(output_dir, f"recon_step_{step}.png"), nrow=max_images)
    model.train()
    return images, recon


@torch.no_grad()
def evaluate(model, loader, device, kl_weight):
    """Mean reconstruction / KL loss over the validation split."""
    was_training = model.training
    model.eval()

    recon_total, kl_total, n = 0.0, 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        recon, z_mu, z_sigma = model(images)
        b = images.shape[0]
        recon_total += float(((recon[:, 0:1] - images) ** 2).mean().item()) * b
        kl_total += float(compute_kl(z_mu, z_sigma).item()) * b
        n += b

    if was_training:
        model.train()

    recon = recon_total / max(1, n)
    kl = kl_total / max(1, n)
    return {"val/recon": recon, "val/kl": kl, "val/loss": recon + kl_weight * kl}


def main():
    parser = argparse.ArgumentParser(description="Train the NLST first-stage autoencoder")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    accelerator = Accelerator(
        mixed_precision=cfg["mixed_precision"],
        gradient_accumulation_steps=int(cfg["accumulate_grad_batches"]),
    )
    device = accelerator.device

    if cfg["wandb_enabled"] and accelerator.is_main_process:
        wandb.init(project=cfg["wandb_project"], config=cfg)

    train_ds = NLSTAutoencoderDataset(
        base_path=cfg["base_path"],
        split=cfg["train_split"],
        image_size=int(cfg["image_size"]),
        augment_flip=bool(cfg["augment_flip"]),
        augment_rotation=bool(cfg["augment_rotation"]),
        rotation_range_deg=float(cfg["rotation_range_deg"]),
    )
    val_ds = NLSTAutoencoderDataset(
        base_path=cfg["base_path"],
        split=cfg["val_split"],
        image_size=int(cfg["image_size"]),
    )
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=2)

    model = MultiHeadAEKL(
        spatial_dims=2,
        in_channels=1,
        channels=tuple(cfg["channels"]),
        latent_channels=int(cfg["latent_channels"]),
        num_res_blocks=int(cfg["num_res_blocks"]),
        norm_num_groups=int(cfg["norm_num_groups"]),
        attention_levels=(False,) * len(cfg["channels"]),
        head_names=("ct",),
    ).to(device)

    discriminator = PatchDiscriminator(
        spatial_dims=2,
        num_layers_d=3,
        channels=64,
        in_channels=1,
        out_channels=1,
        kernel_size=4,
        activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
        norm="BATCH",
        bias=False,
        padding=1,
    ).to(device)

    perceptual_loss = None
    perceptual_weight = float(cfg["perceptual_weight"])
    if PERCEPTUAL_AVAILABLE and perceptual_weight > 0:
        try:
            perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="squeeze").to(device)
        except Exception as e:
            perceptual_weight = 0.0
            print(f"[WARN] PerceptualLoss disabled: {e}")
    else:
        perceptual_weight = 0.0

    optimizer_g = torch.optim.Adam(model.parameters(), lr=float(cfg["lr_g"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=float(cfg["lr_d"]))

    steps_per_epoch = max(1, len(train_loader) // int(cfg["accumulate_grad_batches"]))
    if cfg.get("total_epochs"):
        total_steps = int(cfg["total_epochs"]) * steps_per_epoch
    else:
        total_steps = int(cfg["total_steps"])
    val_interval = int(cfg["val_interval"]) if cfg.get("val_interval") else steps_per_epoch
    print(
        f"{steps_per_epoch} steps/epoch -> training {total_steps} steps "
        f"(~{total_steps / steps_per_epoch:.0f} epochs); val every {val_interval}"
    )
    best_val = float("inf")
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=total_steps)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=total_steps)

    global_step = 0
    if cfg["resume_checkpoint"]:
        ckpt = torch.load(cfg["resume_checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        global_step = int(ckpt.get("global_step", 0))
        print(f"Resumed from {cfg['resume_checkpoint']} at step {global_step}")

    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    model, discriminator, optimizer_g, optimizer_d, train_loader = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_loader
    )

    pbar = tqdm(total=total_steps, initial=global_step, desc="AE", disable=not accelerator.is_main_process)
    accum = {"recon": 0.0, "kl": 0.0, "percept": 0.0, "gen_adv": 0.0, "disc": 0.0, "n": 0}
    start_time = time.time()

    while global_step < total_steps:
        model.train()
        discriminator.train()

        for batch in train_loader:
            train_discriminator = global_step >= int(cfg["warmup_steps_adv"])
            adv_weight = float(cfg["adversarial_weight"]) if train_discriminator else 0.0
            pct_weight = perceptual_weight if global_step >= int(cfg["warmup_steps_percept"]) else 0.0

            images = batch["image"].to(device)

            with accelerator.accumulate(model):
                optimizer_g.zero_grad(set_to_none=True)
                with accelerator.autocast():
                    recon, z_mu, z_sigma = model(images)

                    recon_loss = ((recon[:, 0:1] - images) ** 2).mean()
                    kl_loss = compute_kl(z_mu, z_sigma)

                    if perceptual_loss is not None and pct_weight > 0:
                        p_loss = perceptual_loss(recon[:, 0:1].float(), images.float())
                    else:
                        p_loss = torch.zeros((), device=device)

                    if train_discriminator:
                        logits_fake = discriminator(recon[:, 0:1].contiguous().float())[-1]
                        gen_adv_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    else:
                        gen_adv_loss = torch.zeros((), device=device)

                    loss_g = (
                        recon_loss
                        + float(cfg["kl_weight"]) * kl_loss
                        + pct_weight * p_loss
                        + adv_weight * gen_adv_loss
                    )

                accelerator.backward(loss_g)
                optimizer_g.step()

            if train_discriminator:
                with accelerator.accumulate(discriminator):
                    optimizer_d.zero_grad(set_to_none=True)
                    with accelerator.autocast():
                        logits_fake = discriminator(recon[:, 0:1].contiguous().detach())[-1]
                        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                        logits_real = discriminator(images.contiguous().detach())[-1]
                        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                        disc_loss = 0.5 * (loss_d_fake + loss_d_real)
                        loss_d = adv_weight * disc_loss
                    accelerator.backward(loss_d)
                    optimizer_d.step()
            else:
                disc_loss = torch.zeros((), device=device)

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            scheduler_g.step()
            scheduler_d.step()
            pbar.update(1)

            accum["recon"] += float(recon_loss.item())
            accum["kl"] += float(kl_loss.item())
            accum["percept"] += float(p_loss.item())
            accum["gen_adv"] += float(gen_adv_loss.item())
            accum["disc"] += float(disc_loss.item())
            accum["n"] += 1

            if accelerator.is_main_process and global_step % int(cfg["log_interval"]) == 0:
                n = max(1, accum["n"])
                log_dict = {
                    "step": global_step,
                    "epoch": global_step / steps_per_epoch,
                    "train/recon": accum["recon"] / n,
                    "train/kl": accum["kl"] / n,
                    "train/perceptual": accum["percept"] / n,
                    "train/gen_adv": accum["gen_adv"] / n,
                    "train/disc": accum["disc"] / n,
                    "lr_g": optimizer_g.param_groups[0]["lr"],
                    "lr_d": optimizer_d.param_groups[0]["lr"],
                }
                if cfg["wandb_enabled"]:
                    wandb.log(log_dict)
                pbar.set_description(f"AE recon {log_dict['train/recon']:.5f}")

                log_reconstructions(
                    accelerator.unwrap_model(model), val_loader, device, global_step,
                    os.path.join(output_dir, "recons"),
                )
                accum = {k: 0.0 for k in accum}

            if accelerator.is_main_process and global_step % val_interval == 0:
                val_metrics = evaluate(
                    accelerator.unwrap_model(model), val_loader, device, float(cfg["kl_weight"])
                )
                val_metrics["epoch"] = global_step / steps_per_epoch
                if cfg["wandb_enabled"]:
                    wandb.log(val_metrics, step=global_step)
                print(
                    f"  step {global_step} (epoch {val_metrics['epoch']:.1f}): "
                    f"val recon {val_metrics['val/recon']:.6f}"
                )

                if val_metrics["val/loss"] < best_val:
                    best_val = val_metrics["val/loss"]
                    ckpt_dir = os.path.join(output_dir, "checkpoints")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(
                        {
                            "global_step": global_step,
                            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                            "discriminator_state_dict": accelerator.unwrap_model(discriminator).state_dict(),
                            "image_size": int(cfg["image_size"]),
                            "latent_channels": int(cfg["latent_channels"]),
                            "head_names": ["ct"],
                            "in_channels": 1,
                            "val_loss": best_val,
                        },
                        os.path.join(ckpt_dir, "checkpoint_best.pth"),
                    )

            if accelerator.is_main_process and global_step % int(cfg["save_interval"]) == 0:
                ckpt_dir = os.path.join(output_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                path = os.path.join(ckpt_dir, f"checkpoint_step_{global_step}.pth")
                torch.save(
                    {
                        "global_step": global_step,
                        "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                        "discriminator_state_dict": accelerator.unwrap_model(discriminator).state_dict(),
                        "optimizer_g_state_dict": optimizer_g.state_dict(),
                        "optimizer_d_state_dict": optimizer_d.state_dict(),
                        "image_size": int(cfg["image_size"]),
                        "latent_channels": int(cfg["latent_channels"]),
                        "head_names": ["ct"],
                        "in_channels": 1,
                    },
                    path,
                )
                print(f"Saved {os.path.basename(path)}")

            if global_step >= total_steps:
                break

    pbar.close()

    # HemExp ships its FINAL checkpoint (adversarially finetuned), while NAMD's
    # convention for the VAE stage is lowest-val. Save both so the choice is a
    # config edit rather than a re-run: the last ~25% of training here is the
    # adversarial phase, which raises val recon by design.
    if accelerator.is_main_process:
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {
                "global_step": global_step,
                "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                "discriminator_state_dict": accelerator.unwrap_model(discriminator).state_dict(),
                "image_size": int(cfg["image_size"]),
                "latent_channels": int(cfg["latent_channels"]),
                "head_names": ["ct"],
                "in_channels": 1,
            },
            os.path.join(ckpt_dir, "checkpoint_last.pth"),
        )
        print(f"Saved checkpoint_last.pth at step {global_step} (best val loss {best_val:.6f})")

    print(f"Done in {time.time() - start_time:.1f}s")
    if cfg["wandb_enabled"] and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
