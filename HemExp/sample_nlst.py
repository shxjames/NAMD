#!/usr/bin/env python3
"""Generate predicted year-2 nodule patches for an NLST split.

The hemorrhage-side sampling.py is built around nnUNet NIfTI cases and IPH/IVH
probability maps, none of which exist here, so this is a separate entry point
rather than a branch inside it.

Outputs, under --out-dir:
    predictions.npz   baseline / real follow-up / generated arrays + ids + labels
    grids/*.png       baseline | real follow-up | generated, per batch

Counterfactuals -- HemExp's main clinical affordance -- are available through
the two override flags:

    --flag-override 1     generate every subject as if malignant
    --cond-scale 0.0      replace the clinical vector with the learned null
                          embedding (CFG); >1.0 amplifies its effect

    python sample_nlst.py --config CONFIGS/config_nlst.json \\
        --ckpt results/<stamp>__config_nlst/model-20.pt --split test
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from denoising_diffusion_pytorch import GaussianDiffusion, MultiHeadAEKL, Unet
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    encode_stack_chunked,
    extract_center_latent_from_stack,
    norm_latent,
    unnorm_latent,
)
from nlst_dataset import NLSTPairedDataset


def parse_args():
    p = argparse.ArgumentParser(description="Sample follow-up NLST patches")
    p.add_argument("--config", required=True, help="The diffusion config used for training")
    p.add_argument("--ckpt", required=True, help="Trainer checkpoint (model-<milestone>.pt)")
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cond-scale", type=float, default=1.0,
                   help="Clinical CFG scale: 0 disables clinical conditioning, >1 amplifies it")
    p.add_argument("--flag-override", type=int, default=None, choices=[0, 1],
                   help="Force the oracle flag for every subject (counterfactual generation)")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--limit", type=int, default=None, help="Only sample the first N subjects")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    cfg_data, cfg_ae, cfg_model, cfg_diff = cfg["data"], cfg["ae"], cfg["model"], cfg["diffusion"]

    if cfg_data.get("kind") != "nlst":
        raise ValueError("this script expects an NLST config (data.kind == 'nlst')")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt), f"samples_{args.split}")
    os.makedirs(os.path.join(out_dir, "grids"), exist_ok=True)

    # ------------------------------------------------------------- autoencoder
    head_names = tuple(cfg_ae.get("head_names", ["ct", "iph", "ivh"]))
    ae = MultiHeadAEKL(
        spatial_dims=2,
        in_channels=int(cfg_ae.get("in_channels", len(head_names))),
        channels=tuple(cfg_ae.get("channels", [64, 128, 256])),
        latent_channels=int(cfg_ae["latent_channels"]),
        num_res_blocks=2,
        norm_num_groups=32,
        attention_levels=(False, False, False),
        head_names=head_names,
    ).to(device)
    ae.load_state_dict(torch.load(cfg_ae["ae_ckpt"], map_location=device, weights_only=False)["model_state_dict"])
    ae.eval()

    # ------------------------------------------------------------------- unet
    K = int(cfg_model.get("context_slices", 1))
    unet = Unet(
        dim=int(cfg_model["dim"]),
        dim_mults=tuple(cfg_model["dim_mults"]),
        channels=int(cfg_model["channels"]),
        flash_attn=bool(cfg_model.get("flash_attn", False)),
        dropout=0.0,
        use_clinical=bool(cfg_model["use_clinical"]),
        clinical_dim=int(cfg_model.get("clinical_dim", 7)),
        use_expansion_flag_input=bool(cfg_model["use_expansion_flag_input"]),
        predict_segmentation=False,
        use_latent_diffusion=True,
        context_slices=K,
    )

    diffusion = GaussianDiffusion(
        unet,
        image_size=int(cfg_ae["latent_size"]),
        timesteps=int(cfg_diff["timesteps"]),
        sampling_timesteps=int(cfg_diff["sampling_timesteps"]),
        objective=str(cfg_diff.get("objective", "pred_v")),
        beta_schedule=str(cfg_diff.get("beta_schedule", "sigmoid")),
        auto_normalize=False,
        use_latent_diffusion=True,
        ddim_sampling_eta=float(cfg_diff.get("ddim_eta", 0.0)),
        min_snr_loss_weight=bool(cfg_diff.get("min_snr_loss_weight", False)),
        min_snr_gamma=float(cfg_diff.get("min_snr_gamma", 5.0)),
        autoencoder=ae,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get("ema") if (args.use_ema and ckpt.get("ema")) else ckpt["model"]
    # EMA state dicts prefix the wrapped weights; strip it so both load the same way.
    state = {k.replace("ema_model.", "", 1): v for k, v in state.items() if not k.startswith("initted")}
    missing, unexpected = diffusion.load_state_dict(state, strict=False)
    print(f"loaded {args.ckpt} (ema={args.use_ema}) | missing={len(missing)} unexpected={len(unexpected)}")
    diffusion.eval()

    # ---------------------------------------------------------------- dataset
    ds = NLSTPairedDataset(
        base_path=cfg_data["nlst_base_path"],
        split=args.split,
        image_size=int(cfg_ae["ae_input_size"]),
        context_slices=K,
        selected_features=cfg_data.get("selected_features"),
        excluded_features=cfg_data.get("excluded_features"),
    )

    # Clinical z-scoring MUST reuse the training split's statistics.
    if cfg_data.get("normalize_clinical", True):
        mean, std = cfg_data.get("clinical_mean"), cfg_data.get("clinical_std")
        if mean is None or std is None:
            train_ds = NLSTPairedDataset(
                base_path=cfg_data["nlst_base_path"],
                split=cfg_data.get("nlst_train_split", "train"),
                image_size=int(cfg_ae["ae_input_size"]),
                context_slices=K,
                selected_features=cfg_data.get("selected_features"),
                excluded_features=cfg_data.get("excluded_features"),
            )
            mean, std = train_ds.compute_clinical_stats()
        ds.set_clinical_stats(mean, std)

    if args.limit:
        ds.samples = ds.samples[: args.limit]
        ds.sample_labels = ds.sample_labels[: args.limit]

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"sampling {len(ds)} subjects from split '{args.split}'")

    latent_mean, latent_std = cfg_ae["mean"], cfg_ae["std"]
    predict_delta = bool(cfg_diff.get("latent_predict_delta", True))

    all_bl, all_real, all_gen = [], [], []

    with torch.inference_mode():
        for bi, (bl, fu, aux_bl, aux_fu, clin, flag) in enumerate(tqdm(loader)):
            bl, fu, aux_bl = bl.to(device), fu.to(device), aux_bl.to(device)
            clin, flag = clin.to(device), flag.to(device).squeeze(1)

            if args.flag_override is not None:
                flag = torch.full_like(flag, args.flag_override)

            cond_z = norm_latent(encode_stack_chunked(ae, bl, aux_bl), latent_mean, latent_std, repeat_k=K)
            base_z = extract_center_latent_from_stack(cond_z, K)

            gen_pred = diffusion.sample_conditioned(
                cond_img=cond_z, seg=None, clinical_vars=clin,
                expansion_flag=flag, cond_scale=args.cond_scale,
            )
            gen_z = gen_pred + base_z if predict_delta else gen_pred
            gen = ae.decode(unnorm_latent(gen_z, latent_mean, latent_std, repeat_k=1))[:, 0:1].clamp(0, 1)

            mid = K // 2
            bl_c = bl[:, mid : mid + 1]
            fu_c = fu[:, mid : mid + 1]

            all_bl.append(bl_c.cpu().numpy())
            all_real.append(fu_c.cpu().numpy())
            all_gen.append(gen.cpu().numpy())

            save_image(
                torch.cat([bl_c, fu_c, gen], dim=0),
                os.path.join(out_dir, "grids", f"batch_{bi:04d}.png"),
                nrow=bl_c.shape[0],
            )

    out_npz = os.path.join(out_dir, "predictions.npz")
    np.savez_compressed(
        out_npz,
        baseline=np.concatenate(all_bl, axis=0),
        followup_real=np.concatenate(all_real, axis=0),
        followup_generated=np.concatenate(all_gen, axis=0),
        subject_ids=np.array([s["subject_id"] for s in ds.samples]),
        labels=np.array(ds.sample_labels),
        cond_scale=args.cond_scale,
        flag_override=-1 if args.flag_override is None else args.flag_override,
    )
    print(f"wrote {out_npz}")


if __name__ == "__main__":
    main()
