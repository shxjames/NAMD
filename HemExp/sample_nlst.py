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

_HERE = os.path.dirname(os.path.abspath(__file__))
# The vendored (adapted) fork must win over any pip-installed
# denoising-diffusion-pytorch, including an editable install pointing at a
# different checkout. Path order is the only thing that reliably guarantees it.
sys.path.insert(0, os.path.join(_HERE, "denoising-diffusion-pytorch"))
sys.path.insert(0, _HERE)

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
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="Sampling seeds; one generation pass per seed, as eval_all.py does")
    p.add_argument("--export-triplets", default=None,
                   help="Directory for nodule_quality_eval-format triplets "
                        "(seed_<seed>/sample_<idx>_<label>.npy, channels [T0, T1_gen, T1_real])")
    return p.parse_args()


def export_triplets(out_dir, seed, baseline, generated, real, labels):
    """Write per-sample uint8 triplets in the layout nodule_quality_eval expects.

    Mirrors eval_all.py:export_eval_triplets so this arm drops into the same
    harness as every other NAMD arm. One difference matters: eval_all's images
    are in [-1, 1] and it rescales with (x+1)/2, whereas this pipeline works in
    [0, 1] throughout (HemExp windows CT to [0, 1]), so the conversion here is a
    plain x*255. Applying eval_all's formula would silently compress everything
    into the upper half of the range.
    """
    seed_dir = os.path.join(out_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    def to_u8(a):
        return (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    for i in range(baseline.shape[0]):
        trip = np.stack([
            to_u8(baseline[i, 0]),   # T0      (baseline / condition)
            to_u8(generated[i, 0]),  # T1_gen  (generated follow-up)
            to_u8(real[i, 0]),       # T1_real (ground-truth follow-up)
        ])
        lbl = int(labels[i]) if labels is not None else -1
        np.save(os.path.join(seed_dir, f"sample_{i:04d}_{lbl}.npy"), trip)

    return baseline.shape[0]


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

    # An EMA state dict holds the averaged weights under `ema_model.`, the live
    # ones under `online_model.`, plus bookkeeping scalars. Select the averaged
    # subset rather than merely stripping the prefix -- otherwise every
    # online_model.* key rides along as "unexpected", which both hides a real
    # mismatch and makes the diagnostic below meaningless.
    state, source = None, "model"
    if args.use_ema and ckpt.get("ema"):
        ema_state = {
            k[len("ema_model."):]: v
            for k, v in ckpt["ema"].items()
            if k.startswith("ema_model.")
        }
        if ema_state:
            state, source = ema_state, "ema"

    if state is None:
        state = ckpt["model"]

    missing, unexpected = diffusion.load_state_dict(state, strict=False)
    print(f"loaded {args.ckpt} (weights={source}) | missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}")
    diffusion.eval()

    # ---------------------------------------------------------------- dataset
    ds = NLSTPairedDataset(
        base_path=cfg_data["nlst_base_path"],
        split=args.split,
        image_size=int(cfg_ae["ae_input_size"]),
        resize_mode=cfg_data.get("resize_mode", "resize"),
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
                resize_mode=cfg_data.get("resize_mode", "resize"),
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

    labels = np.array(ds.sample_labels)
    per_seed_gen = {}
    all_bl = all_real = None

    for seed in args.seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        bl_chunks, real_chunks, gen_chunks = [], [], []

        with torch.inference_mode():
            for bi, (bl, fu, aux_bl, aux_fu, clin, flag) in enumerate(
                tqdm(loader, desc=f"seed {seed}")
            ):
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

                bl_chunks.append(bl_c.cpu().numpy())
                real_chunks.append(fu_c.cpu().numpy())
                gen_chunks.append(gen.cpu().numpy())

                if seed == args.seeds[0]:
                    save_image(
                        torch.cat([bl_c, fu_c, gen], dim=0),
                        os.path.join(out_dir, "grids", f"batch_{bi:04d}.png"),
                        nrow=bl_c.shape[0],
                    )

        # The loader is shuffle=False, so batch order matches ds.samples for
        # every seed; baseline/real are identical across seeds by construction.
        all_bl = np.concatenate(bl_chunks, axis=0)
        all_real = np.concatenate(real_chunks, axis=0)
        per_seed_gen[seed] = np.concatenate(gen_chunks, axis=0)

        if args.export_triplets:
            n = export_triplets(args.export_triplets, seed, all_bl,
                                per_seed_gen[seed], all_real, labels)
            print(f"  seed {seed}: exported {n} triplets")

    out_npz = os.path.join(out_dir, "predictions.npz")
    np.savez_compressed(
        out_npz,
        baseline=all_bl,
        followup_real=all_real,
        # [n_seeds, N, 1, H, W]; index 0 is the first seed, for single-seed use.
        followup_generated=np.stack([per_seed_gen[s] for s in args.seeds], axis=0),
        seeds=np.array(args.seeds),
        subject_ids=np.array([s["subject_id"] for s in ds.samples]),
        labels=labels,
        cond_scale=args.cond_scale,
        flag_override=-1 if args.flag_override is None else args.flag_override,
    )
    print(f"wrote {out_npz}")
    if args.export_triplets:
        print(f"triplets under {args.export_triplets} "
              f"(nodule_quality_eval --input-format npy)")


if __name__ == "__main__":
    main()
