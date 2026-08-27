"""Per-(seed, sample) image + classifier metrics for the pregenerated NAMD_3 triplets.

Mirrors the preprocessing in NLST-Img/eval_pregenerated.py exactly (npy uint8 ->
[-1, 1], grayscale repeated to 3 channels, LPIPS on the [0, 1] rescale) so the
numbers here aggregate back to the published NAMD_3 figures.

Writes results/NAMD_3_cases/per_sample_seed_metrics.npz with arrays shaped
[n_seeds, n_samples] (seeds sorted by int, samples sorted by index).
"""
import argparse
import os
import re
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

# Running this as a script puts scripts/namd3_cases/ on sys.path, not the repo
# root, so `ldm` has to be put there explicitly.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from ldm.util import instantiate_from_config  # noqa: E402

DEFAULT_NAMD_DIR = "/scratch/liyues_root/liyues/shxjames/NLST-Img/NAMD_3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def npy_to_tensor(img_np):
    """(H, W, 1) uint8 -> (1, 1, H, W) float32 in [-1, 1]. Same rule as eval_pregenerated."""
    img = img_np.squeeze(-1).astype(np.float32)
    if img_np.dtype == np.uint8 or img.max() > 1.5:
        img = img / 127.5 - 1.0
    elif img.min() >= -0.1 and img.max() <= 1.1:
        img = img * 2.0 - 1.0
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)


def load_seed_dir(seed_dir):
    pattern = re.compile(r"^sample_(\d+)_(\d+)\.npy$")
    items = []
    for fname in os.listdir(seed_dir):
        m = pattern.match(fname)
        if m is None:
            continue
        data = np.load(os.path.join(seed_dir, fname))  # (3, 64, 64, 1)
        items.append((int(m.group(1)), int(m.group(2)), data[0], data[1], data[2]))
    items.sort(key=lambda x: x[0])
    return items


def discover_seeds(namd_dir):
    pattern = re.compile(r"^boot_(\d+)$")
    seeds = []
    for name in os.listdir(namd_dir):
        m = pattern.match(name)
        if m and os.path.isdir(os.path.join(namd_dir, name)):
            seeds.append((int(m.group(1)), os.path.join(namd_dir, name)))
    seeds.sort(key=lambda x: x[0])
    return seeds


def roi_mask(h, w, frac=0.25):
    """Centre disc used elsewhere in this repo as the nodule region proxy."""
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * frac)
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--namd_dir", default=DEFAULT_NAMD_DIR)
    ap.add_argument("--eval_config", default="configs/evaluator/vit.yaml")
    ap.add_argument("--out_dir", default="results/NAMD_3_cases")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    eval_model = instantiate_from_config(OmegaConf.load(args.eval_config).model).to(device)
    eval_model.eval()
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True).to(device)

    seeds = discover_seeds(args.namd_dir)
    assert seeds, f"no boot_* dirs under {args.namd_dir}"
    print(f"{len(seeds)} seeds: {[s for s, _ in seeds]}", flush=True)

    lpips_all, logit_gen_all = [], []
    logit_t0_all, logit_real_all = [], []
    mse_all, mse_roi_all, psnr_all = [], [], []
    sample_ids = labels = None
    t0_ref = real_ref = None
    mask = None

    for seed_int, seed_dir in seeds:
        items = load_seed_dir(seed_dir)
        ids = np.array([x[0] for x in items])
        lab = np.array([x[1] for x in items])
        if sample_ids is None:
            sample_ids, labels = ids, lab
        else:
            assert np.array_equal(ids, sample_ids) and np.array_equal(lab, labels), \
                f"{seed_dir}: sample id/label order differs from the first seed"

        t0_stack = np.stack([x[2].squeeze(-1) for x in items]).astype(np.float32)
        real_stack = np.stack([x[4].squeeze(-1) for x in items]).astype(np.float32)
        if t0_ref is None:
            t0_ref, real_ref = t0_stack, real_stack
            mask = roi_mask(*t0_ref.shape[1:])
        else:
            # T0 / T1_real must be seed-invariant; the exporter re-wrote them per seed.
            assert np.abs(t0_stack - t0_ref).max() <= 2, f"{seed_dir}: T0 drifts across seeds"
            assert np.abs(real_stack - real_ref).max() <= 2, f"{seed_dir}: T1_real drifts"

        s_lpips, s_gen, s_t0, s_real = [], [], [], []
        for start in tqdm(range(0, len(items), args.batch_size), desc=f"boot_{seed_int}"):
            chunk = items[start:start + args.batch_size]
            t0 = torch.cat([npy_to_tensor(x[2]) for x in chunk]).repeat(1, 3, 1, 1).to(device)
            gen = torch.cat([npy_to_tensor(x[3]) for x in chunk]).repeat(1, 3, 1, 1).to(device)
            real = torch.cat([npy_to_tensor(x[4]) for x in chunk]).repeat(1, 3, 1, 1).to(device)

            with torch.no_grad():
                s_gen.append(eval_model(gen, return_map=True)[0].cpu().numpy().reshape(-1))
                s_t0.append(eval_model(t0, return_map=True)[0].cpu().numpy().reshape(-1))
                s_real.append(eval_model(real, return_map=True)[0].cpu().numpy().reshape(-1))

                gen01 = (gen.clamp(-1, 1) + 1) / 2
                real01 = (real.clamp(-1, 1) + 1) / 2
                for i in range(gen01.shape[0]):
                    s_lpips.append(lpips_metric(gen01[i:i + 1], real01[i:i + 1]).item())

        gen_stack = np.stack([x[3].squeeze(-1) for x in items]).astype(np.float32)
        diff = (gen_stack - real_stack) / 127.5          # error in [-1, 1] units
        mse = (diff ** 2).mean(axis=(1, 2))
        mse_roi = (diff ** 2)[:, mask].mean(axis=1)
        psnr = 10 * np.log10(4.0 / np.maximum(mse, 1e-12))  # peak-to-peak = 2

        lpips_all.append(np.array(s_lpips))
        logit_gen_all.append(np.concatenate(s_gen))
        logit_t0_all.append(np.concatenate(s_t0))
        logit_real_all.append(np.concatenate(s_real))
        mse_all.append(mse)
        mse_roi_all.append(mse_roi)
        psnr_all.append(psnr)

    out = os.path.join(args.out_dir, "per_sample_seed_metrics.npz")
    np.savez(
        out,
        seeds=np.array([s for s, _ in seeds]),
        sample_ids=sample_ids,
        labels=labels,
        lpips=np.stack(lpips_all),
        logit_gen=np.stack(logit_gen_all),
        logit_t0=np.stack(logit_t0_all),
        logit_real=np.stack(logit_real_all),
        mse=np.stack(mse_all),
        mse_roi=np.stack(mse_roi_all),
        psnr=np.stack(psnr_all),
    )
    print(f"wrote {out}  lpips shape {np.stack(lpips_all).shape}", flush=True)
    print(f"LPIPS overall mean = {np.mean(lpips_all):.4f}", flush=True)


if __name__ == "__main__":
    main()
