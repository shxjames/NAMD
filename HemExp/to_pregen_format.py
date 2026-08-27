#!/usr/bin/env python3
"""Convert exported triplets to the layout NLST-Img/eval_pregenerated.py reads.

That script is the one the NAMD arms are scored with for ViT AUROC/AUPRC,
LPIPS, FID, HaarPSI and the global / nodule / background MSE split. It expects
a slightly different shape than nodule_quality_eval:

    nodule_quality_eval        eval_pregenerated
    seed_<seed>/               boot_<seed>/
    (3, H, W)                  (3, H, W, 1)

Channel order is the same in both -- [T0/cond, T1_gen, T1_real] -- and the
sample_<idx>_<label>.npy filename convention is unchanged, so only the
directory prefix and a trailing singleton axis need fixing.

    python to_pregen_format.py --in-dir eval_triplets_final_c64 \\
                               --out-dir pregen_final
"""

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    n_seeds = 0
    n_files = 0

    for entry in sorted(os.listdir(args.in_dir)):
        src = os.path.join(args.in_dir, entry)
        if not os.path.isdir(src):
            continue
        if not entry.startswith("seed_"):
            raise ValueError(f"unexpected directory {entry!r}; expected seed_<n>")

        seed = entry[len("seed_"):]
        dst = os.path.join(args.out_dir, f"boot_{seed}")
        os.makedirs(dst, exist_ok=True)
        n_seeds += 1

        for fn in sorted(os.listdir(src)):
            if not fn.endswith(".npy"):
                continue
            a = np.load(os.path.join(src, fn))
            if a.ndim != 3 or a.shape[0] != 3:
                raise ValueError(f"{fn}: expected (3, H, W), got {a.shape}")
            np.save(os.path.join(dst, fn), a[..., None].astype(np.uint8))
            n_files += 1

    print(f"wrote {n_files} triplets across {n_seeds} boot_ dirs to {args.out_dir}")


if __name__ == "__main__":
    main()
