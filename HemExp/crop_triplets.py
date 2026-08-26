#!/usr/bin/env python3
"""Re-frame exported triplets to the 64x64 centre crop the NAMD arms use.

Every arm already in nodule_quality_eval was built on a 64x64 field of view.
`NLST_with_second_large_cleaned` is a LARGER field-of-view version of that same
data: their images are byte-identical to img[32:96, 32:96] of the 128x128 patch
(verified exactly, 0 difference).

So a 128x128 HemExp triplet and a 64x64 NAMD triplet are not measuring the same
thing -- areas and diameters are in different pixel scales, and MedSAM 2 has a
harder segmentation problem when the nodule fills a quarter of the frame rather
than all of it. Cropping to the same window makes the metrics commensurable.

Caveat worth stating wherever these numbers appear: the model still GENERATED
the full 128x128 field. This re-frames evaluation to the common window; it is
not the same as training on 64x64 crops.

    python crop_triplets.py --in-dir eval_triplets_final --out-dir eval_triplets_final_c64
"""

import argparse
import os

import numpy as np

CROP = slice(32, 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    n_in = int((128 - args.size) // 2)
    crop = slice(n_in, n_in + args.size)

    total = 0
    for seed_dir in sorted(os.listdir(args.in_dir)):
        src = os.path.join(args.in_dir, seed_dir)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(args.out_dir, seed_dir)
        os.makedirs(dst, exist_ok=True)

        for fn in sorted(os.listdir(src)):
            if not fn.endswith(".npy"):
                continue
            a = np.load(os.path.join(src, fn))
            if a.shape[-2:] != (128, 128):
                raise ValueError(f"{fn}: expected 128x128, got {a.shape}")
            np.save(os.path.join(dst, fn), a[:, crop, crop])
            total += 1

    print(f"wrote {total} cropped triplets to {args.out_dir} ({args.size}x{args.size})")


if __name__ == "__main__":
    main()
