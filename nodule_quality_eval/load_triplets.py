"""Triplet loaders.

Two supported input layouts:

1. Image layout (default):

       <input_root>/
           <sample_id>/
               T0.{png,jpg,jpeg,tif,tiff,bmp}
               T1_real.{png,jpg,jpeg,tif,tiff,bmp}
               T1_gen.{png,jpg,jpeg,tif,tiff,bmp}

   Or with multiple generation seeds:

       <input_root>/
           <seed>/
               <sample_id>/
                   T0.png
                   T1_real.png
                   T1_gen.png

2. NPY layout:

       <input_root>/
           <seed>/
               <sample_id>.npy        shape (3, H, W) — idx 0 = T0,
                                                       idx 1 = T1_gen,
                                                       idx 2 = T1_real

Each patch must be a 2D grayscale uint8 array (8-bit, 0..255). Images
are read with PIL and converted to grayscale.
"""
import os
import glob

import numpy as np
from PIL import Image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
NPY_EXT = ".npy"

T0_NAMES = ("T0", "t0", "baseline")
T1R_NAMES = ("T1_real", "t1_real", "T1real", "real", "followup", "T1")
T1G_NAMES = ("T1_gen", "t1_gen", "T1gen", "gen", "generated")


def _read_grayscale(path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def _find_file(folder, candidates):
    """Find the first file matching any name in `candidates` with any image ext."""
    for name in candidates:
        for ext in IMAGE_EXTS:
            p = os.path.join(folder, name + ext)
            if os.path.exists(p):
                return p
    return None


def load_image_triplet(sample_dir):
    """Read T0 / T1_real / T1_gen images from a single sample folder.

    Returns (T0, T1_real, T1_gen) as 2-D uint8 arrays.
    """
    p_T0 = _find_file(sample_dir, T0_NAMES)
    p_T1r = _find_file(sample_dir, T1R_NAMES)
    p_T1g = _find_file(sample_dir, T1G_NAMES)
    missing = [name for name, p in zip(("T0", "T1_real", "T1_gen"),
                                       (p_T0, p_T1r, p_T1g)) if p is None]
    if missing:
        raise FileNotFoundError(
            f"{sample_dir}: missing image(s) for {missing}. "
            f"Expected files like T0.png / T1_real.png / T1_gen.png."
        )
    return _read_grayscale(p_T0), _read_grayscale(p_T1r), _read_grayscale(p_T1g)


def load_npy_triplet(npy_path):
    """Read a 3-channel .npy patch.

    idx0 = T0, idx1 = T1_gen, idx2 = T1_real.
    """
    data = np.load(npy_path)
    arrs = []
    for i in range(3):
        a = data[i]
        if a.ndim == 3:
            a = a[..., 0]
        arrs.append(a.astype(np.uint8))
    T0, T1_gen, T1_real = arrs
    return T0, T1_real, T1_gen


def parse_label(sample_id):
    """Optional binary label suffix: '<id>_<label>'. Returns -1 if not present."""
    parts = sample_id.split("_")
    try:
        return int(parts[-1])
    except ValueError:
        return -1


def iter_image_triplets(input_root):
    """Yield (seed, sample_id, T0, T1_real, T1_gen) for every triplet under
    `input_root`. Handles both flat and seed-organized layouts.
    """
    entries = sorted(os.listdir(input_root))
    seed_dirs = []

    # If the first entry that's a directory contains direct image files, treat
    # the entire root as a single-seed flat layout. Otherwise treat its
    # subdirectories as seeds.
    flat = False
    for e in entries:
        ep = os.path.join(input_root, e)
        if not os.path.isdir(ep):
            continue
        has_image = any(
            f.lower().endswith(IMAGE_EXTS) for f in os.listdir(ep)
        )
        if has_image:
            flat = True
        break

    if flat:
        seed_dirs = [("default", input_root)]
    else:
        for e in entries:
            ep = os.path.join(input_root, e)
            if os.path.isdir(ep):
                seed_dirs.append((e, ep))

    for seed, seed_dir in seed_dirs:
        samples = sorted(
            d for d in os.listdir(seed_dir)
            if os.path.isdir(os.path.join(seed_dir, d))
        )
        for sid in samples:
            sample_dir = os.path.join(seed_dir, sid)
            T0, T1r, T1g = load_image_triplet(sample_dir)
            yield seed, sid, T0, T1r, T1g


def iter_npy_triplets(input_root):
    """Yield (seed, sample_id, T0, T1_real, T1_gen) from a .npy directory tree.

    Supports both:
        <input_root>/<seed>/<sample_id>.npy
        <input_root>/<sample_id>.npy
    """
    npy_at_root = sorted(glob.glob(os.path.join(input_root, "*" + NPY_EXT)))
    if npy_at_root:
        for path in npy_at_root:
            sid = os.path.splitext(os.path.basename(path))[0]
            T0, T1r, T1g = load_npy_triplet(path)
            yield "default", sid, T0, T1r, T1g
        return

    seeds = sorted(
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    )
    for seed in seeds:
        seed_dir = os.path.join(input_root, seed)
        for path in sorted(glob.glob(os.path.join(seed_dir, "*" + NPY_EXT))):
            sid = os.path.splitext(os.path.basename(path))[0]
            T0, T1r, T1g = load_npy_triplet(path)
            yield seed, sid, T0, T1r, T1g
