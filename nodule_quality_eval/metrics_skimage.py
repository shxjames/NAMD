"""Metric computations: area, intensity distribution, shape, texture.

Uses only scikit-image + scipy + numpy — no pyradiomics dependency.
Feature families covered:
  * Area / size change (binary mask area, delta consistency)
  * Intensity distribution (mean, std, skew, kurt, entropy, Wasserstein)
  * Shape regularity (skimage regionprops: circularity, eccentricity,
    solidity, aspect ratio, custom spiculation index)
  * Texture (first-order + GLCM via skimage.feature.graycomatrix)
"""
import numpy as np
from scipy import stats
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import convex_hull_image


# ----------------------------------------------------------------------
# Area / size change
# ----------------------------------------------------------------------
def area_metrics(mask_T0, mask_T1r, mask_T1g):
    a0 = int(mask_T0.sum())
    a1r = int(mask_T1r.sum())
    a1g = int(mask_T1g.sum())
    d_real = a1r - a0
    d_gen = a1g - a0
    return {
        "area_T0": a0,
        "area_T1r": a1r,
        "area_T1g": a1g,
        "delta_area_real": d_real,
        "delta_area_gen": d_gen,
        "abs_area_error_T1": abs(a1g - a1r),
        "delta_area_error": abs(d_gen - d_real),
        "relative_delta_error": abs(d_gen - d_real) / (abs(d_real) + 1.0),
    }


# ----------------------------------------------------------------------
# Intensity distribution (masked)
# ----------------------------------------------------------------------
def _hist_entropy(values, bins=64):
    if values.size < 2:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def _intensity_stats(values):
    if values.size == 0:
        return {"mean": np.nan, "std": np.nan, "skew": np.nan,
                "kurt": np.nan, "entropy": np.nan, "energy": np.nan}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "skew": float(stats.skew(values)) if values.size > 1 else 0.0,
        "kurt": float(stats.kurtosis(values)) if values.size > 1 else 0.0,
        "entropy": _hist_entropy(values),
        "energy": float(np.sum(values.astype(np.float64) ** 2)),
    }


def intensity_metrics(img_T0, img_T1r, img_T1g, mask_T0, mask_T1r, mask_T1g):
    v0 = img_T0[mask_T0].astype(np.float64)
    v1r = img_T1r[mask_T1r].astype(np.float64)
    v1g = img_T1g[mask_T1g].astype(np.float64)

    s0 = _intensity_stats(v0)
    s1r = _intensity_stats(v1r)
    s1g = _intensity_stats(v1g)

    if v1r.size and v1g.size:
        wd = float(stats.wasserstein_distance(v1r, v1g))
    else:
        wd = np.nan

    return {
        "mean_T0": s0["mean"], "mean_T1r": s1r["mean"], "mean_T1g": s1g["mean"],
        "std_T0": s0["std"], "std_T1r": s1r["std"], "std_T1g": s1g["std"],
        "skew_T1r": s1r["skew"], "skew_T1g": s1g["skew"],
        "kurt_T1r": s1r["kurt"], "kurt_T1g": s1g["kurt"],
        "entropy_T0": s0["entropy"], "entropy_T1r": s1r["entropy"],
        "entropy_T1g": s1g["entropy"],
        "mean_shift_real": s1r["mean"] - s0["mean"],
        "mean_shift_gen": s1g["mean"] - s0["mean"],
        "mean_shift_error": abs((s1g["mean"] - s0["mean"])
                                - (s1r["mean"] - s0["mean"])),
        "wasserstein_T1r_T1g": wd,
    }


# ----------------------------------------------------------------------
# Shape regularity (skimage regionprops + custom spiculation index)
# ----------------------------------------------------------------------
def shape_features(image_2d, mask_2d):
    if mask_2d.sum() < 10:
        return {}
    labeled = measure.label(mask_2d.astype(np.uint8))
    props = measure.regionprops(labeled)
    if not props:
        return {}
    p = max(props, key=lambda r: r.area)  # largest CC

    perimeter = float(p.perimeter) if p.perimeter else 1.0
    circularity = 4.0 * np.pi * float(p.area) / (perimeter ** 2)

    hull = convex_hull_image(mask_2d)
    convexity = float(p.area) / max(int(hull.sum()), 1)

    minor = max(float(p.minor_axis_length), 1e-6)

    return {
        "shape_area": float(p.area),
        "shape_perimeter": float(p.perimeter),
        "shape_circularity": float(circularity),
        "shape_eccentricity": float(p.eccentricity),
        "shape_solidity": float(p.solidity),
        "shape_extent": float(p.extent),
        "shape_major_axis": float(p.major_axis_length),
        "shape_minor_axis": float(p.minor_axis_length),
        "shape_aspect_ratio": float(p.major_axis_length) / minor,
        "spiculation_idx": 1.0 - convexity,
    }


# ----------------------------------------------------------------------
# Texture: first-order + GLCM (no pyradiomics needed)
# ----------------------------------------------------------------------
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity",
              "energy", "correlation", "ASM")


def _glcm_features(image_2d, mask_2d, levels=32):
    """Average GLCM features over distances [1, 2] and 4 angles, masked."""
    masked_vals = image_2d[mask_2d]
    if masked_vals.size < 10:
        return {f"glcm_{p}": np.nan for p in GLCM_PROPS}

    vmin, vmax = float(masked_vals.min()), float(masked_vals.max())
    if vmax <= vmin:
        return {f"glcm_{p}": np.nan for p in GLCM_PROPS}

    quantized = np.zeros_like(image_2d, dtype=np.uint8)
    quantized[mask_2d] = np.clip(
        ((image_2d[mask_2d].astype(np.float64) - vmin)
         / (vmax - vmin) * (levels - 1)).round(),
        0, levels - 1,
    ).astype(np.uint8)

    distances = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(quantized, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)

    return {f"glcm_{prop}": float(graycoprops(glcm, prop).mean())
            for prop in GLCM_PROPS}


def radiomic_features(image_2d, mask_2d):
    """First-order + GLCM texture features for one (image, mask) pair."""
    if mask_2d.sum() < 10:
        return {}
    vals = image_2d[mask_2d].astype(np.float64)

    fo = _intensity_stats(vals)
    out = {f"firstorder_{k}": v for k, v in fo.items()}
    out["firstorder_min"] = float(vals.min())
    out["firstorder_max"] = float(vals.max())
    out["firstorder_range"] = float(vals.max() - vals.min())
    out["firstorder_p10"] = float(np.percentile(vals, 10))
    out["firstorder_p90"] = float(np.percentile(vals, 90))
    out["firstorder_iqr"] = float(np.percentile(vals, 75)
                                  - np.percentile(vals, 25))

    out.update(_glcm_features(image_2d, mask_2d))
    return out
