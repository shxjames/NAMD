"""Metric computations: area, intensity distribution, shape, full radiomics.

Uses PyRadiomics for shape2D + the six standard texture classes
(firstorder, GLCM, GLRLM, GLSZM, NGTDM, GLDM).
"""
import logging

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from scipy import stats
from skimage.morphology import convex_hull_image

logging.getLogger("radiomics").setLevel(logging.ERROR)


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
def _intensity_stats(values):
    if values.size == 0:
        return {"mean": np.nan, "std": np.nan, "skew": np.nan, "kurt": np.nan}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "skew": float(stats.skew(values)) if values.size > 1 else 0.0,
        "kurt": float(stats.kurtosis(values)) if values.size > 1 else 0.0,
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
        "mean_shift_real": s1r["mean"] - s0["mean"],
        "mean_shift_gen": s1g["mean"] - s0["mean"],
        "mean_shift_error": abs((s1g["mean"] - s0["mean"])
                                - (s1r["mean"] - s0["mean"])),
        "wasserstein_T1r_T1g": wd,
    }


# ----------------------------------------------------------------------
# SimpleITK helper
# ----------------------------------------------------------------------
def _to_sitk_2d(arr2d):
    return sitk.GetImageFromArray(arr2d[None, ...])


# ----------------------------------------------------------------------
# Shape (PyRadiomics shape2D + spiculation index)
# ----------------------------------------------------------------------
_shape_extractor = None


def _get_shape_extractor():
    global _shape_extractor
    if _shape_extractor is None:
        ex = featureextractor.RadiomicsFeatureExtractor()
        ex.disableAllFeatures()
        ex.enableFeatureClassByName("shape2D")
        ex.settings["force2D"] = True
        ex.settings["force2Ddimension"] = 0
        _shape_extractor = ex
    return _shape_extractor


def _is_pyradiomics_safe(mask_2d, min_pixels=10):
    """Check the mask spans at least 2 rows and 2 columns. PyRadiomics
    needs a 2D ROI bounding box; thin strips or single-pixel lines fail."""
    if mask_2d.sum() < min_pixels:
        return False
    ys, xs = np.where(mask_2d)
    return (ys.max() - ys.min()) >= 2 and (xs.max() - xs.min()) >= 2


def shape_features(image_2d, mask_2d):
    if not _is_pyradiomics_safe(mask_2d):
        return {}
    img_sitk = _to_sitk_2d(image_2d.astype(np.float32))
    mask_sitk = _to_sitk_2d(mask_2d.astype(np.uint8))
    feats = _get_shape_extractor().execute(img_sitk, mask_sitk)
    out = {}
    for k, v in feats.items():
        if not k.startswith("original_shape2D_"):
            continue
        try:
            out[k.replace("original_shape2D_", "shape_")] = float(v)
        except (TypeError, ValueError):
            continue
    hull = convex_hull_image(mask_2d)
    convexity = mask_2d.sum() / max(int(hull.sum()), 1)
    out["spiculation_idx"] = 1.0 - float(convexity)
    return out


# ----------------------------------------------------------------------
# Full radiomics: firstorder + GLCM + GLRLM + GLSZM + NGTDM + GLDM
# ----------------------------------------------------------------------
_full_extractor = None


def _get_full_extractor():
    global _full_extractor
    if _full_extractor is None:
        ex = featureextractor.RadiomicsFeatureExtractor()
        ex.disableAllFeatures()
        ex.enableFeatureClassByName("firstorder")
        ex.enableFeatureClassByName("glcm")
        ex.enableFeatureClassByName("glrlm")
        ex.enableFeatureClassByName("glszm")
        ex.enableFeatureClassByName("ngtdm")
        ex.enableFeatureClassByName("gldm")
        ex.settings["force2D"] = True
        ex.settings["force2Ddimension"] = 0
        _full_extractor = ex
    return _full_extractor


def radiomic_features(image_2d, mask_2d):
    if not _is_pyradiomics_safe(mask_2d):
        return {}
    img_sitk = _to_sitk_2d(image_2d.astype(np.float32))
    mask_sitk = _to_sitk_2d(mask_2d.astype(np.uint8))
    feats = _get_full_extractor().execute(img_sitk, mask_sitk)
    out = {}
    for k, v in feats.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out
