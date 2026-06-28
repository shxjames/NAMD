"""MedSAM 2 nodule segmentation for 2D CT patches.

MedSAM 2 (Ma et al., 2025; bowang-lab/MedSAM2) is the current SOTA for
bounding-box prompted 2D lung tumor / nodule segmentation
(Dice ~0.91 on Lung1 vs ~0.64 for the original MedSAM).

Setup:
  1. Clone & install MedSAM 2 (bundles SAM2 code + medical configs):
       git clone https://github.com/bowang-lab/MedSAM2.git
       cd MedSAM2
       pip install -e ".[dev]"
  2. Checkpoint is auto-downloaded from HuggingFace
     (repo_id="wanglab/MedSAM2") on first use.
"""
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max


def _select_central_component(mask, connectivity=2):
    """Keep only the connected component closest to the patch center, fill holes.

    `connectivity=2` (default) uses 8-neighbor connectivity (diagonal touches
    count). Pass `connectivity=1` for 4-neighbor (cardinal-only) — strict,
    drops fragments joined only by diagonal pixels."""
    if not mask.any():
        return mask.astype(bool)
    labeled = measure.label(mask.astype(np.uint8), connectivity=connectivity)
    if labeled.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    props = measure.regionprops(labeled)
    best = max(
        props,
        key=lambda p: p.area - 5 * np.hypot(p.centroid[0] - cy, p.centroid[1] - cx),
    )
    out = (labeled == best.label)
    return ndi.binary_fill_holes(out).astype(bool)


def _has_edge_contact(mask, min_edge_pixels=3):
    """True if the mask substantially touches any patch boundary — heuristic
    for "chest wall is in the mask" (real nodules in centered patches don't
    reach the patch boundary)."""
    if not mask.any():
        return False
    return max(
        int(mask[0, :].sum()),
        int(mask[-1, :].sum()),
        int(mask[:, 0].sum()),
        int(mask[:, -1].sum()),
    ) >= min_edge_pixels


def _trim_wall_bright_in_mask(image, mask, percentile=85, min_pixels=5):
    """Drop in-mask pixels whose intensity matches the brightest off-mask
    pixels (i.e., chest wall brightness).

    Many patches have a clearly bimodal intensity histogram: chest wall is
    much brighter than the nodule core. We learn the lower bound of "wall
    brightness" from the OFF-mask area (which contains the wall but not
    the nodule) and remove any in-mask pixel above that bound.
    """
    if mask.sum() < min_pixels:
        return mask
    off_mask_vals = image[~mask]
    if off_mask_vals.size < 50:
        return mask
    wall_lower = np.percentile(off_mask_vals, percentile)
    refined = mask & (image < wall_lower)
    refined = _select_central_component(refined)
    if refined.sum() < min_pixels:
        return mask  # too aggressive, abort
    return refined


def _detect_lung_wall(image, central_margin=10):
    """Segment the chest wall as bright pixels connected to the patch
    boundary, then return it as a mask to be subtracted from the nodule mask.

    Key idea: patches are nodule-centered, so the chest wall always touches
    one or more patch edges by construction. Flood-filling bright pixels
    from the patch boundary therefore reaches every chest wall pixel. A
    fixed central protected zone (radius `central_margin` around the patch
    center) blocks the flood-fill from entering the nodule -- so even for
    juxtapleural cases where wall and nodule form one connected bright
    region, only the wall band is captured.

    Returns
    -------
    wall_mask : bool [H, W]
        True where the chest wall was detected.
    """
    H, W = image.shape
    try:
        thresh = filters.threshold_otsu(image)
    except ValueError:
        return np.zeros((H, W), dtype=bool)

    bright = image > thresh

    # Protected central zone — keeps the nodule out of the wall mask.
    yy, xx = np.mgrid[:H, :W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    central_zone = np.hypot(yy - cy, xx - cx) <= central_margin
    bright_outside = (bright & ~central_zone).astype(np.uint8)

    # Boundary seeds: bright pixels on any patch edge, outside central zone.
    seed = np.zeros((H, W), dtype=np.uint8)
    seed[0, :] = bright_outside[0, :]
    seed[-1, :] = bright_outside[-1, :]
    seed[:, 0] = bright_outside[:, 0]
    seed[:, -1] = bright_outside[:, -1]
    if seed.sum() == 0:
        return np.zeros((H, W), dtype=bool)

    # Morphological reconstruction by dilation = flood-fill from seed,
    # constrained to stay inside `bright_outside`.
    wall = morphology.reconstruction(seed, bright_outside, method="dilation")
    return wall.astype(bool)


def _hard_radial_cap(mask, max_radius):
    """Drop any mask pixel beyond `max_radius` of the patch center.

    Patches are nodule-centered by construction, so the chest wall lies in
    the outer ring -- a fixed radial cap from patch center is the most
    direct way to exclude it when geometric methods fail to separate a
    broadly-attached nodule and wall."""
    if not mask.any():
        return mask
    H, W = mask.shape
    yy, xx = np.mgrid[:H, :W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    dist = np.hypot(yy - cy, xx - cx)
    return mask & (dist <= max_radius)


def _isolate_nodule_via_erode_reconstruct(mask, erode_radius=2, min_core_pixels=5):
    """Separate a juxtapleural nodule from an attached chest wall.

    Step 1: erode the mask. Any thin pixel bridge (such as the neck where
            a chest wall band joins a round nodule) is broken; the nodule
            core and the wall band become separate connected components.
    Step 2: pick the connected component of the eroded mask that is closest
            to the patch center. That's the nodule core.
    Step 3: morphological reconstruction by dilation grows the core back
            WITHIN the original mask. Because the bridge is broken in the
            eroded image, the dilation cannot cross it; the chest-wall band
            is left out of the final mask.

    For isolated nodules with no bridge to break, the erode-reconstruct
    cycle returns the original mask unchanged.
    """
    if mask.sum() < 20:
        return mask
    eroded = morphology.binary_erosion(
        mask, footprint=morphology.disk(erode_radius)
    )
    if eroded.sum() < min_core_pixels:
        return mask  # erosion too aggressive -- abort

    labeled = measure.label(eroded)
    if labeled.max() == 0:
        return mask

    props = measure.regionprops(labeled)
    H, W = mask.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    best = max(
        props,
        key=lambda p: p.area - 5 * np.hypot(p.centroid[0] - cy, p.centroid[1] - cx),
    )
    core = (labeled == best.label).astype(np.uint8)

    reconstructed = morphology.reconstruction(
        core, mask.astype(np.uint8), method="dilation"
    )
    return reconstructed.astype(bool)


def segment_from_prior(image, prior_mask, dilation=2, min_pixels=5):
    """Otsu segmentation guided by a trusted spatial prior — no MedSAM call.

    For T0 patches paired with a reliable T1 mask: trying MedSAM on T0
    often fails when the baseline nodule is small or invisible (MedSAM
    latches onto adjacent vessels / chest wall instead). Restricting
    segmentation to the dilated T1 mask region eliminates that failure
    mode while still letting Otsu adapt to the T0 image's contrast.

    Returns an empty mask if no bright signal exists inside the prior
    region — which is the correct answer for nodules that grew from
    nothing at baseline.
    """
    prior_mask = prior_mask.astype(bool)
    if prior_mask.sum() < min_pixels:
        return prior_mask

    search = morphology.binary_dilation(
        prior_mask, footprint=morphology.disk(dilation)
    )
    vals = image[search]
    if vals.size < 20:
        return prior_mask
    try:
        thresh = filters.threshold_otsu(vals)
    except ValueError:
        return prior_mask

    mask = (image > thresh) & search
    mask = _select_central_component(mask)
    if mask.sum() < min_pixels:
        return prior_mask

    # Geometric wall subtraction; gate intensity trim on whether it
    # actually removed pixels (avoids hurting isolated nodules).
    wall = _detect_lung_wall(image, central_margin=10)
    wall_pixels_removed = 0
    if wall.any():
        mask_before = mask.copy()
        mask = mask & ~wall
        wall_pixels_removed = int((mask_before & ~mask).sum())
        mask = _select_central_component(mask)
        if mask.sum() < min_pixels:
            return prior_mask

    if wall_pixels_removed > 5:
        mask = _trim_wall_bright_in_mask(image, mask, percentile=85)
        if mask.sum() < min_pixels:
            return prior_mask
    # Final consolidation: drop diagonally-touching fragments (no-op for
    # clean 4-connected masks).
    mask = _select_central_component(mask, connectivity=1)
    return mask


def _otsu_refine_boundary(image, init_mask, dilation=3, min_pixels=10):
    """Refine boundary by Otsu thresholding within a dilated initial mask.

    MedSAM 2's mask provides correct *localization* (and excludes chest wall
    by construction, thanks to the tight box). Otsu inside that dilated
    region then snaps the boundary to the actual nodule/parenchyma intensity
    transition for each individual sample — adaptive to bright solid nodules
    and dimmer ground-glass nodules alike.
    """
    if init_mask.sum() < min_pixels:
        return init_mask

    # Search region: dilated MedSAM 2 mask. Lets Otsu reach pixels that
    # MedSAM 2 just missed, but stays well away from chest wall.
    search = morphology.binary_dilation(
        init_mask, footprint=morphology.disk(dilation)
    )

    vals = image[search]
    if vals.size < 20:
        return init_mask
    try:
        thresh = filters.threshold_otsu(vals)
    except ValueError:
        return init_mask

    refined = (image > thresh) & search
    refined = _select_central_component(refined)
    if refined.sum() < min_pixels:
        return init_mask

    # Light morphological smoothing — close 1-pixel notches, fill holes.
    refined = morphology.binary_closing(refined, footprint=morphology.disk(1))
    refined = ndi.binary_fill_holes(refined)
    return refined.astype(bool)


HF_REPO = "wanglab/MedSAM2"
HF_FILENAME = "MedSAM2_latest.pt"
DEFAULT_CFG = "configs/sam2.1_hiera_t512.yaml"  # tiny Hiera @ 512 input — MedSAM2 default


def _supports_bf16(device):
    if device == "cpu":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8  # Ampere (sm_80) and newer have native bf16


class MedSAM2Segmenter:
    def __init__(self, checkpoint_path=None, model_cfg=None, device=None,
                 hf_repo=HF_REPO, hf_filename=HF_FILENAME):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Pascal (GTX 10xx, e.g. 1080 Ti) lacks bf16 / Flash Attention.
        # Force fp32 + eager attention to keep inference correct on older GPUs.
        if self.device.startswith("cuda") and not _supports_bf16(self.device):
            os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "0")
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            self._autocast_dtype = None  # fp32
        else:
            self._autocast_dtype = torch.bfloat16

        if checkpoint_path is None:
            checkpoint_path = hf_hub_download(
                repo_id=hf_repo,
                filename=hf_filename,
                cache_dir="./checkpoints",
            )
        if model_cfg is None:
            model_cfg = DEFAULT_CFG

        sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def _autocast_ctx(self):
        if self._autocast_dtype is None:
            return torch.autocast(device_type="cuda", enabled=False)
        return torch.autocast(device_type="cuda", dtype=self._autocast_dtype)

    @torch.no_grad()
    def segment(self, image_2d, box=None):
        """Segment a single 2D nodule patch.

        image_2d : [H, W] uint8 array (already nodule-centered).
        box      : [x0, y0, x1, y1] in original pixel coords;
                   None -> tight centered box (~30% of patch area)
                   so MedSAM 2 focuses on the central nodule and
                   ignores chest wall / vessels at the patch edges.

        Returns  : [H, W] bool mask.
        """
        H, W = image_2d.shape
        if box is None:
            # ~20x20 centered box on a 64x64 patch (margin = 22 pixels each side).
            m = int(min(H, W) * 0.34)
            box = np.array([m, m, W - m, H - m], dtype=np.float32)
        box = np.asarray(box, dtype=np.float32)

        img_rgb = np.repeat(image_2d[:, :, None], 3, axis=-1).astype(np.uint8)

        with self._autocast_ctx():
            self.predictor.set_image(img_rgb)
            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
        raw_mask = masks[0].astype(bool)

        # Post-process pipeline:
        #   1. Single central connected component.
        #   2. Otsu refinement inside a small (2-px) dilation -- adaptive
        #      intensity boundary, won't reach distant chest wall.
        #   3. If the mask touches the patch boundary (juxtapleural):
        #        a. erode-then-reconstruct to break thin bridges, then
        #        b. hard radial cap from patch center -- excludes any wall
        #           that survived (broad attachments etc).
        #   4. Final central-CC selection to guarantee one closed shape.
        H, W = image_2d.shape
        mask = _select_central_component(raw_mask)
        mask = _otsu_refine_boundary(image_2d, mask, dilation=2)

        # Geometric wall subtraction (Option A): bright pixels flood-filled
        # from the patch boundary, with a central protected zone.
        # The intensity-based trim (Option B) is gated on this step having
        # actually changed the mask -- otherwise B over-trims isolated
        # nodules with bright vessels nearby.
        wall = _detect_lung_wall(image_2d, central_margin=10)
        wall_pixels_removed = 0
        if wall.any():
            mask_before = mask.copy()
            mask = mask & ~wall
            wall_pixels_removed = int((mask_before & ~mask).sum())
            mask = _select_central_component(mask)

        if wall_pixels_removed > 5:
            mask = _trim_wall_bright_in_mask(image_2d, mask, percentile=85)

        if _has_edge_contact(mask):
            mask = _isolate_nodule_via_erode_reconstruct(mask, erode_radius=2)
            mask = _hard_radial_cap(mask, max_radius=min(H, W) // 4 + 2)
        # Final consolidation pass with 4-connectivity drops any fragment
        # still hanging on by a diagonal pixel touch (e.g. sample 104_1).
        # Clean masks are 1 CC under both 4 and 8 connectivity, so this is
        # a no-op for them.
        mask = _select_central_component(mask, connectivity=1)
        return mask
