"""NLST paired-nodule dataset for the HemExp pipeline.

Adapts HemExp's "baseline -> follow-up, conditioned on tabular clinical
variables + a binary oracle flag" formulation to the NAMD/NLST task:

    HemExp                              NLST (this file)
    ------------------------------      --------------------------------
    baseline CT slice (NIfTI)           year-1 nodule patch (left half of jpg)
    follow-up CT slice (NIfTI)          year-2 nodule patch (right half)
    iph / ivh segmentation channels     none (n_aux == 0)
    7 clinical variables (csv)          biomarker + EHR vector (demo/*.npy)
    expansion_flag (oracle 0/1)         malignancy label

Layout consumed (matches NAMD's `NLST_with_second_large_cleaned`):

    base_path/
      image/{train,val,test}/<subject_id>.jpg    256x128 grayscale, [baseline | followup]
      demo/{train,val,test}/<subject_id>.npy     32 floats = 16 first-year, 16 second-year

CONDITIONING IS FIRST-YEAR ONLY. The second-year half of the demo vector is
read solely to assert the label is time-invariant and is never returned; the
follow-up image is the prediction target, never an input.
"""

import os
from typing import List, Optional, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

# Mirrors dataset.Features._features_names in the NAMD repo root. Kept as a
# literal rather than imported so this file does not drag in pytorch_lightning.
FEATURE_NAMES: List[str] = [
    "SCT_PRE_ATT",    # nodule attenuation / texture   \
    "SCT_EPI_LOC",    # lobe / epicentre location       |
    "SCT_LONG_DIA",   # long-axis diameter (mm)         |- imaging biomarkers
    "SCT_PERP_DIA",   # perpendicular diameter (mm)     |
    "SCT_MARGINS",    # margin morphology              /
    "unique_ids",     # patient identifier   <- not clinical
    "year",           # screening year       <- not clinical
    "age",            # \
    "diagemph",       #  |
    "gender",         #  |
    "famfather",      #  |- EHR / demographics
    "fammother",      #  |
    "fambrother",     #  |
    "famsister",      #  |
    "famchild",       # /
    "label",          # malignancy  <- carried by the expansion-flag path
]

BIOMARKER_FEATURES = FEATURE_NAMES[0:5]
EHR_FEATURES = FEATURE_NAMES[7:15]

# Default conditioning vector: every biomarker + every EHR field, i.e. exactly
# the content NAMD's conditional diffusion sees, minus the two pure identifiers
# (`unique_ids` is ~1e5 and would dominate the first Linear) and `label`, which
# HemExp delivers separately through the flag embedding. Set
# `excluded_features: []` in the config to recover literal NAMD parity (16-dim,
# identifiers included), matching ehr_encoder.EHREncoder(in_dim=16).
DEFAULT_EXCLUDED_FEATURES = ["unique_ids", "year", "label"]

LABEL_INDEX = FEATURE_NAMES.index("label")


def _impute(values: np.ndarray) -> np.ndarray:
    """NaN policy copied verbatim from dataset.Features.__init__."""
    out = values.astype(np.float64).copy()
    for i, name in enumerate(FEATURE_NAMES):
        if not np.isnan(out[i]):
            continue
        if "fam" in name:
            out[i] = 0.0
        elif name == "SCT_MARGINS":
            out[i] = 3.0
        else:
            raise ValueError(f"unimputable NaN in feature '{name}'")
    return out


def resolve_feature_indices(
    selected_features: Optional[Sequence[str]] = None,
    excluded_features: Optional[Sequence[str]] = None,
) -> List[int]:
    """Index list into the 16-wide feature vector, in canonical order.

    Same selection semantics as dataset.Features.get_features: an empty
    `selected_features` means "all", and exclusion is applied afterwards.
    """
    selected = list(selected_features) if selected_features else list(FEATURE_NAMES)
    excluded = set(excluded_features if excluded_features is not None else DEFAULT_EXCLUDED_FEATURES)

    unknown = (set(selected) | excluded) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown feature name(s): {sorted(unknown)}")

    idx = [i for i, n in enumerate(FEATURE_NAMES) if n in selected and n not in excluded]
    if not idx:
        raise ValueError("feature selection left an empty conditioning vector")
    return idx


def list_subject_ids(base_path: str, split: str) -> List[str]:
    image_dir = os.path.join(base_path, "image", split)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"missing image split directory: {image_dir}")
    return sorted(f[:-4] for f in os.listdir(image_dir) if f.endswith(".jpg"))


class NLSTPairedDataset(Dataset):
    """Paired year-1 / year-2 NLST nodule patches with first-year conditioning.

    __getitem__ returns the aux-channel tuple the HemExp Trainer consumes:

        baseline   float32 [K, H, W]      in [0, 1]
        followup   float32 [K, H, W]      in [0, 1]
        aux_bl     float32 [0, K, H, W]   empty: NLST has no segmentations
        aux_fu     float32 [0, K, H, W]   empty
        clinical   float32 [D]            first-year biomarkers + EHR
        flag       int64   [1]            malignancy / oracle expansion flag

    K (`context_slices`) is 1 for NLST: the patches are single 2D slices, so
    the neighbouring-slice stacking HemExp uses for 2.5D CT collapses to a
    single plane. K > 1 simply repeats that plane so the UNet's channel
    arithmetic is unchanged.
    """

    def __init__(
        self,
        base_path: str,
        split: str = "train",
        image_size: Optional[int] = None,
        resize_mode: str = "resize",
        context_slices: int = 1,
        selected_features: Optional[Sequence[str]] = None,
        excluded_features: Optional[Sequence[str]] = None,
        clinical_mean: Optional[Sequence[float]] = None,
        clinical_std: Optional[Sequence[float]] = None,
        augment_flip: bool = False,
        augment_rotation: bool = False,
        rotation_range_deg: float = 0.0,
    ):
        super().__init__()

        self.base_path = base_path
        self.split = split
        self.image_size = image_size

        # How to reach image_size from the native 128x128 patch.
        #   "resize" -- bilinear downsample, keeps the whole field of view
        #   "crop"   -- centre crop, MATCHES the NAMD benchmark: every arm in
        #               nodule_quality_eval was built on patch[32:96, 32:96],
        #               byte-identical to a centre crop and NOT to any
        #               downsample. Use "crop" for a comparable arm.
        if resize_mode not in ("resize", "crop"):
            raise ValueError(f"resize_mode must be 'resize' or 'crop', got {resize_mode!r}")
        self.resize_mode = resize_mode
        self.context_slices = int(context_slices)
        self.augment_flip = bool(augment_flip)
        self.augment_rotation = bool(augment_rotation)
        self.rotation_range_deg = float(rotation_range_deg)

        # NLST carries no segmentation masks; the Trainer branches on this.
        self.n_aux = 0

        self.feature_indices = resolve_feature_indices(selected_features, excluded_features)
        self.feature_names = [FEATURE_NAMES[i] for i in self.feature_indices]

        self.clinical_mean = None if clinical_mean is None else np.asarray(clinical_mean, dtype=np.float32)
        self.clinical_std = None if clinical_std is None else np.asarray(clinical_std, dtype=np.float32)
        if self.clinical_mean is not None and len(self.clinical_mean) != len(self.feature_indices):
            raise ValueError(
                f"clinical_mean has {len(self.clinical_mean)} entries but the selection is "
                f"{len(self.feature_indices)}-dimensional ({self.feature_names})"
            )

        self.samples = self._index_samples()
        if not self.samples:
            raise RuntimeError(f"no usable subjects found for split '{split}' under {base_path}")

        # Consumed by ExpansionBalancedSampler and the validation selection.
        self.sample_labels = [s["label"] for s in self.samples]

    # ------------------------------------------------------------------ index

    def _index_samples(self) -> List[dict]:
        samples = []
        for subject_id in list_subject_ids(self.base_path, self.split):
            image_path = os.path.join(self.base_path, "image", self.split, f"{subject_id}.jpg")
            demo_path = os.path.join(self.base_path, "demo", self.split, f"{subject_id}.npy")
            if not os.path.exists(demo_path):
                continue

            info = np.load(demo_path)
            half = len(info) // 2
            if half != len(FEATURE_NAMES):
                raise ValueError(
                    f"{demo_path}: expected {2 * len(FEATURE_NAMES)} values, got {len(info)}"
                )

            try:
                first_year = _impute(info[:half])
                second_year = _impute(info[half:])
            except ValueError as e:
                print(f"[skip] {subject_id}: {e}")
                continue

            # The flag is a patient-level outcome, identical in both halves for
            # every NLST subject; asserting it keeps the first-year-only
            # guarantee honest rather than assumed.
            if first_year[LABEL_INDEX] != second_year[LABEL_INDEX]:
                raise ValueError(
                    f"{subject_id}: label differs between years "
                    f"({first_year[LABEL_INDEX]} vs {second_year[LABEL_INDEX]}); the flag would "
                    "leak follow-up information"
                )

            samples.append(
                {
                    "subject_id": subject_id,
                    "image_path": image_path,
                    # Only the first-year half is retained. The second-year
                    # features are deliberately dropped here so no downstream
                    # code can reach them.
                    "features": first_year[self.feature_indices].astype(np.float32),
                    "label": int(first_year[LABEL_INDEX]),
                }
            )
        return samples

    # ----------------------------------------------------------- clinical fit

    def compute_clinical_stats(self):
        """Per-feature mean/std over this split, for z-scoring the MLP input."""
        feats = np.stack([s["features"] for s in self.samples], axis=0)
        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        std[std < 1e-6] = 1.0  # constant column -> leave centred, unscaled
        return mean.astype(np.float32), std.astype(np.float32)

    def set_clinical_stats(self, mean, std):
        self.clinical_mean = np.asarray(mean, dtype=np.float32)
        self.clinical_std = np.asarray(std, dtype=np.float32)

    # -------------------------------------------------------------- accessors

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pair(self, image_path: str):
        image = Image.open(image_path)
        arr = np.asarray(image, dtype=np.float32) / 255.0  # [0, 1], as HemExp's CT windowing yields

        if arr.ndim == 3:  # a stray RGB jpg: collapse to luminance-free mean
            arr = arr.mean(axis=-1)

        h, w = arr.shape
        if w % 2 != 0:
            raise ValueError(f"{image_path}: width {w} is odd, cannot split into a pair")

        baseline = torch.from_numpy(arr[:, : w // 2].copy())[None]   # [1, H, W]
        followup = torch.from_numpy(arr[:, w // 2 :].copy())[None]

        if self.image_size is not None and baseline.shape[-1] != self.image_size:
            if self.resize_mode == "crop":
                h, w = baseline.shape[-2:]
                if self.image_size > min(h, w):
                    raise ValueError(
                        f"cannot crop {h}x{w} to {self.image_size}; crop must be smaller"
                    )
                top = (h - self.image_size) // 2
                left = (w - self.image_size) // 2
                sl_h = slice(top, top + self.image_size)
                sl_w = slice(left, left + self.image_size)
                baseline = baseline[..., sl_h, sl_w]
                followup = followup[..., sl_h, sl_w]
            else:
                size = [self.image_size, self.image_size]
                baseline = TF.resize(baseline, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
                followup = TF.resize(followup, size, interpolation=InterpolationMode.BILINEAR, antialias=True)

        return baseline, followup

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        baseline, followup = self._load_pair(s["image_path"])

        # Identical geometric augmentation on both halves: the model predicts a
        # residual between them, so any desynchronised transform is pure noise.
        if self.augment_flip and torch.rand(()) < 0.5:
            baseline = TF.hflip(baseline)
            followup = TF.hflip(followup)

        if self.augment_rotation and self.rotation_range_deg > 0:
            angle = float((torch.rand(()) * 2 - 1).item() * self.rotation_range_deg)
            baseline = TF.rotate(baseline, angle, interpolation=InterpolationMode.BILINEAR)
            followup = TF.rotate(followup, angle, interpolation=InterpolationMode.BILINEAR)

        if self.context_slices > 1:
            baseline = baseline.repeat(self.context_slices, 1, 1)
            followup = followup.repeat(self.context_slices, 1, 1)

        h, w = baseline.shape[-2:]
        empty_aux = torch.zeros(0, self.context_slices, h, w, dtype=torch.float32)

        clinical = s["features"]
        if self.clinical_mean is not None:
            clinical = (clinical - self.clinical_mean) / self.clinical_std

        return (
            baseline.float(),
            followup.float(),
            empty_aux,
            empty_aux.clone(),
            torch.from_numpy(np.ascontiguousarray(clinical)).float(),
            torch.tensor([s["label"]], dtype=torch.long),
        )


class NLSTAutoencoderDataset(Dataset):
    """Flat single-image view of the same data, for first-stage AE training.

    Each paired jpg contributes two independent training images (baseline and
    follow-up), which is what the VAE should see: it models the marginal
    distribution of nodule patches, not the pairing.
    """

    def __init__(
        self,
        base_path: str,
        split: str = "train",
        image_size: Optional[int] = None,
        resize_mode: str = "resize",
        augment_flip: bool = False,
        augment_rotation: bool = False,
        rotation_range_deg: float = 0.0,
    ):
        super().__init__()
        self.pairs = NLSTPairedDataset(
            base_path=base_path,
            split=split,
            image_size=image_size,
            resize_mode=resize_mode,
            context_slices=1,
            augment_flip=False,
            augment_rotation=False,
        )
        self.augment_flip = bool(augment_flip)
        self.augment_rotation = bool(augment_rotation)
        self.rotation_range_deg = float(rotation_range_deg)

    def __len__(self) -> int:
        return 2 * len(self.pairs)

    def __getitem__(self, idx: int):
        s = self.pairs.samples[idx // 2]
        baseline, followup = self.pairs._load_pair(s["image_path"])
        image = baseline if idx % 2 == 0 else followup

        if self.augment_flip and torch.rand(()) < 0.5:
            image = TF.hflip(image)
        if self.augment_rotation and self.rotation_range_deg > 0:
            angle = float((torch.rand(()) * 2 - 1).item() * self.rotation_range_deg)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)

        return {"image": image.float(), "filename": s["subject_id"]}
