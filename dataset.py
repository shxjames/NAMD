import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
import os
import pytorch_lightning as pl
from typing import Optional
from utils import image_translation, image_rotation, band_pass_filter

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

class Features:
    """Ultra-simple feature accessor using namedtuple-like approach."""
    def __init__(self, feature_array: np.ndarray):
        self._features_names = ['SCT_PRE_ATT', 'SCT_EPI_LOC', 'SCT_LONG_DIA', 'SCT_PERP_DIA', 'SCT_MARGINS', 'unique_ids', 'year',
                'age', 'diagemph', 'gender', 'famfather', 'fammother', 'fambrother', 'famsister', 'famchild', 'label']
        
        self.array = feature_array
        assert len(feature_array) == len(self._features_names), "Feature array length does not match features names length"
        for i, name in enumerate(self._features_names):
            if i < len(feature_array):
                if np.isnan(feature_array[i]):
                    if 'fam' in name:
                        setattr(self, name, 0)
                    elif name == "SCT_MARGINS":
                        setattr(self, name, 3.0)
                    else:
                        print(f"NaN found in {name} for {feature_array[5]}")
                        setattr(self, name, feature_array[i])
                else:
                    setattr(self, name, feature_array[i])

        # reset self.array
        self.array = np.array([getattr(self, name) for name in self._features_names])

    def has_nan(self):
        return np.isnan(self.array).any()

    def __getitem__(self, idx):
        return self.array[idx]

    def get_features(self, selected_features: list = [], excluded_features: list=[]):
        if not selected_features:
            selected_features = self._features_names

        selected_features = [name for name in self._features_names if name not in excluded_features and name in selected_features]

        return np.array([getattr(self, feature) for feature in selected_features])

    def __repr__(self):
        return f"Features(array={self.array})"

class Demographic_Features:
    def __init__(self, path, construct_prompt=False, retriever=None):
        self.path = path

        info = np.load(path)

        first_year, second_year = info[:len(info)//2], info[len(info)//2:]
        
        # Create feature accessors for easy named access
        self.first_year = Features(first_year)
        self.second_year = Features(second_year)

        self.label = self.second_year.label
        self.sentence = None
        self.sentence_encoded = None

        if construct_prompt:
            if retriever is None:
                raise ValueError("retriever is required when construct_prompt=True")
            self.sentence = self.construct_prompt(self.first_year.array, self.second_year.array)
            self.sentence_encoded = self._encode_prompt(self.sentence, retriever)

    @staticmethod
    def _encode_prompt(sentence, retriever):
        target_length = retriever.max_length - retriever.context_length
        if target_length <= 0:
            raise ValueError(
                f"Invalid prompt token length: max_length={retriever.max_length}, "
                f"context_length={retriever.context_length}"
            )

        token_ids = retriever.tokenizer.encode(sentence)
        if len(token_ids) > target_length:
            token_ids = (
                token_ids[:target_length - 1] + [retriever.tokenizer.tokenizer.vocab_size]
            )

        encoded = torch.tensor(token_ids, dtype=torch.long)
        pad_length = max(0, target_length - encoded.shape[0])
        if pad_length > 0:
            encoded = torch.cat([encoded, torch.zeros(pad_length, dtype=torch.long)], dim=0)
        return encoded

    @staticmethod
    def construct_prompt(ehr, ehr_2=None, scale_factor=1.0):
        # Helper to extract scalar value (works for both numpy and torch)
        def get_val(x):
            return x.item() if hasattr(x, 'item') else float(x)
        
        sentence = 'Lung cancer screening with low dose computed tomography performed for this ' + str(int(get_val(ehr[7]))) + ' years old '
        
        if ehr[9] == 1:
            sentence = sentence +  'male'
        else:
            sentence = sentence +  'female'

        if ehr[8] == 1:
            sentence = sentence +  ' with prior diagnosis of emphysema'
        
        if ehr[10] or ehr[11] or ehr[12] or ehr[13] or ehr[14] == 1:
            
            if ehr[8] == 1:
                sentence = sentence + ' and family history of cancer. '
            else:
                sentence = sentence +  ' with family history of cancer. '
        else:
            sentence = sentence +  '. '


        ### A part-solid nodule, 27 mm in size, is present in the superior segment of the right lower lobe
              
        sentence = sentence + 'A '

        if ehr[0] == 1:
            sentence = sentence + 'soft '
        elif ehr[0] == 2:
            sentence = sentence + 'ground glass '
        else:
            sentence = sentence + 'part solid '
            

        sentence = sentence + 'nodule, with '


        if ehr[4] == 1:
            sentence = sentence + 'spiculated margin, '
        elif ehr[4] == 2:
            sentence = sentence + 'smooth margin, '
        else:
            sentence = sentence + 'poorly defined margin, '


        sentence = sentence + str(get_val(ehr[2])) + ' mm in size '

        if ehr[1] == 1:
            sentence = sentence + 'is present in the right upper lobe.'
        elif ehr[1] == 2:
            sentence = sentence + 'is present in the right middle lobe.'
        elif ehr[1] == 3:
            sentence = sentence + 'is present in the right lower lobe.'
        elif ehr[1] == 4:
            sentence = sentence + 'is present in the left upper lobe.'
        elif ehr[1] == 5:
            sentence = sentence + 'is present in the lingula.'
        elif ehr[1] == 6:
            sentence = sentence + 'is present in the left lower lobe.'
        #else:
        #    sentence = sentence + 'is present in the the other side of lung.'

        # if ehr_2 is not None:
        #     sentence = sentence + ' One year later, the size of nodule will become ' + str(get_val(ehr_2[2]) * scale_factor) + ' mm.'

        return sentence

def normalize_image(image: np.ndarray):
    image = image.astype(np.float32)
    image = image / 255.0
    image = 2 * image - 1
    return image

def _process_single_image_augmentation(args):
    """
    Worker function for multiprocessing augmentation.
    Process a single image with its feature and label.
    """
    image, feature, label, split, translate, rotate, kwargs = args
    
    h, w = image.shape[-2], image.shape[-1]
    is_paired = (w == 2 * h)
    
    results = {
        'images': [],
        'features': [],
        'labels': []
    }
    
    if is_paired:
        # Handle paired images (cond_img and gt_img)
        cond_img, gt_img = image[:, :, :w//2], image[:, :, w//2:]
        centre_cond_img, translated_cond_images = image_translation(cond_img, **kwargs)
        centre_gt_img, translated_gt_images = image_translation(gt_img, **kwargs)

        together_centre_img = np.concatenate([centre_cond_img, centre_gt_img], axis=-1)
        results['images'].append(together_centre_img)
        results['features'].append(feature)
        results['labels'].append(label)
        
        if translate and translated_cond_images is not None and split == 'train':
            translated_together_imgs = [np.concatenate([cond_img, gt_img], axis=-1) for cond_img, gt_img in zip(translated_cond_images, translated_gt_images)]
            results['images'].extend(translated_together_imgs)
            results['features'].extend([feature] * len(translated_together_imgs))
            results['labels'].extend([label] * len(translated_together_imgs))
            
        if rotate and split == 'train':
            rotated_cond_images = image_rotation(cond_img, **kwargs)
            rotated_gt_images = image_rotation(gt_img, **kwargs)
            rotated_together_imgs = [np.concatenate([cond_img, gt_img], axis=-1) for cond_img, gt_img in zip(rotated_cond_images, rotated_gt_images)]
            results['images'].extend(rotated_together_imgs)
            results['features'].extend([feature] * len(rotated_together_imgs))
            results['labels'].extend([label] * len(rotated_together_imgs))
    else:
        # Handle single images (Luna25)
        centre_img, translated_images = image_translation(image, **kwargs)
        results['images'].append(centre_img)
        results['features'].append(feature)
        results['labels'].append(label)

        if translate and translated_images is not None and split == 'train':
            results['images'].extend(translated_images)
            results['features'].extend([feature] * len(translated_images))
            results['labels'].extend([label] * len(translated_images))

        if rotate and split == 'train':
            rotated_images = image_rotation(image, **kwargs)
            results['images'].extend(rotated_images)
            results['features'].extend([feature] * len(rotated_images))
            results['labels'].extend([label] * len(rotated_images))
    
    return results

def _process_bandpass_filter(args):
    """
    Worker function for multiprocessing band-pass filtering.
    """
    image, kwargs = args
    h, w = image.shape[-2], image.shape[-1]
    is_paired = (w == 2 * h)
    
    if is_paired:
        # Handle paired images
        cond_img, gt_img = image[:, :, :w//2], image[:, :, w//2:]
        band_pass_cond_images, cond_diff = band_pass_filter(cond_img, **kwargs)
        band_pass_gt_images, gt_diff = band_pass_filter(gt_img, **kwargs)
        band_pass_together_img = np.concatenate([band_pass_cond_images[0], band_pass_gt_images[0]], axis=-1)
        return band_pass_together_img, gt_diff
    else:
        # Handle single images (Luna25)
        band_pass_images, diff = band_pass_filter(image, **kwargs)
        return band_pass_images[0], diff

def get_all_subjects(split='test', type='image', path: str = "NLST_with_second_2023"):
    folder_path = f"{path}/{type}/{split}"
    subjects = []

    for filename in os.listdir(folder_path):
        subjects.append(filename[:-4])
    return subjects


def _load_single_luna25(args):
    """Worker function for loading a single Luna25 image."""
    filename, image_folder, label_folder, num_channels = args
    # Load image
    image_path = os.path.join(image_folder, filename)
    image = np.load(image_path)  # Shape: (128, 128)
    
    # Load corresponding label
    label_path = os.path.join(label_folder, filename)
    label = np.load(label_path)  # Shape: (1,)
    
    # Expand dimensions to add channel dimension and match dataset format
    # Shape: (128, 128) -> (1, 128, 128) -> (C, 128, 128)
    image = np.expand_dims(image, axis=0)
    image = np.repeat(image, num_channels, axis=0)  # Match number of channels
    
    # Normalize to [-1, 1] range (Luna25 images are already in [0, 1])
    image = 2 * image - 1
    
    return image, label[0]


def _load_single_dlcs(args):
    """Worker function for loading a single DLCS image."""
    filename, image_folder, metadata_folder, num_channels = args
    # Load image
    image_path = os.path.join(image_folder, filename)
    image = np.load(image_path)  # Shape: (128, 128)
    
    # Load corresponding metadata
    metadata_filename = filename.replace('.npy', '_metadata.npy')
    metadata_path = os.path.join(metadata_folder, metadata_filename)
    metadata = np.load(metadata_path, allow_pickle=True).item()
    label = metadata['malignancy']  # 0=benign, 1=malignant
    
    # Expand dimensions to add channel dimension and match dataset format
    # Shape: (128, 128) -> (1, 128, 128) -> (C, 128, 128)
    image = np.expand_dims(image, axis=0)
    image = np.repeat(image, num_channels, axis=0)  # Match number of channels
    
    # Normalize from HU range to [-1, 1]
    # Typical HU range for lung CT: -1000 to 400
    image = np.clip(image, -1000, 400)
    image = (image + 1000) / 1400  # Scale to [0, 1]
    image = 2 * image - 1  # Scale to [-1, 1]
    
    return image, label


def _load_single_luna16(args):
    """Worker function for loading a single Luna16 image. Returns None if label is unknown."""
    filename, image_folder, diameter_folder, num_channels = args
    # Load corresponding diameter/label info first to check if we should skip
    diameter_path = os.path.join(diameter_folder, filename)
    diameter_info = np.load(diameter_path, allow_pickle=True)  # [diameter_mm, label]
    raw_label = diameter_info[1]
    
    # Skip samples with unknown label (label == 0)
    if raw_label == 0:
        return None
    
    # Convert label: 1=benign->0, >1=malignant->1
    label = 0 if raw_label == 1 else 1
    
    # Load image
    image_path = os.path.join(image_folder, filename)
    image = np.load(image_path)  # Shape: (128, 128)
    
    # Expand dimensions to add channel dimension and match dataset format
    # Shape: (128, 128) -> (1, 128, 128) -> (C, 128, 128)
    image = np.expand_dims(image, axis=0)
    image = np.repeat(image, num_channels, axis=0)  # Match number of channels
    
    # Normalize to [-1, 1] range (Luna16 images are already in [0, 1])
    image = 2 * image - 1
    
    return image, label


class Lung_Pair_DS(Dataset):
    def __init__(self, split='test', ch=1, base_path: str = "NLST_with_second_2023", augment: bool = False, retriever=None, selected_features: list = [], excluded_features: list = [], model_type="autoencoder", has_Luna25: bool = False, has_DLCS: bool = False, has_Luna16: bool = False, precomputed_embeddings_dir: str = None, construct_prompt: bool = False, **augment_configs):
        self.split = split
        self.selected_features = selected_features
        self.excluded_features = excluded_features
        self.model_type = model_type
        self.precomputed_embeddings_dir = precomputed_embeddings_dir
        self.retriever = retriever
        self.construct_prompt = model_type == "conditional" and construct_prompt

        self.subject_IDs = get_all_subjects(split, path=base_path)
        self.images = []
        self.features = []

        # Load precomputed embeddings if available
        self.precomputed_embeddings = None
        self.embedding_metadata = None
        if precomputed_embeddings_dir is not None:
            self._load_precomputed_embeddings(split, precomputed_embeddings_dir)

        for subject_ID in self.subject_IDs:
            path = os.path.join(base_path, "demo/", split, f"{subject_ID}.npy")

            try:
                curr_features = Demographic_Features(
                    path,
                    construct_prompt=self.construct_prompt,
                    retriever=self.retriever,
                )
            except Exception as e:
                print(f"Error loading features for {subject_ID}: {e}")
                continue
            if curr_features.first_year.has_nan() or curr_features.second_year.has_nan():
                print("Has NaN features")
                continue
            self.features.append(curr_features)

            image = Image.open(os.path.join(base_path, "image/", split, f"{subject_ID}.jpg"))

            image = np.expand_dims(np.array(image), axis=0)
            image = np.repeat(image, ch, axis=0)

            self.images.append(normalize_image(image))

        self.images = np.stack(self.images, axis=0)
        self.labels = np.array([feature.label for feature in self.features])

        if model_type == "autoencoder" or model_type == "unconditional":
            cond_imgs, gt_imgs = self.images[:, :, :, :self.images.shape[-1]//2], self.images[:, :, :, self.images.shape[-1]//2:]
            cond_features = np.array([feature.first_year.get_features(self.selected_features, self.excluded_features) for feature in self.features])
            gt_features = np.array([feature.second_year.get_features(self.selected_features, self.excluded_features) for feature in self.features])

            self.labels = np.array([feature.label for feature in self.features] * 2)

            self.images = np.concatenate([cond_imgs, gt_imgs], axis=0)
            self.features = np.concatenate([cond_features, gt_features], axis=0)
            if has_Luna25 and self.split == 'train':
                self.add_Luna25()
            if has_DLCS and self.split == 'train':
                self.add_DLCS()
            if has_Luna16 and self.split == 'train':
                self.add_Luna16()

        if augment:
            self.augment_images(**augment_configs)
            print("Dataset size (after augmentation): ", len(self.images))


    def add_Luna25(self):
        """
        Add Luna25 nodule dataset images to the existing dataset.
        Luna25 images are 128x128 grayscale images stored as .npy files.
        """
        luna25_path = "Luna25_nodule_2D_Checked"
        image_folder = os.path.join(luna25_path, "image")
        label_folder = os.path.join(luna25_path, "label")
        
        if not os.path.exists(image_folder):
            print(f"Warning: Luna25 dataset not found at {luna25_path}")
            return
        
        # Get all .npy files from the image folder
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.npy')])
        
        print(f"Loading {len(image_files)} Luna25 nodule images...")
        
        num_channels = self.images.shape[1]
        
        # Prepare args for worker function
        args_list = [(f, image_folder, label_folder, num_channels) for f in image_files]
        
        num_workers = min(cpu_count(), len(image_files))
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(_load_single_luna25, args_list), 
                               total=len(image_files), desc="Loading Luna25 images"))
        
        luna25_images = [r[0] for r in results]
        luna25_labels = [r[1] for r in results]

        luna25_images = np.stack(luna25_images, axis=0)
        luna25_labels = np.array(luna25_labels)
        
        # Create NaN feature vectors for Luna25 images (same shape as existing features)
        # NaN indicates missing demographic features for Luna25 nodule images
        if len(self.features) > 0:
            feature_dim = self.features.shape[1]
            luna25_features = np.full((len(luna25_images), feature_dim), np.nan)
        else:
            luna25_features = np.array([])

        self.images = np.concatenate([self.images, luna25_images], axis=0)
        self.labels = np.concatenate([self.labels, luna25_labels], axis=0)
        
        if len(luna25_features) > 0:
            self.features = np.concatenate([self.features, luna25_features], axis=0)
        
        print(f"Successfully added {len(luna25_images)} Luna25 images to dataset")
        print(f"New dataset size: {len(self.images)} images")

    def add_DLCS(self):
        """
        Add DLCS nodule dataset images to the existing dataset.
        DLCS images are 128x128 grayscale images stored as .npy files.
        Metadata contains diameter_mm and malignancy (0=benign, 1=malignant).
        """
        dlcs_path = "DLCS_patches"
        image_folder = os.path.join(dlcs_path, "npy")
        metadata_folder = os.path.join(dlcs_path, "metadata")
        
        if not os.path.exists(image_folder):
            print(f"Warning: DLCS dataset not found at {dlcs_path}")
            return
        
        # Get all .npy files from the image folder
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.npy')])
        
        print(f"Loading {len(image_files)} DLCS nodule images...")
        
        num_channels = self.images.shape[1]
        
        # Prepare args for worker function
        args_list = [(f, image_folder, metadata_folder, num_channels) for f in image_files]
        
        num_workers = min(cpu_count(), len(image_files))
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(_load_single_dlcs, args_list), 
                               total=len(image_files), desc="Loading DLCS images"))
        
        dlcs_images = [r[0] for r in results]
        dlcs_labels = [r[1] for r in results]

        dlcs_images = np.stack(dlcs_images, axis=0)
        dlcs_labels = np.array(dlcs_labels)
        
        # Create NaN feature vectors for DLCS images (same shape as existing features)
        # NaN indicates missing demographic features for DLCS nodule images
        if len(self.features) > 0:
            feature_dim = self.features.shape[1]
            dlcs_features = np.full((len(dlcs_images), feature_dim), np.nan)
        else:
            dlcs_features = np.array([])

        self.images = np.concatenate([self.images, dlcs_images], axis=0)
        self.labels = np.concatenate([self.labels, dlcs_labels], axis=0)
        
        if len(dlcs_features) > 0:
            self.features = np.concatenate([self.features, dlcs_features], axis=0)
        
        print(f"Successfully added {len(dlcs_images)} DLCS images to dataset")
        print(f"New dataset size: {len(self.images)} images")

    def add_Luna16(self):
        """
        Add Luna16 nodule dataset images to the existing dataset.
        Luna16 images are 128x128 grayscale images stored as .npy files.
        Diameter files contain [diameter_mm, label] where:
        - label 0: unknown (excluded)
        - label 1: benign (0)
        - label >1: malignant (1)
        """
        luna16_path = "Luna16_patches"
        image_folder = os.path.join(luna16_path, "npy")
        diameter_folder = os.path.join(luna16_path, "diameter")
        
        if not os.path.exists(image_folder):
            print(f"Warning: Luna16 dataset not found at {luna16_path}")
            return
        
        # Get all .npy files from the image folder
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.npy')])
        
        print(f"Loading {len(image_files)} Luna16 nodule images...")
        
        num_channels = self.images.shape[1]
        
        # Prepare args for worker function
        args_list = [(f, image_folder, diameter_folder, num_channels) for f in image_files]
        
        num_workers = min(cpu_count(), len(image_files))
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(_load_single_luna16, args_list), 
                               total=len(image_files), desc="Loading Luna16 images"))
        
        # Filter out None results (samples with unknown labels)
        results = [r for r in results if r is not None]
        
        if len(results) == 0:
            print("Warning: No Luna16 images with known labels found")
            return
        
        luna16_images = [r[0] for r in results]
        luna16_labels = [r[1] for r in results]

        luna16_images = np.stack(luna16_images, axis=0)
        luna16_labels = np.array(luna16_labels)
        
        # Create NaN feature vectors for Luna16 images (same shape as existing features)
        # NaN indicates missing demographic features for Luna16 nodule images
        if len(self.features) > 0:
            feature_dim = self.features.shape[1]
            luna16_features = np.full((len(luna16_images), feature_dim), np.nan)
        else:
            luna16_features = np.array([])

        self.images = np.concatenate([self.images, luna16_images], axis=0)
        self.labels = np.concatenate([self.labels, luna16_labels], axis=0)
        
        if len(luna16_features) > 0:
            self.features = np.concatenate([self.features, luna16_features], axis=0)
        
        print(f"Successfully added {len(luna16_images)} Luna16 images to dataset (excluded {len(image_files) - len(luna16_images)} with unknown labels)")
        print(f"New dataset size: {len(self.images)} images")


    def augment_images(
        self, 
        translate: bool = False,
        rotate: bool = False,
        band_pass: bool = False,
        num_workers: int = None,
        **kwargs,
    ):
        """
        Augment images with optional multiprocessing support.
        
        Args:
            translate: Whether to apply translation augmentation
            rotate: Whether to apply rotation augmentation
            band_pass: Whether to apply band-pass filtering
            num_workers: Number of workers for multiprocessing. If None, uses cpu_count(). Set to 1 to disable multiprocessing.
            **kwargs: Additional arguments for augmentation functions
        """
        assert len(self.images) == len(self.features)

        if not hasattr(self, 'labels'):
            self.labels = np.zeros(len(self.features)) # dummy for conditional training
        
        # Determine number of workers
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU free
        # Prepare arguments for parallel processing
        args_list = [
            (image, feature, label, self.split, translate, rotate, kwargs)
            for image, feature, label in zip(self.images, self.features, self.labels)
        ]
        
        # Process images with multiprocessing
        if num_workers > 1 and len(args_list) > 1:
            print(f"Augmenting images with {num_workers} workers...")
            with Pool(processes=num_workers) as pool:
                # Using larger chunksize to reduce overhead
                chunksize = max(10, len(args_list) // num_workers)
                results = list(tqdm(
                    pool.imap(_process_single_image_augmentation, args_list, chunksize=chunksize),
                    total=len(args_list),
                    desc="Augmenting images"
                ))
        else:
            print("Augmenting images (single process)...")
            results = [_process_single_image_augmentation(args) for args in tqdm(args_list, desc="Augmenting images")]
        
        # Collect results
        new_images = []
        features = []
        labels = []
        
        for result in results:
            new_images.extend(result['images'])
            features.extend(result['features'])
            labels.extend(result['labels'])
        self.diffs = []
        if band_pass:
            # Prepare arguments for band-pass filtering
            bandpass_args = [(image, kwargs) for image in new_images]
            
            # Process band-pass filtering with multiprocessing
            if num_workers > 1 and len(bandpass_args) > 1:
                print(f"Applying band-pass filter with {num_workers} workers...")
                with Pool(processes=num_workers) as pool:
                    # Using larger chunksize to reduce overhead
                    chunksize = max(10, len(bandpass_args) // num_workers)
                    bandpass_results = list(tqdm(
                        pool.imap(_process_bandpass_filter, bandpass_args, chunksize=chunksize),
                        total=len(bandpass_args),
                        desc="Band-pass filtering"
                    ))
            else:
                print("Applying band-pass filter (single process)...")
                bandpass_results = [_process_bandpass_filter(args) for args in tqdm(bandpass_args, desc="Band-pass filtering")]
            
            # Update images and collect diffs
            for i, (filtered_img, diff) in enumerate(bandpass_results):
                new_images[i] = filtered_img
                self.diffs.append(diff)

        self.images = np.stack(new_images, axis=0)
        self.features = features
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.model_type == "autoencoder" or self.model_type == "unconditional":
            image = torch.tensor(self.images[idx])
            feature = torch.tensor(self.features[idx])
            label = torch.tensor(self.labels[idx])
            return image, feature, label

        c, h, w = self.images[idx].shape
        cond_img, gt_img = self.images[idx][:, :, :w//2], self.images[idx][:, :, w//2:]

        feature = self.features[idx]
        if self.construct_prompt:
            return (
                torch.tensor(cond_img),
                torch.tensor(gt_img),
                torch.tensor(feature.sentence_encoded),
                torch.tensor(feature.label),
            )

        first_year_features = feature.first_year.get_features(self.selected_features, self.excluded_features)
        second_year_features = feature.second_year.get_features(self.selected_features, self.excluded_features)

        # if len(self.diffs) > 0:
        #     diff = torch.tensor(self.diffs[idx])
        # else:
        #     diff = torch.tensor(0)

        return (torch.tensor(cond_img), torch.tensor(first_year_features)), (torch.tensor(gt_img), torch.tensor(second_year_features)), torch.tensor(feature.label)

class FeatureGroupedBatchSampler(torch.utils.data.Sampler):
    """
    Custom batch sampler that groups samples by feature availability.
    Each batch will contain either almost all samples with valid (non-NaN) features,
    or all samples with NaN features.

    Args:
        features: Array of features where NaN indicates missing features
        batch_size: Total batch size
        shuffle: Whether to shuffle indices within each group
        drop_last: Whether to drop the last incomplete batch
        mix_ratio: Ratio of batches to allocate to valid-feature samples (default: proportional to dataset)
    """
    def __init__(self, features, batch_size=32, shuffle=True, drop_last=False, mix_ratio=None):
        self.features = np.array(features)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Identify samples with valid features vs NaN features
        # A sample has NaN features if any feature value is NaN
        has_nan = np.array([np.isnan(f).any() if hasattr(f, '__len__') else np.isnan(f) for f in self.features])

        self.valid_indices = np.where(~has_nan)[0]
        self.nan_indices = np.where(has_nan)[0]

        print(f"FeatureGroupedBatchSampler: {len(self.valid_indices)} samples with valid features, {len(self.nan_indices)} samples with NaN features")

        # Calculate number of batches for each group
        self.n_valid_batches = len(self.valid_indices) // batch_size
        self.n_nan_batches = len(self.nan_indices) // batch_size

        if not drop_last:
            if len(self.valid_indices) % batch_size > 0:
                self.n_valid_batches += 1
            if len(self.nan_indices) % batch_size > 0:
                self.n_nan_batches += 1

        self.n_batches = self.n_valid_batches + self.n_nan_batches

        print(f"Batch distribution: {self.n_valid_batches} valid-feature batches + {self.n_nan_batches} NaN-feature batches = {self.n_batches} total")

    def __iter__(self):
        # Shuffle indices within each group if required
        if self.shuffle:
            valid_perm = np.random.permutation(self.valid_indices)
            nan_perm = np.random.permutation(self.nan_indices)
        else:
            valid_perm = self.valid_indices.copy()
            nan_perm = self.nan_indices.copy()

        # Create all batches
        all_batches = []

        # Create batches from valid-feature samples
        for i in range(self.n_valid_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(valid_perm))
            if self.drop_last and end_idx - start_idx < self.batch_size:
                continue
            batch = valid_perm[start_idx:end_idx]
            all_batches.append(batch.tolist())

        # Create batches from NaN-feature samples
        for i in range(self.n_nan_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(nan_perm))
            if self.drop_last and end_idx - start_idx < self.batch_size:
                continue
            batch = nan_perm[start_idx:end_idx]
            all_batches.append(batch.tolist())

        # Shuffle the order of batches (not within batches)
        if self.shuffle:
            np.random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        return self.n_batches


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Custom batch sampler that ensures each batch contains both benign and malignant samples.

    Args:
        labels: Array/tensor of binary labels (0=benign, 1=malignant)
        batch_size: Total batch size
        benign_ratio: Ratio of benign samples in each batch (default 0.5)
        shuffle: Whether to shuffle indices within each class
        drop_last: Whether to drop the last incomplete batch
    """
    def __init__(self, labels, batch_size=32, benign_ratio=0.5, shuffle=True, drop_last=False):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.benign_ratio = benign_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Calculate number of samples per class per batch
        self.n_benign_per_batch = int(batch_size * benign_ratio)
        self.n_malignant_per_batch = batch_size - self.n_benign_per_batch
        
        # Get indices for each class

        self.benign_indices = np.where(self.labels == 0)[0]
        self.malignant_indices = np.where(self.labels == 1)[0]
        
        print(f"BalancedBatchSampler: {len(self.benign_indices)} benign, {len(self.malignant_indices)} malignant")
        print(f"Batch composition: {self.n_benign_per_batch} benign + {self.n_malignant_per_batch} malignant = {batch_size}")
        
        # Calculate number of batches
        self.n_benign_batches = len(self.benign_indices) // self.n_benign_per_batch
        self.n_malignant_batches = len(self.malignant_indices) // self.n_malignant_per_batch
        self.n_batches = min(self.n_benign_batches, self.n_malignant_batches)
        
        if self.n_batches == 0:
            raise ValueError(
                f"Not enough samples to create balanced batches. "
                f"Need at least {self.n_benign_per_batch} benign and {self.n_malignant_per_batch} malignant samples, "
                f"but got {len(self.benign_indices)} benign and {len(self.malignant_indices)} malignant."
            )
    
    def __iter__(self):
        # Shuffle indices if required
        if self.shuffle:
            benign_perm = np.random.permutation(self.benign_indices)
            malignant_perm = np.random.permutation(self.malignant_indices)
        else:
            benign_perm = self.benign_indices
            malignant_perm = self.malignant_indices
        
        # Create batches
        for i in range(self.n_batches):
            # Get indices for this batch
            benign_batch = benign_perm[i * self.n_benign_per_batch:(i + 1) * self.n_benign_per_batch]
            malignant_batch = malignant_perm[i * self.n_malignant_per_batch:(i + 1) * self.n_malignant_per_batch]
            
            # Combine and shuffle within batch
            batch_indices = np.concatenate([benign_batch, malignant_batch])
            if self.shuffle:
                np.random.shuffle(batch_indices)
            
            yield batch_indices.tolist()
    
    def __len__(self):
        return self.n_batches


class Lung_DM(pl.LightningDataModule):
    def __init__(
        self,
        dataset_class: Dataset,  # Pass the actual class object, e.g., Lung_Pair_DS
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        use_balanced_sampler: bool = False,
        use_feature_grouped_sampler: bool = False,
        benign_ratio: float = 0.5,
        **input_kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset_class'] + list(input_kwargs.keys()))
        self.dataset_cls = dataset_class
        self.input_kwargs = input_kwargs
        self.use_balanced_sampler = use_balanced_sampler
        self.use_feature_grouped_sampler = use_feature_grouped_sampler
        self.benign_ratio = benign_ratio

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.retriever = None

        self.train_sampler: Optional[BalancedBatchSampler] = None
        self.val_sampler: Optional[BalancedBatchSampler] = None

    def set_retriever(self, retriever):
        self.retriever = retriever

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_cls(split='train', retriever=self.retriever, num_workers=self.hparams.num_workers, **self.input_kwargs)
            self.val_dataset = self.dataset_cls(split='val', retriever=self.retriever, num_workers=self.hparams.num_workers, **self.input_kwargs)

            # Create feature-grouped samplers if requested (takes priority over balanced sampler)
            if self.use_feature_grouped_sampler:
                self.train_sampler = FeatureGroupedBatchSampler(
                    features=self.train_dataset.features,
                    batch_size=self.hparams.batch_size,
                    shuffle=self.hparams.shuffle,
                    drop_last=True
                )
                self.val_sampler = FeatureGroupedBatchSampler(
                    features=self.val_dataset.features,
                    batch_size=self.hparams.batch_size,
                    shuffle=False,
                    drop_last=False
                )
            # Create balanced samplers if requested
            elif self.use_balanced_sampler:
                self.train_sampler = BalancedBatchSampler(
                    labels=self.train_dataset.labels,
                    batch_size=self.hparams.batch_size,
                    benign_ratio=self.benign_ratio,
                    shuffle=self.hparams.shuffle,
                    drop_last=True
                )
                self.val_sampler = BalancedBatchSampler(
                    labels=self.val_dataset.labels,
                    batch_size=self.hparams.batch_size,
                    benign_ratio=self.benign_ratio,
                    shuffle=False,
                    drop_last=False
                )

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_cls(split='test', retriever=self.retriever, num_workers=self.hparams.num_workers, **self.input_kwargs)

    def train_dataloader(self):
        if self.use_feature_grouped_sampler or self.use_balanced_sampler:
            return DataLoader(
                self.train_dataset,
                batch_sampler=self.train_sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=self.hparams.shuffle,
            )

    def val_dataloader(self):
        if self.use_feature_grouped_sampler or self.use_balanced_sampler:
            return DataLoader(
                self.val_dataset,
                batch_sampler=self.val_sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

def Lung_Pair_DM(**kwargs):
    return Lung_DM(dataset_class=Lung_Pair_DS, **kwargs)

