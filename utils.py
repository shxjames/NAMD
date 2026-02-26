import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import importlib

def get_dataset_features(dataset, device='cuda', autoencoder=None,):
    assert autoencoder is not None, "autoencoder must be provided"
    autoencoder.eval()

    autoencoder.to(device)
    dataset_features = []
    image_set = []
    for (images, labels) in dataset:
        images = images.to(device)

        with torch.no_grad():
            posterior = autoencoder.encode(images.unsqueeze(0))
            z = posterior.mode()

            z = z.view(z.shape[0], -1)
            dataset_features.append(z.cpu().numpy())
        
    dataset_features = np.concatenate(dataset_features, axis=0)
    return dataset_features

def get_k_nearest_neighbor(image, k=5, device='cuda', dataset=None, dataset_features=None, autoencoder=None):
    if dataset_features is None:
        assert dataset is not None, "Either dataset_features or dataset must be provided"
        dataset_features = get_dataset_features(dataset, device, autoencoder)

    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(dataset_features)

    posterior = autoencoder.encode(image)
    z = posterior.mode()

    if z.dim() == 3:
        z = z[None, :, :, :]

    z = z.view(z.shape[0], -1)
    z = z.cpu().numpy()
    distance, index = nn.kneighbors(z)

    nearest_neighbors = dataset[index][0]

    return nearest_neighbors

def image_translation(
    image: np.ndarray, 
    dims: int = 64, 
    translation: int = -1,
    direction: list = ['down', 'up', 'left', 'right'],
    **kwargs
):
    assert image.shape[-1] >= dims and image.shape[-2] >= dims, "image must be of shape (c, h, w) where w >= dims and h >= dims"

    h, w = image.shape[-2], image.shape[-1]

    h_offset = (h - dims) // 2
    w_offset = (w - dims) // 2

    centre_dims = ((h_offset, w_offset), (h_offset + dims, w_offset + dims))

    centre_img = image[:, centre_dims[0][0]:centre_dims[1][0], centre_dims[0][1]:centre_dims[1][1]]

    if translation == -1:
        return centre_img, None
    else:
        translated_images = []
        if 'down' in direction:
            down = image[:, centre_dims[0][0] + translation:centre_dims[1][0] + translation, centre_dims[0][1]:centre_dims[1][1]]
            translated_images.append(down)
        if 'up' in direction:
            up = image[:, centre_dims[0][0] - translation:centre_dims[1][0] - translation, centre_dims[0][1]:centre_dims[1][1]]
            translated_images.append(up)
        if 'left' in direction:
            left = image[:, centre_dims[0][0]:centre_dims[1][0], centre_dims[0][1] - translation:centre_dims[1][1] - translation]
            translated_images.append(left)
        if 'right' in direction:    
            right = image[:, centre_dims[0][0]:centre_dims[1][0], centre_dims[0][1] + translation:centre_dims[1][1] + translation]
            translated_images.append(right)

        return centre_img, translated_images

def image_rotation(
    image: np.ndarray, 
    dims: int = 64, 
    angles: list = [90, 180, 270],
    **kwargs
):
    from scipy.ndimage import rotate
    
    assert image.shape[-1] >= dims and image.shape[-2] >= dims, "image must be of shape (c, h, w) where w >= dims and h >= dims"

    h, w = image.shape[-2], image.shape[-1]
    
    h_offset = (h - dims) // 2
    w_offset = (w - dims) // 2
    
    centre_dims = ((h_offset, w_offset), (h_offset + dims, w_offset + dims))
    
    rotated_images = []
    
    for angle in angles:
        # Rotate each channel separately
        rotated_channels = []
        
        for c in range(image.shape[0]):
            rotated_channel = rotate(image[c], angle, reshape=False, order=1, mode='constant', cval=0)
            rotated_channels.append(rotated_channel)
        
        # Stack channels back
        rotated_img = np.stack(rotated_channels, axis=0)
        
        # Crop center region from rotated image
        rotated_centre = rotated_img[:, centre_dims[0][0]:centre_dims[1][0], centre_dims[0][1]:centre_dims[1][1]]
        rotated_images.append(rotated_centre)
    
    return rotated_images

def butter_bandpass_filter(data, low, high, order):
    from scipy.signal import butter, filtfilt
    
    # Design the filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter
    filtered_data = filtfilt(b, a, data, axis=-1)  # or appropriate axis
    
    return filtered_data

def band_pass_filter(
    image,
    low=0.01,
    high=0.25, 
    order=2,
    normalize=True,
    dims=64,
    **kwargs,
):
    original_image = image.copy()
    # Normalize image to 0-1
    image = (image + 1) / 2

    # Apply band-pass filter
    filtered_image = butter_bandpass_filter(image, low, high, order)


    filtered_image = np.clip(filtered_image, 0, 1)
        
    # Option 2: Normalize to [0,1] range (uncomment if you prefer this)
    # filtered_image = (filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min())
    
    # Normalize back to -1 to 1
    filtered_image = (filtered_image - 0.5) * 2

    # Get the centre of the image
    h, w = image.shape[-2], image.shape[-1]
    
    h_offset = (h - dims) // 2
    w_offset = (w - dims) // 2
    
    centre_dims = ((h_offset, w_offset), (h_offset + dims, w_offset + dims))

    filtered_image = filtered_image[:, centre_dims[0][0]:centre_dims[1][0], centre_dims[0][1]:centre_dims[1][1]]

    return [filtered_image], original_image - filtered_image
    

def visualize_image(image: np.ndarray, augmented_image: np.ndarray, save_path: str = "augmented_image.png"):
    h, w = image.shape[-2], image.shape[-1]
    h_a, w_a = augmented_image.shape[-2], augmented_image.shape[-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    image = image.transpose(1, 2, 0)
    augmented_image = augmented_image.transpose(1, 2, 0)

    image = (image + 1) / 2
    augmented_image = (augmented_image + 1) / 2
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(augmented_image)
    axes[1].set_title("Augmented Image")
    axes[1].axis("off")

    plt.savefig(save_path)




