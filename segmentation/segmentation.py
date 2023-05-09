#!/usr/bin/env python

# segmentation.py
# Add your code here


import numpy as np
import matplotlib.pyplot as plt

from skimage import io, morphology
from skimage.morphology import disk
import skimage.filters
import skimage.measure
from skimage.measure import regionprops_table


def crop_image(image, crop_size, mode='random', overlap=0):
    height, width = image.shape
    crop_height, crop_width = crop_size

    if mode == 'random':
        start_y = np.random.randint(0, height - crop_height + 1)
        start_x = np.random.randint(0, width - crop_width + 1)

        cropped_image = image[start_y : start_y + crop_height, start_x : start_x + crop_width]

    elif mode == 'grid':
        grid_height = (height - crop_height) // (crop_height - overlap) + 1
        grid_width = (width - crop_width) // (crop_width - overlap) + 1

        cropped_images = []

        for i in range(grid_height):
            for j in range(grid_width):
                start_y = i * (crop_height - overlap)
                start_x = j * (crop_width - overlap)

                cropped_image = image[start_y : start_y + crop_height, start_x : start_x + crop_width]
                cropped_images.append(cropped_image)

        return cropped_images

    else:
        raise ValueError("Invalid mode specified. Choose either 'random' or 'grid'.")

    return cropped_image


def top_hat_transform(image, selem_size=5):
    """
    Apply a top-hat transform filtering operation to an image.

    Parameters:
    -----------
    image : ndarray
        Input image.
    selem_size : int, optional, default: 5
        Size of the structuring element used for the top-hat transform.

    Returns:
    --------
    result : ndarray
        Image after applying the top-hat transform.
    """
    selem = disk(selem_size)
    result = morphology.white_tophat(image, selem)
    return result


def segment_image(image, method='otsu', custom_threshold=None):
    if method == 'otsu':
        threshold = skimage.filters.threshold_otsu(image)
    elif method == 'mean':
        threshold = np.mean(image)
    elif method == 'custom':
        if custom_threshold is not None:
            threshold = custom_threshold
        else:
            raise ValueError("Please provide a custom threshold value.")
    else:
        raise ValueError("Invalid method specified. Choose either 'otsu', 'mean', or 'custom'.")

    segmented_image = image > threshold
    labeled_image = skimage.measure.label(segmented_image)
    properties = regionprops_table(labeled_image, properties=('centroid',))
    positions = np.column_stack([properties['centroid-0'], properties['centroid-1']])

    return segmented_image, positions


def compare_segmentation(image, segmented_image, positions):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(segmented_image, cmap='gray')
    axes[1].set_title('Segmented Image')

    axes[2].imshow(image, cmap='gray')
    axes[2].scatter(positions[:, 1], positions[:, 0], c='red', s=1, marker='o', alpha=0.5)
    axes[2].set_title('Object Positions')

    plt.show()
