#!/usr/bin/env python

# segmentation.py

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

                # Check if we are at the last cell of the grid in y or x direction
                # If so, ensure we don't exceed the image boundaries
                if i == grid_height - 1:
                    end_y = height
                else:
                    end_y = start_y + crop_height

                if j == grid_width - 1:
                    end_x = width
                else:
                    end_x = start_x + crop_width

                cropped_image = image[start_y:end_y, start_x:end_x]
                cropped_images.append(cropped_image)

        # Test to check that the sum of areas of all crops is equal to the area of the original image
        assert np.sum([crop.size for crop in cropped_images]) == image.size

        return cropped_images

    else:
        raise ValueError("Invalid mode specified. Choose either 'random' or 'grid'.")



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
    
    '''
    Old version that combined thresholding and labeling.
    '''
    print('Warning, segment_image() is deprecated. Use threshold_image() and label_image() instead.')

    # Threshold the image
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

def threshold_image(image, method='otsu', custom_threshold=None):
        
        # Threshold the image
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
    
        thresholded_image = image > threshold

        # make sure the image is binary
        thresholded_image = thresholded_image.astype(int)

        return thresholded_image


def label_image(thresholded_image):   

    # Assert the image is binary (already thresholded)
    assert np.all(np.unique(thresholded_image) == [0, 1]), "Image is not binary." 
    
    # Apply regionprops to extract the positions
    labeled_image = skimage.measure.label(thresholded_image)
    properties = regionprops_table(labeled_image, properties=('centroid',))
    positions = np.column_stack([properties['centroid-0'], properties['centroid-1']])
    
    return labeled_image, positions

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


'''
Some background subtraction methods:
'''


def slide_background_subtraction(image, atlas, method='mean'):
    """
    Subtracts the slide background from an image.

    Parameters:
    -----------
    image : ndarray
        Input image.
    atlas : ndarray
        Atlas labels.
    method : str, optional, default: 'mean'
        Method to calculate the central tendency of the pixel values.
        Options are 'mean', 'median', 'mode'.

    Returns:
    --------
    result : ndarray
        Image after subtracting the slide background.
    """
    # Get the pixel values outside the tissue (where the atlas label is 0)
    background_pixels = image[atlas == 0]
    
    # Calculate the central tendency of the background pixels
    if method == 'mean':
        background_value = np.mean(background_pixels)
    elif method == 'median':
        background_value = np.median(background_pixels)
    elif method == 'mode':
        background_value = stats.mode(background_pixels)[0][0]
    else:
        raise ValueError("Invalid method specified. Choose either 'mean', 'median', or 'mode'.")

    # Subtract the background from the image
    result = image - background_value
    result = np.clip(result, 0, None)  # Ensure that the pixel values stay non-negative

    return result


def tissue_background_subtraction(image, atlas, method='mean'):
    """
    Subtracts the tissue background from an image.

    Parameters:
    -----------
    image : ndarray
        Input image.
    atlas : ndarray
        Atlas labels.
    method : str, optional, default: 'mean'
        Method to calculate the central tendency of the pixel values.
        Options are 'mean', 'median', 'mode'.

    Returns:
    --------
    result : ndarray
        Image after subtracting the tissue background.
    """
    # Get the pixel values inside the tissue (where the atlas label is not 0)
    tissue_pixels = image[atlas != 0]

    # Calculate the central tendency of the tissue pixels
    if method == 'mean':
        tissue_value = np.mean(tissue_pixels)
    elif method == 'median':
        tissue_value = np.median(tissue_pixels)
    elif method == 'mode':
        tissue_value = stats.mode(tissue_pixels)[0][0]
     
    else:
        raise ValueError("Invalid method specified. Choose either 'mean', 'median', or 'mode'.")

    # Subtract the background from the image
    result = image - tissue_value
    result = np.clip(result, 0, None)  # Ensure that the pixel values stay non-negative

    return result

from scipy.stats import mode

def region_background_subtraction(image, atlas_slice, label_data, region_id, statistic='median'):
    print('region_background_subtraction() is deprecated. Use region_difference_mask() instead.')
    
    """Subtract region-specific background from image.

    Parameters:
    image: np.array
        Image from which to subtract background.
    atlas_slice: np.array
        Atlas slice corresponding to the image.
    label_data: dict
        Dictionary containing label data for the atlas.
    region_id: int
        ID of the region for which to subtract background.
    statistic: str, optional
        Statistic to use for calculating background ('mean', 'median', or 'mode').
        Default is 'median'.
        
    Returns:
    np.array: Background subtracted image.
    """
    # Get the mask for this region
    region_mask = (atlas_slice == region_id)

    # Apply the mask to the image
    masked_img = image * region_mask

    # Calculate the central tendency of the pixels within the mask
    if statistic == 'mean':
        central_tendency = np.mean(masked_img[region_mask])
    elif statistic == 'median':
        central_tendency = np.median(masked_img[region_mask])
    elif statistic == 'mode':
        central_tendency = mode(masked_img[region_mask])[0][0]
    else:
        raise ValueError("Invalid statistic: choose from 'mean', 'median', or 'mode'.")

    # print(f"{label_data[region_id]['label']} background value: {central_tendency}")

    # Subtract this value from the masked image
    result = masked_img - central_tendency

    # Ensure that the minimum pixel value is 0 (no negative values)
    result = np.clip(result, 0, None)

    # Replace the region in the original image with the background subtracted region
    image[region_mask] = result[region_mask]

    return image



def region_difference_mask(input_img, mask, atlas_slice, label_data, region_id, statistic='median'):
    # Get the mask for this region
    region_mask = (atlas_slice == region_id)

    # Apply the mask to the input image
    masked_img = input_img * region_mask

    # Calculate the central tendency of the pixels within the mask
    if statistic == 'mean':
        central_tendency = np.mean(masked_img[region_mask])
    elif statistic == 'median':
        central_tendency = np.median(masked_img[region_mask])
    elif statistic == 'mode':
        central_tendency = mode(masked_img[region_mask])[0][0]
    else:
        raise ValueError("Invalid statistic: choose from 'mean', 'median', or 'mode'.")

    # Update the mask
    mask[region_mask] = central_tendency

    return mask
def get_display_range(image, method='std_dev'):
    """
    Calculate the display range for an image.

    Parameters:
    image: np.array
        Image for which to calculate the display range.
    method: str
        Method for calculating the display range. Options are 'std_dev' (default),
        'mad' (median absolute deviation), 'percentile', and 'min_max'.

    Returns:
    tuple: (min_intensity, max_intensity)
    """

    if method == 'std_dev':
        mean = np.mean(image)
        std = np.std(image)
        min_intensity = max(0, mean - 3*std)  # clip at zero, as we can't have negative intensities
        max_intensity = mean + 3*std

    elif method == 'mad':
        median = np.median(image)
        mad = np.median(np.abs(image - median))
        min_intensity = max(0, median - 3*mad)
        max_intensity = median + 3*mad

    elif method == 'percentile':
        min_intensity = np.percentile(image, 1)  # 1st percentile
        max_intensity = np.percentile(image, 99)  # 99th percentile

    elif method == 'min_max':
        min_intensity = np.min(image)
        max_intensity = np.max(image)

    else:
        raise ValueError(f"Invalid method: {method}. Choose from 'std_dev', 'mad', 'percentile', or 'min_max'.")

    return min_intensity, max_intensity







def calculate_difference_mask(input_img,scaled_label_slice):
    #label_data must be defined

    difference_mask = np.zeros_like(input_img)

    for region_id in tqdm(list(np.unique(scaled_label_slice))):
        # Update the difference mask for each region
        difference_mask = region_difference_mask(
            input_img,
            difference_mask,
            scaled_label_slice,
            label_data,
            region_id=region_id,
            statistic='mean'
        )

    return difference_mask


def apply_difference_mask(input_img, difference_mask):
    '''Apply difference mask while avoiding an int overflow error'''

    # Convert to a larger data type before the operation
    input_img = input_img.astype(np.int32)
    difference_mask = difference_mask.astype(np.int32)

    # Subtract the difference mask from the original image
    processed_img = input_img - difference_mask

    # Clip the values back to the desired range
    processed_img = np.clip(processed_img, 0, 65535)

    # Convert back to the original data type if necessary
    processed_img = processed_img.astype(np.uint16)

    return processed_img