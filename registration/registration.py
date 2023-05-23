#!/usr/bin/env python

# registration.py

import os
import numpy as np
import imageio
from skimage import io, transform
import matplotlib.pyplot as plt
from tqdm import tqdm
import re




def load_atlas_volume(filename):

    print('Loading registered atla volume from: ' + filename)
          
    # Load the reference atlas in image coordinates
    labels = io.imread(filename).astype(np.int32) 

    # Reorder the labels to be more intuitive, where the z-axis is the 3rd dimension
    labels = np.moveaxis(labels, 0, -1)
    print('Atlas shape: ', labels.shape)

    # lab_xdim, lab_ydim, lab_zdim = labels.shape # Should we be using these somewhere??
    # labels.shape
    return labels

def map_img_to_label(img_identifier, identifier_list, label_volume):

    '''Take an image name (identifier), 
    and map this to the corresponding slice
      of the label volume'''
    
    # img_zdim = fs_zdim # This may need to be changed if we use downsampled images that are also downsampled in z.
    # The z dimension of the full-sized image is the number of images.
    img_zdim = len(identifier_list)#fs_zdim # This may need to be changed if we use downsampled images that are also downsampled in z.
    lab_zdim = label_volume.shape[2]

    # Get the index of the identifier in the list of identifiers
    idx = identifier_list.index(img_identifier)
 
    # Get the corresponding slice of the label volume by interpolating the index
    label_ind = int(idx * lab_zdim / img_zdim)
    label_slice = label_volume[:,:,label_ind] # Note that

    # Return a tuple containing the image index and the corresponding label index, along with that slice of the label volume
    return  (idx,label_ind), label_slice

def map_label_to_img(first_slice, last_slice, identifier_list, label_volume):
    '''Take a range of slices (first_slice to last_slice inclusive) in the label volume,
    and map these to the corresponding image identifiers'''

    # Get the total number of slices in the image set and label volume
    img_zdim = len(identifier_list)
    lab_zdim = label_volume.shape[2]

    # Interpolate to find the corresponding image indices
    first_img_ind = int(first_slice * img_zdim / lab_zdim)
    last_img_ind = int((last_slice+1) * img_zdim / lab_zdim)  # +1 to make the range inclusive

    # Err on the side of including more images by expanding the range
    first_img_ind = max(0, first_img_ind - 1)
    last_img_ind = min(img_zdim - 1, last_img_ind + 1)

    # Return the corresponding image identifiers
    return identifier_list[first_img_ind : last_img_ind + 1]  # +1 to make the range inclusive


# For a given int value of the label, return a mask of the same shape as the image
def get_mask_from_label(label, labels):
    mask = labels == label
    return mask 

# Scale up the mask to be compatible with both the downsampled image
def scale_mask(mask, img):
    '''Scale up the mask  to be compatible with both the downsampled image
    and the fullsized image
    Should work on either a 2d or 3d mask

    '''
    mask = transform.resize(mask, img.shape, order=0, preserve_range=True)

    return mask

def apply_mask_to_img(img, mask):
    '''Scale up the mask and apply mask to the image'''

    # First check if the provided mask is 2d or 3d, and extract slice if 3d
    if len(mask.shape) == 2:
        scaled_mask = scale_mask(mask, img)
    elif len(mask.shape) == 3:
        print('apply_mask_to_img() is not currently able to handle 3d masks input')
        scaled_mask = scale_mask(mask[:,:,lab_ind], img)    
    
    img = img * scaled_mask
    return img



def apply_region_mask(img, slice_identifier, identifiers, labels, region_id, plot=False): 
    
    print('apply_region_mask() should only be used if you dont have cle installed')
    '''Apply a mask to an image to only keep the voxels corresponding to a given region'''
    
    '''  Note: This functions seems a little overloaded, but I need to be able to pass
      the image as it undergoes upstream image processing steps. I also need the slice_identifier and list of identifiers
      to map the image to the label volume. Finally, I need the labels and region_id to get the mask.
    '''

    # Get the index of this image with respect to
    (img_ind, lab_ind), this_atlas_slice = map_img_to_label(slice_identifier, identifiers, labels)

    # Get the mask this region
    mask = get_mask_from_label(region_id, labels)
    
    # Get the corresponding slice of the mask for this image
    mask_slice = mask[:,:,lab_ind]

    # Use it for the mask we just created with the full-sized image
    masked_img = apply_mask_to_img(img, mask_slice)

    return masked_img



def parse_itk_snap_label_file(file_path):
    labels = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Skip lines that start with '#' or are empty
            if line.startswith("#") or line.strip() == "":
                continue

            # Split the line into fields
            fields = line.split()

            # Extract the index, RGB values, and label
            index = int(fields[0])
            rgb = (int(fields[1]), int(fields[2]), int(fields[3]))
            # Extract the label using a regular expression
            label = re.search(r'"(.+)"', line).group(1)

            # Store the extracted information in a dictionary
            labels[index] = {'rgb': rgb, 'label': label}


    return labels


'''Functions for Looking at only the slice that contains the region of interest from the atlas.'''

def get_slices_containing_region(volume, region_id):
    # Identify where in the volume the region_id is found
    indices = np.where(volume == region_id)

    # indices is a tuple of 3 1D arrays (for the 3 dimensions of the volume)
    # The third element of the tuple gives the indices in the third dimension (slices)
    slice_indices = indices[2]

    # Get unique slice indices, as there may be multiple voxels with region_id in a single slice
    unique_slices = np.unique(slice_indices)

    return unique_slices

def test_slices_contiguity(slices):
    # Calculate the differences between adjacent elements
    differences = np.diff(slices)
    
    # Check if all differences are 1 (which indicates contiguity)
    is_contiguous = np.all(differences == 1)
    
    return is_contiguous

def plot_slices(labels, slices, region_id):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))

    # Define the slice indices ensuring we don't exceed the volume boundaries
    slice_indices = [
        max(slices[0] - 1, 0),
        slices[0]+2,
        slices[len(slices) // 2],
        slices[-1]-2,
        min(slices[-1] + 1, labels.shape[2] - 1)
    ]

    for ax, slice_index in zip(axs, slice_indices):
        # Plot the grayscale volume slice
        ax.imshow(labels[:, :, slice_index], cmap='gray')
        
        # Overlay the selected region in red
        overlay = np.where(labels[:, :, slice_index] == region_id, 1, np.nan)
        ax.imshow(overlay, cmap='Reds', alpha=1, vmin=0, vmax=1)
        
        ax.set_title(f'Slice {slice_index}')

    plt.tight_layout()
    plt.show()





# def process_slice(slice_identifier, plot=False): #Try a more descriptive name..
    
#     tif_filename = fullres_folder + slice_identifier + '.tif'

#     # Load the full-sized tif
#     fs_img = io.imread(tif_filename)

#     # Load the fos image
#     fos_filename = fos_folder + slice_identifier + '.tif'
#     fos_img = io.imread(fos_filename)

#     # Get the index of this image with respect to
#     (img_ind, lab_ind), this_atlas_slice = map_img_to_label(slice_identifier, identifiers, labels)

#     # Example usage:
#     mask = get_mask_from_label(region_id, labels)

#     # Use it for the mask we just created with the full-sized image
#     masked_fs_img = apply_mask_to_img(fs_img, mask)

#     # Apply the mask to the fos image
#     masked_fos_img = apply_mask_to_img(fos_img, mask)

#     if(plot):
#         plt.clf()
#         # Matplotlib subplot figure to compare 2 plt.imshow plots
#         fig, (ax1, ax2) = plt.subplots(1, 2)  

#         # Overlay masked_ds_img on ds_img, highlighting the masked region
#         ax1.imshow(fs_img)
#         ax1.imshow(masked_fs_img, alpha=0.5, vmin=0, vmax=1000)

#         # Overlay masked_ds_img on ds_img, highlighting the masked region
#         ax2.imshow(fos_img, vmin=0, vmax=1000)
#         ax2.imshow(masked_fos_img, alpha=0.5, vmin=0, vmax=500)

#         plt.show()

#         # Save the subnplot figure
#         fig.savefig(output_path + slice_identifier + '.png', dpi=300)

#     # Return any necessary results
#     return masked_fs_img, masked_fos_img, this_atlas_slice
