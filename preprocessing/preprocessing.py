# preprocessing.py

import numpy as np
import math
import imageio
import os
from skimage import io, transform
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from brain_segmentations.config import *
from brain_segmentations.preprocessing.file_io import *




def downsample_image(image, downsample_factor=4):
    downsampled_shape = tuple(s // downsample_factor for s in image.shape)
    return transform.resize(image, downsampled_shape, order=3, anti_aliasing=True)

def upscale_slice(image_slice, original_shape):
    return transform.resize(image_slice, original_shape, order=3, anti_aliasing=True)


def save_downsampled_png(img_ds, filepath):
    
    # Normalize the intensity values to the range [0, 255]
    normalized_img = (img_ds - img_ds.min()) / (img_ds.max() - img_ds.min())
    scaled_img = (normalized_img * 255).astype(np.uint8)

    # Save the image as a 24-bit PNG
    imageio.imwrite(filepath+'.png', scaled_img)


def downsample_folder_pairs(file_list):

    ''' Downsample both images from the NeuN and cFOS folders'''

    for identifier in tqdm(file_list):
        if identifier.endswith('.tif'):
            print(identifier)
            neun_img, cfos_img = load_paired_images(identifier)

            assert neun_img.shape == cfos_img.shape
            IMG_DIM = neun_img.shape # Temp, to config. Then this assert will make sense. 
            assert neun_img.shape == IMG_DIM
            assert cfos_img.shape == IMG_DIM

            neun_img_ds, cfos_img_ds =  downsample_image(neun_img, downsample_factor=DS_FACTOR),  downsample_image(cfos_img, downsample_factor=DS_FACTOR)

            save_downsampled_png(neun_img_ds, neun_output + identifier[:-4])
            save_downsampled_png(cfos_img_ds, cfos_output + identifier[:-4])

        


def downsample_file(identifier, output_dir, sample, tqdm_instance=None):
    if identifier.endswith('.tif'):
        if sample == 'neun':
            neun_img = load_single_image(identifier, folder='neun')
            neun_img_ds = downsample_image(neun_img, downsample_factor=DS_FACTOR)
            save_downsampled_png(neun_img_ds, output_dir + identifier[:-4])

        elif sample == 'cfos':
            cfos_img = load_single_image(identifier, folder='cfos')
            cfos_img_ds = downsample_image(cfos_img, downsample_factor=DS_FACTOR)
            save_downsampled_png(cfos_img_ds, output_dir + identifier[:-4])
    if tqdm_instance is not None:
        tqdm_instance.set_description(f"Processing {identifier}")
        tqdm_instance.update(1)


def downsample_folder(file_list, output_dir, sample='neun', multithread=False,max_workers=10):
    if(multithread):
        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(file_list)) as progress_bar:
            futures = [executor.submit(downsample_file, identifier, output_dir, sample, progress_bar) for identifier in file_list]
            for _ in concurrent.futures.as_completed(futures):
                pass
    else:
        for identifier in tqdm(file_list):
            downsample_file(identifier, output_dir, sample )



try:
    import pyclesperanto_prototype as cle

    device = cle.select_device("TX")
    print("Successfully import pyclesperanto, using GPU: ", device)

    def downsample_image_cle(image, downsample_factor=DS_FACTOR):
        # Calculate the number of iterations needed to achieve the downsample factor
        # We calculate the base 2 logarithm of the downsample factor since each iteration downsamples by half
        iterations = math.log(downsample_factor, 2)
        
        if not iterations.is_integer():
            warnings.warn(f"Downsample factor {downsample_factor} is not a power of 2, "
                        "the actual downsampling factor will be less.")
            iterations = int(iterations)
        else:
            iterations = int(iterations)

        # Initialize the downsampled image
        downsampled_image = image

        # Iteratively downsample the image
        for _ in range(iterations):
            downsampled_image = cle.downsample_slice_by_slice_half_median(downsampled_image)

        return downsampled_image
    
    def downsample_file_cle(identifier, output_dir, sample, tqdm_instance=None):
        
        '''
        The function downsample_file_cle is a wrapper for the downsample_file function.
        Called by the downsample_folder_cle function.
        '''
        
        if identifier.endswith('.tif'):
            if sample == 'neun':
                neun_img = load_single_image(identifier, folder='neun')
                neun_img_ds = downsample_image_cle(neun_img, downsample_factor=DS_FACTOR)
                save_downsampled_png(neun_img_ds, output_dir + identifier[:-4])

            elif sample == 'cfos':
                cfos_img = load_single_image(identifier, folder='cfos')
                cfos_img_ds = downsample_image_cle(cfos_img, downsample_factor=DS_FACTOR)
                save_downsampled_png(cfos_img_ds, output_dir + identifier[:-4])
        if tqdm_instance is not None:
            tqdm_instance.set_description(f"Processing {identifier}")
            tqdm_instance.update(1)


    def downsample_folder(file_list, output_dir, sample='neun', multithread=False,max_workers=10):
        print('Using pyclesperanto to downsample images.')
        if(multithread):
            with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(file_list)) as progress_bar:
                futures = [executor.submit(downsample_file_cle, identifier, output_dir, sample, progress_bar) for identifier in file_list]
                for _ in concurrent.futures.as_completed(futures):
                    pass
        else:
            for identifier in tqdm(file_list):
                downsample_file_cle(identifier, output_dir, sample)




except ImportError:
    print("Failed to import  pyclesperanto_prototype, downsample_image_cle() and downsample_file_cle won't be defined.")
