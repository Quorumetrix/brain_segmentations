#file_io.py

import os
import numpy as np
import imageio
from skimage import io

from brain_segmentations.config import *

'''There are loaded from th config.
But they may change if a part of a loop 
or as a part of a list
so for now they are left lower case'''

# experiment_folder
# neun_folder
# cFOS_folder

# 
def load_single_image(img_identifier, folder='neun'):
    
    # Check if the img_identifier already has a '.tif' extension
    if not img_identifier.lower().endswith('.tif'):
        img_identifier += '.tif'

    if folder=='neun':  
        path =  ROOT+experiment_folder+neun_folder + img_identifier
    
        img = io.imread(path).astype(np.int32) 
        
    elif folder=='cfos':
        path = ROOT+experiment_folder+cFOS_folder + img_identifier
        img = io.imread(path).astype(np.int32) 

    return img


def load_paired_images(img_identifier):
    
    fos_path = ROOT+experiment_folder+cFOS_folder + img_identifier#+'.tif'
    neun_path =  ROOT+experiment_folder+neun_folder + img_identifier#+'.tif'
   
    fos_img = io.imread(fos_path).astype(np.int32) 
    neun_img = io.imread(neun_path).astype(np.int32) 
#     neun_img = np.empty(fos_img.shape)
    
    print('loaded: ', img_identifier, ' from: ',fos_path)
    print('min, max: ', np.min(fos_img), np.max(fos_img))
    
    return neun_img, fos_img


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

