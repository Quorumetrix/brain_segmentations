# brain_segmentations
 

# Installation

This package is still under active development, and therefore doesn't yet have a straightforward path to install all the components easily within one environment. There are currently several working environments for different parts of the worflow, and the corrsponding .yml files are found in the 'env' subdirectory. 

Presently, the preprocessing steps are done using the 'gpu-analysi.yml' environemnt, which uses clesperanto to improve computational performance using the GPU. 

The atlas registration (brainreg) and interactive segmentation (napari-assistant, clesperanto) steps are done using the 'napari_dev.yml' environment.


# 1. Pre-Processing

In order to align the image volume with the Waxholm space atlas using brainreg, we must first downsample the images. The image downsampling can be computationally expensive, and therefore time-consuming, so the module provides multiple options.

Downsampling with SciKit image (CPU, single threaded): ~ 20s per image
Downsampling with clesperanto (GPU, single threaded):< 1s per image. This ends up being closer to 3s when loading and saving operations are included.

The Notebook "Downsample Images.ipynb" will load the module and filepaths from the configuration file. Running the notebook in entirety will create a file list of all the tifs in the folder, and downsample the images according to the DS_FACTOR defined. 

The preprocessing.py module will automatically try and load clesperanto and perform downsampling operations on the GPU if available. Otherwise, it will default to using SciKit on the CPU. 

# 2. Registration

Registration of the images to the Waxholm space atlas is done in Napari using brainreg. Run Napari, and load the folder of downsampled images as a stack.
File > Open Images As Stack

At this point you will be able to scan through the stack and change the brightness, contrast, gamma, etc. 
Next, load the Atlas Registration plugin in Napari, and enter the following information:

Plugins > Atlas Resitration (brainreg-napari)
Image Layer: The stck you just loaded, should be the name of the first image
Atlas: whs_sd_rat_39um
Data orientation: sal (for horizontal specimens)
Brain geometry: Full brain
Voxel size (z):
Voxel size (x):
Voxel size (y):
Voxel size depends on the original image dimensions, and the downsampling factor. For a 20x downsampled image (in xy, full z), the voxel sizes are voxel_sizes: ['4.0', '36.0', '36.0']
Based on the original data voxel sizes of  (4, 1.8, 1.8) microns

Save original orientation: be sure to check this box.
![Example Image](example_images\napari-brainreg-snapshot.png)

** Ideally this step could be run entirely in a notebook, Atlas Registration.ipynb using the brainreg python module. 

The succesful registration will result in multiple files produced in the output folder. Of importance for us is the file:
"registered_atlas_original_orientation.tiff"
That has transformed the brain atlas to fit the downsampled image volume. 
We can then easily scale this up to apply region-specific masking to the full-sized images. 

![Atlas overlay](example_images\neun_cfos_atlas_overlay.png)



# 3. Segmentation

[In progress...]

Steps:

 - Background substraction
 - Image filtering (Tophat)
 - Thresholding (Otsu)
 - Segmentation (Otsu-Voronoi)
 - Region-specific masking (using registered atlas)
 - Cell counts


 # 4. Summarizing and data visualization

