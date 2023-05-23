ROOT = 'Z://Collaboration_data/Iordonova_lab/'
DS_FACTOR = 16 # Downsampling factor
VOXEL_DIMS = [1.8, 1.8, 4] # x,y,z voxel dimensions in microns of full sized images
experiment_folder = 'Iordanova_06082022_SOC-R9-F_NeuN-cFOS/'
cFOS_folder = '561nm_NeuN/'
neun_folder = '647nm_cFOS/'
print('experiment_folder defined in config. Be careful in the future!',experiment_folder)

# Path to the atlas volume registered to this brain 
atlas_volume_path = 'M://Brain_Registration/brainreg_napari_output/may17_ds16x_fullz/'
# atlas_volume_path = 'M://Brain_Registration/brainreg_napari_output/may10_20ds_fullz_preds/'
atlas_identifier = 'registered_atlas_original_orientation'
atlas_volume_filename = atlas_volume_path + atlas_identifier + '.tiff'


#Path to the file containing the label for each region (this doesn't change for each brain)
REGION_LABEL_PATH = 'Z://Open_data_sets/EBrains/WHS_SD_rat_atlas_v4_pack/WHS_SD_rat_atlas_v4.label'#"your_label_file.txt"
'''
This one should be included with the package to make things easier, 
it could be a relative path to the package folder.
'''

