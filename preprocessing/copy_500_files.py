import os
import shutil

source_folder = 'path/to/source/folder'
destination_folder = 'path/to/destination/folder'

target_file_count = 500

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# List all files in the source folder
file_list = os.listdir(source_folder)

# Filter only '.tif' files
tif_files = [f for f in file_list if f.endswith('.tif')]

total_files = len(tif_files)

# Calculate the value of N to get the target number of files
N = max(total_files // target_file_count, 1)

# Sort the list of files
sorted_tif_files = sorted(tif_files)

# Copy every Nth file to the destination folder
for i, file in enumerate(sorted_tif_files):
    if i % N == 0:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        shutil.copy2(source_path, destination_path)

print(f"Files copied successfully. N={N}")
