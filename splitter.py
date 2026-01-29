import splitfolders
import os
import shutil
from config import INPUT_DATA_PATH, EXPORT_DATA_PATH

# Paths
input_folder = INPUT_DATA_PATH
output_folder =  EXPORT_DATA_PATH


# input_folder = '/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Datasets/new_ph_clips' #INPUT_DATA_PATH
# output_folder = '/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Datasets/new_ph_clips_exported' #EXPORT_DATA_PATH


# Remove output folder if it exists
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Split directly into train/val/test with 80/10/10 ratio
splitfolders.ratio(
    input_folder, 
    output=output_folder, 
    seed=42, 
    ratio=(0.8, 0.1, 0.1)  # train, val, test
)

print("Dataset successfully split into train (80%), val (10%), and test (10%)!")

