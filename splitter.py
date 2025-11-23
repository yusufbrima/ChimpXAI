# import splitfolders
# import os
# import shutil
# from config import INPUT_DATA_PATH, EXPORT_DATA_PATH,DATA_SENTINEL

# print(INPUT_DATA_PATH, EXPORT_DATA_PATH)
# # Input and output paths
# input_folder = INPUT_DATA_PATH
# output_folder = EXPORT_DATA_PATH

# # Remove output folder if it exists
# if os.path.exists(output_folder):
#     shutil.rmtree(output_folder)
# os.makedirs(output_folder, exist_ok=True)

# # Step 1: Split original dataset into train (90%) and val (10%)
# splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.9, 0.1))

# # Step 2: Rename 'val' to 'test'
# old_val_path = os.path.join(output_folder, "val")
# test_path = os.path.join(output_folder, "test")
# if os.path.exists(old_val_path):
#     os.rename(old_val_path, test_path)

# # Step 3: Rename original 'train' to 'train_full' before further splitting
# original_train_path = os.path.join(output_folder, "train")
# train_full_path = os.path.join(output_folder, "train_full")
# if os.path.exists(original_train_path):
#     os.rename(original_train_path, train_full_path)

# # Step 4: Split 'train_full' into 'train' and 'val' (e.g., 80-20 split of the original train)
# splitfolders.ratio(train_full_path, output=output_folder, seed=42, ratio=(0.8, 0.2))

# print("Dataset successfully split into train, val, and test!")

import splitfolders
import os
import shutil
from config import INPUT_DATA_PATH, EXPORT_DATA_PATH

# Paths
input_folder = INPUT_DATA_PATH
output_folder = EXPORT_DATA_PATH

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

