import os
import random
import shutil
from pathlib import Path

# adjustable ratio
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1  # Must sum to 1.0

# Base paths relative to this script
BASE_IMG = Path("data_raw/images")
BASE_LBL = Path("data_raw/labels")

def setup_folders(base_path):
    """Creates train, val, and test subfolders."""
    for folder in ['train', 'val', 'test']:
        (base_path / folder).mkdir(parents=True, exist_ok=True)

def main():
    # 1. Get all raw images (not already in subfolders)
    images = [f for f in os.listdir(BASE_IMG) if f.endswith(".jpg")]
    
    if not images:
        print("No images found in root folder. Are they already moved?")
        return

    random.shuffle(images)

    # 2. Calculate Split Points
    total = len(images)
    train_idx = int(total * TRAIN_RATIO)
    val_idx = train_idx + int(total * VAL_RATIO)

    # 3. Define the Assignments
    splits = {
        'train': images[:train_idx],
        'val':   images[train_idx:val_idx],
        'test':  images[val_idx:]
    }

    # 4. Move Files
    setup_folders(BASE_IMG)
    setup_folders(BASE_LBL)

    for folder_name, file_list in splits.items():
        print(f"Moving {len(file_list)} files to {folder_name}...")
        for img_name in file_list:
            # Move Image
            shutil.move(BASE_IMG / img_name, BASE_IMG / folder_name / img_name)
            
            # Move corresponding Label
            lbl_name = img_name.replace(".jpg", ".txt")
            if (BASE_LBL / lbl_name).exists():
                shutil.move(BASE_LBL / lbl_name, BASE_LBL / folder_name / lbl_name)

    print("Dataset successfully split!")

if __name__ == "__main__":
    main()