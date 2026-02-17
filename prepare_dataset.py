import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. WHERE IS YOUR DOWNLOADED DATA?
# Based on your description, it is here:
SOURCE_DIR = Path("./raw_dataset/vehicle-orientation-1")

# 2. WHERE DO YOU WANT THE READY-TO-TRAIN DATA?
OUTPUT_DIR = Path("./datasets/truck_rear_v1")

# 3. CLASS SETTINGS
# In the downloaded dataset, 'truck_back' is ID 6
OLD_CLASS_ID = 6
# In our new custom model, 'truck_back' will be the ONLY class (ID 0)
NEW_CLASS_ID = 0


def process_dataset():
    # Create the destination folders
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Get all .txt files from the source directory
    # (We scan for .txt because we only care about images that HAVE annotations)
    label_files = list(SOURCE_DIR.glob("*.txt"))

    print(f"🔍 Scanning {len(label_files)} annotation files for 'truck_back' (ID 6)...")

    # Statistics counters
    count_kept = 0
    count_skipped = 0

    # Shuffle files to ensure random distribution in Train/Val/Test
    random.shuffle(label_files)

    for label_path in tqdm(label_files):
        # 1. Check if the file contains the target class
        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])

            # KEY STEP: Only keep Class 6 (Truck Rear)
            if class_id == OLD_CLASS_ID:
                # Rewrite line with NEW Class ID 0
                # Keep the original coordinates (parts[1:])
                new_lines.append(f"{NEW_CLASS_ID} {' '.join(parts[1:])}\n")

        # 2. If valid data was found, save it to the new dataset
        if new_lines:
            # Determine split (70% train, 20% val, 10% test)
            rand = random.random()
            if rand < 0.7:
                split = "train"
            elif rand < 0.9:
                split = "val"
            else:
                split = "test"

            # Define output paths
            dest_label_path = OUTPUT_DIR / "labels" / split / label_path.name

            # Find the corresponding image (check .jpg and .png)
            img_name_jpg = label_path.with_suffix(".jpg").name
            src_img_path = label_path.with_suffix(".jpg")

            if not src_img_path.exists():
                # Try png just in case
                src_img_path = label_path.with_suffix(".png")
                if not src_img_path.exists():
                    # Skip if no image found
                    continue

            dest_img_path = OUTPUT_DIR / "images" / split / src_img_path.name

            # Write the NEW label file
            with open(dest_label_path, "w") as f:
                f.writelines(new_lines)

            # Copy the image
            shutil.copy2(src_img_path, dest_img_path)

            count_kept += 1
        else:
            count_skipped += 1

    print(f"\n✅ Processing Complete!")
    print(f"📦 Output Directory: {OUTPUT_DIR.resolve()}")
    print(f"🚛 Images with Truck Rears found & moved: {count_kept}")
    print(f"🗑️ Images skipped (no truck rears): {count_skipped}")
    print("\nNext Step: Run your training command pointing to this folder.")


if __name__ == "__main__":
    process_dataset()
