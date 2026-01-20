import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# BDD100K with per-image JSON files (image.jpg + image.json pairs)
RAW_IMAGES_TRAIN = "bdd100k_raw/100k/train"
RAW_IMAGES_VAL = "bdd100k_raw/100k/val"
OUTPUT_DIR = "datasets/rear_trucks"

# Filtering parameters
MIN_HEIGHT = 80  # Increased from 50 to 80 to ignore far-away background trucks
# Changed from 1.4 to 1.0 to strictly target tall/square rear ends
MAX_ASPECT_RATIO = 1.0  # Max width/height ratio for rear view


def filter_rear_trucks(images_path, output_split, image_width=1280, image_height=720):
    """
    Filter BDD100K dataset for rear-view trucks from per-image JSON files.

    Args:
        images_path: Path to folder with image.jpg + image.json pairs
        output_split: 'train' or 'val'
        image_width: BDD100K image width
        image_height: BDD100K image height

    Returns:
        Number of images processed
    """

    print(f"\nProcessing {output_split} set...")
    print(f"Loading from: {images_path}")

    images_dir = Path(images_path)
    if not images_dir.exists():
        print(f"❌ Error: Directory not found at {images_path}")
        return 0

    # Get all JSON files
    json_files = list(images_dir.glob("*.json"))

    if len(json_files) == 0:
        print(f"❌ Error: No JSON files found in {images_path}")
        print(f"Please verify the structure - each .jpg should have a matching .json")
        return 0

    print(f"Found {len(json_files):,} JSON annotation files")

    count = 0
    stats = {
        "total_images": len(json_files),
        "total_trucks": 0,
        "filtered_aspect_ratio": 0,
        "filtered_size": 0,
        "filtered_attributes": 0,
        "kept_rear_trucks": 0,
    }

    for json_file in tqdm(json_files, desc=f"Filtering {output_split}"):
        # Load JSON annotation
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️  Warning: Could not parse {json_file.name}")
            continue

        # Get image name
        image_name = json_file.stem + ".jpg"  # Replace .json with .jpg
        image_path = images_dir / image_name

        if not image_path.exists():
            print(f"⚠️  Warning: Image not found for {json_file.name}")
            continue

        # Get labels from JSON
        # BDD100K format: {"name": "...", "frames": [{"objects": [...]}], "attributes": {...}}
        frames = data.get("frames", [])
        if not frames:
            continue

        # Get objects from first frame (video format with single frame)
        objects = frames[0].get("objects", []) if frames else []
        if not objects:
            continue

        yolo_lines = []
        has_rear_truck = False

        for obj in objects:
            category = obj.get("category", "")

            # Only process trucks
            if category != "truck":
                continue

            stats["total_trucks"] += 1

            # Get bounding box - BDD100K format: "box2d": {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
            box = obj.get("box2d")
            if not box:
                continue

            x1 = box.get("x1")
            y1 = box.get("y1")
            x2 = box.get("x2")
            y2 = box.get("y2")

            if None in (x1, y1, x2, y2):
                continue

            # Calculate dimensions
            width = x2 - x1
            height = y2 - y1

            # Filter 1: Aspect Ratio
            # Rear view: roughly square or tall (ratio ≤ 1.4)
            # Side view: wide (ratio > 2.0)
            aspect_ratio = width / height if height > 0 else 999
            if aspect_ratio > MAX_ASPECT_RATIO:
                stats["filtered_aspect_ratio"] += 1
                continue

            # Filter 2: Minimum size
            # Too small = too far away for underride risk
            if height < MIN_HEIGHT:
                stats["filtered_size"] += 1
                continue

            # Filter 3: BDD100K attributes (optional)
            attributes = obj.get("attributes", {})
            if attributes.get("occluded", False) and attributes.get("truncated", False):
                # Skip heavily occluded AND truncated trucks
                stats["filtered_attributes"] += 1
                continue

            # Convert to YOLO format (normalized coordinates)
            dw = 1.0 / image_width
            dh = 1.0 / image_height
            cx = (x1 + x2) / 2.0 * dw
            cy = (y1 + y2) / 2.0 * dh
            w = width * dw
            h = height * dh

            # Ensure coordinates are within bounds
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = max(0, min(1, w))
            h = max(0, min(1, h))

            # Class 0 = Rear Truck
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            has_rear_truck = True
            stats["kept_rear_trucks"] += 1

        # Save if we found valid rear trucks
        if has_rear_truck:
            # Save label file
            txt_name = image_name.replace(".jpg", ".txt")
            label_path = f"{OUTPUT_DIR}/labels/{output_split}/{txt_name}"
            with open(label_path, "w") as out_f:
                out_f.write("\n".join(yolo_lines))

            # Copy image
            src = image_path
            dst = os.path.join(OUTPUT_DIR, "images", output_split, image_name)

            shutil.copy(src, dst)
            count += 1

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Statistics for {output_split} set:")
    print(f"{'='*60}")
    print(f"Total images processed:       {stats['total_images']:,}")
    print(f"Total trucks found:           {stats['total_trucks']:,}")
    print(f"Filtered (aspect ratio):      {stats['filtered_aspect_ratio']:,}")
    print(f"Filtered (too small):         {stats['filtered_size']:,}")
    print(f"Filtered (occlusion):         {stats['filtered_attributes']:,}")
    print(f"Kept rear trucks:             {stats['kept_rear_trucks']:,}")
    print(f"Images with rear trucks:      {count:,}")
    print(f"{'='*60}\n")

    return count


def create_yaml_config(num_train, num_val):
    """Create YOLO dataset configuration file."""

    yaml_content = f"""# KICSafe Rear Truck Detection Dataset
# BDD100K filtered for rear-view semi trucks (underride crash detection)

path: {os.path.abspath(OUTPUT_DIR)}  # Dataset root
train: images/train  # Train images (relative to 'path')
val: images/val      # Validation images (relative to 'path')

# Classes
names:
  0: rear_truck

# Dataset Info
nc: 1  # Number of classes
images_train: {num_train}
images_val: {num_val}

# Filtering criteria applied:
# - Aspect ratio: width/height <= {MAX_ASPECT_RATIO}
# - Minimum height: {MIN_HEIGHT}px
# - Focus: Rear-view semi trucks for underride detection
"""

    yaml_path = f"{OUTPUT_DIR}/data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"✅ Created YOLO config: {yaml_path}")


def verify_structure():
    """Verify the input dataset structure."""
    print("=" * 60)
    print("VERIFYING INPUT STRUCTURE")
    print("=" * 60)

    train_dir = Path(RAW_IMAGES_TRAIN)
    val_dir = Path(RAW_IMAGES_VAL)

    train_exists = train_dir.exists()
    val_exists = val_dir.exists()

    if not train_exists and not val_exists:
        print(f"❌ Neither train nor val directory found!")
        print(f"   Train: {RAW_IMAGES_TRAIN}")
        print(f"   Val: {RAW_IMAGES_VAL}")
        return False, False, False

    # Check for jpg and json pairs
    train_jpgs = list(train_dir.glob("*.jpg")) if train_exists else []
    train_jsons = list(train_dir.glob("*.json")) if train_exists else []
    val_jpgs = list(val_dir.glob("*.jpg")) if val_exists else []
    val_jsons = list(val_dir.glob("*.json")) if val_exists else []

    print(f"Train: {len(train_jpgs):,} images, {len(train_jsons):,} JSON files")
    print(f"Val:   {len(val_jpgs):,} images, {len(val_jsons):,} JSON files")

    # Determine which splits to process
    process_train = len(train_jpgs) > 0 and len(train_jsons) > 0
    process_val = len(val_jpgs) > 0 and len(val_jsons) > 0

    if not process_train and not process_val:
        print("❌ No valid data found in either train or val directory!")
        return False, False, False

    if not process_train:
        print("⚠️  Train directory is empty or missing - will skip training set")
    if not process_val:
        print("⚠️  Val directory is empty or missing - will skip validation set")

    # Check a sample pair from available data
    sample_jsons = train_jsons if train_jsons else val_jsons
    sample_dir = train_dir if train_jsons else val_dir

    if sample_jsons:
        sample_json = sample_jsons[0]
        sample_jpg = sample_dir / (sample_json.stem + ".jpg")

        if sample_jpg.exists():
            print(f"✅ Sample pair verified: {sample_json.name} ↔ {sample_jpg.name}")
        else:
            print(f"⚠️  Warning: JSON without matching image: {sample_json.name}")

    print("=" * 60 + "\n")
    return True, process_train, process_val


def main():
    print("=" * 60)
    print("KICSafe Project: BDD100K Rear Truck Dataset Generator")
    print("Per-Image JSON Format (image.jpg + image.json pairs)")
    print("=" * 60)

    # Verify input structure
    valid, process_train, process_val = verify_structure()
    if not valid:
        print("\n❌ Dataset structure verification failed!")
        print("Expected structure:")
        print("  bdd100k_raw/100k/train/  (or)")
        print("    ├── image1.jpg")
        print("    ├── image1.json")
        print("    └── ...")
        print("  bdd100k_raw/100k/val/")
        print("    ├── image1.jpg")
        print("    ├── image1.json")
        print("    └── ...")
        return

    # Create directory structure for splits we'll process
    print("Creating output directory structure...")
    splits_to_create = []
    if process_train:
        splits_to_create.append("train")
    if process_val:
        splits_to_create.append("val")

    for split in splits_to_create:
        Path(f"{OUTPUT_DIR}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_DIR}/labels/{split}").mkdir(parents=True, exist_ok=True)
    print("✅ Directories created\n")

    # Process training set (if available)
    train_count = 0
    if process_train:
        train_count = filter_rear_trucks(RAW_IMAGES_TRAIN, "train")
    else:
        print("\n⏭️  Skipping training set (no data available)\n")

    # Process validation set (if available)
    val_count = 0
    if process_val:
        val_count = filter_rear_trucks(RAW_IMAGES_VAL, "val")
    else:
        print("\n⏭️  Skipping validation set (no data available)\n")

    # Create YAML configuration
    create_yaml_config(train_count, val_count)

    # Final summary
    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETE")
    print("=" * 60)
    print(f"Training images:      {train_count:,}")
    print(f"Validation images:    {val_count:,}")
    print(f"Total images:         {train_count + val_count:,}")
    print(f"\nDataset location: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)
    print("\n✅ Ready for YOLO training!")
    print("Next step: Run the YOLO training script")
    print("  python train_yolo.py")


if __name__ == "__main__":
    main()
