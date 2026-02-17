import os
import random
import cv2
import glob
from pathlib import Path

# CONFIGURATION
DATASET_DIR = Path("./datasets/truck_rear_v1/images/train")
LABEL_DIR = Path("./datasets/truck_rear_v1/labels/train")
OUTPUT_IMAGE = "visual_verification.jpg"


def draw_yolo_box(img, line):
    # Parse YOLO format: class x_center y_center width height
    parts = line.split()
    class_id = int(parts[0])
    x_c, y_c, w, h = map(float, parts[1:])

    # Convert to Pixel Coordinates
    height, width, _ = img.shape
    x_min = int((x_c - w / 2) * width)
    y_min = int((y_c - h / 2) * height)
    x_max = int((x_c + w / 2) * width)
    y_max = int((y_c + h / 2) * height)

    # Draw Rectangle (Green)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    # Add Label
    cv2.putText(
        img,
        f"ID: {class_id}",
        (x_min, y_min - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    return img


def create_mosaic():
    # Get all images
    img_paths = list(DATASET_DIR.glob("*.jpg")) + list(DATASET_DIR.glob("*.png"))

    if not img_paths:
        print("❌ No images found! Did you run the prepare_dataset.py script?")
        return

    print(f"🔍 Found {len(img_paths)} images. Sampling 20 for verification...")

    # Pick 20 random images
    samples = random.sample(img_paths, min(20, len(img_paths)))
    annotated_images = []

    for img_path in samples:
        # Load Image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Load Corresponding Label
        label_path = LABEL_DIR / img_path.with_suffix(".txt").name

        if label_path.exists():
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    img = draw_yolo_box(img, line)

        # Resize for Mosaic (Thumbnail size)
        img = cv2.resize(img, (320, 320))
        annotated_images.append(img)

    # Create Mosaic Grid (4 rows x 5 columns)
    # Create blank rows if we don't have enough images
    while len(annotated_images) < 20:
        annotated_images.append(np.zeros((320, 320, 3), dtype=np.uint8))

    # Stack images
    try:
        row1 = cv2.hconcat(annotated_images[0:5])
        row2 = cv2.hconcat(annotated_images[5:10])
        row3 = cv2.hconcat(annotated_images[10:15])
        row4 = cv2.hconcat(annotated_images[15:20])
        mosaic = cv2.vconcat([row1, row2, row3, row4])

        cv2.imwrite(OUTPUT_IMAGE, mosaic)
        print(
            f"✅ Success! Verification image saved to: {os.path.abspath(OUTPUT_IMAGE)}"
        )
        print(
            "👉 Open this image to verify that the Green Boxes are indeed TRUCK REARS."
        )
    except Exception as e:
        print(f"Error building mosaic: {e}")


if __name__ == "__main__":
    create_mosaic()
