# Truck Rear & Underride Guard Detection
https://www.forbes.com/sites/jensen/2017/05/10/new-tests-evaluate-how-well-side-guards-on-semis-do-at-preventing-side-underride-crashes/
## Project Overview

This is a computer vision research project for **AEBS (Advanced Emergency Braking System) validation**, focused on detecting the **rear of trucks** and specifically the **underride protection guard**. The project is part of the [KICSafe](https://www.thi.de/en/forschung/carissma/c-isafe-navi/projekte/kicsafe/) research initiative at CARISSMA (THI).

Three models are compared:

| Model | Type | Weight File | Description |
|-------|------|-------------|-------------|
| **YOLO26l** | Supervised (fine-tuned) | `models/yolo26l/best.pt` | YOLO26 Large, trained 300 epochs on BDD100K rear trucks |
| **YOLOv5** | Supervised (TFLite) | `models/yolov5/best-fp16.tflite` | YOLOv5 FP16 TFLite, vehicle orientation detection |
| **YOLOe** | Zero-shot (visual prompts) | `models/yoloe/yoloe-26l-seg.pt` | YOLOe with visual prompts for underride guard segmentation |

The YOLOe model uses a **two-stage approach**: detect the underride guard in the full image using visual prompts, then crop the bottom 35% of each detection for refined confirmation.

---

## Repository Structure

```
project_root/
├── data/
│   ├── raw/                        # Original unmodified images
│   │   ├── bdd100k_samples/        # 25 BDD100K frames selected for validation
│   │   ├── aebs_target/            # AEBS soft target photos
│   │   └── real_trucks/            # Real truck photos (Arspet, outdoor)
│   ├── prompts/                    # Visual prompt reference images for YOLOe
│   │   ├── prompt_underride_na.jpg     # Stoughton North American guard
│   │   ├── prompt_underride_eu.jpg     # STAS European tubular guard
│   │   └── prompt_underride_aebs.jpg   # Cropped from AEBS soft target
│   └── labeled/                    # 25 labeled images for quantitative evaluation
│       ├── images/                 # Images (jpg)
│       └── labels/                 # YOLO format .txt label files
├── datasets/
│   └── rear_trucks/                # BDD100K filtered dataset (4595 train / 646 val)
│       ├── data.yaml
│       ├── images/
│       └── labels/
├── models/
│   ├── yolo26l/
│   │   └── best.pt                # YOLO26l fine-tuned on rear trucks
│   ├── yolov5/
│   │   └── best-fp16.tflite       # YOLOv5 FP16 TFLite
│   └── yoloe/
│       └── yoloe-26l-seg.pt       # YOLOe segmentation model
├── scripts/
│   ├── underride_detection.py      # YOLOe two-stage underride detection
│   ├── label_images.py             # Manual image labeling tool (OpenCV)
│   ├── run_comparison.py           # Run all 3 models and compare results
│   ├── filter_bdd100k.py           # Filter BDD100K for rear trucks
│   ├── prepare_dataset.py          # Prepare vehicle-orientation dataset
│   ├── check_results.py            # Print training results summary
│   ├── hardwaretest.py             # GPU/CUDA hardware check
│   ├── optimized_train.sh          # YOLO26l training script (Jetson)
│   ├── validate_model.sh           # Validation script
│   └── test_predictions.sh         # Quick inference test
├── results/
│   ├── yolo26l/                    # YOLO26l detection outputs
│   ├── yolov5/                     # YOLOv5 detection outputs
│   ├── yoloe/                      # YOLOe detection outputs
│   └── summary.csv                 # Combined comparison CSV
├── matlab/
│   ├── quantitative_eval.m         # Quantitative evaluation & plots
│   ├── setup_image_labeler.m       # MATLAB Image Labeler setup
│   ├── export_labels.m             # Export Image Labeler -> YOLO format
│   ├── images/                     # 25 selected images for MATLAB labeling
│   └── labels/                     # YOLO labels (from Python or MATLAB)
├── runs/                           # Training run artifacts
├── yolov5/                         # YOLOv5 repository clone
├── .gitignore
└── README.md
```

---

## Setup & Installation

This project runs on a **Jetson Orin** with CUDA. All commands assume you are in the project root.

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch for Jetson (JP6 / CUDA 12.6)
pip install typeguard pyyaml
python3 -m pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

# Install core dependencies
pip install 'numpy<2'
pip install ultralytics opencv-python pillow tqdm

# Verify installation
python scripts/hardwaretest.py
```

**Current environment:**
- Python 3.10.12
- PyTorch 2.8.0 (CUDA)
- Ultralytics 8.4.6
- OpenCV 4.13.0

---

## Data

### Raw Images

| Subdirectory | Contents | Count |
|---|---|---|
| `data/raw/bdd100k_samples/` | BDD100K dashcam frames with visible truck rears | 25 |
| `data/raw/aebs_target/` | AEBS soft target photos | 1 |
| `data/raw/real_trucks/` | Real truck photographs (outdoor) | 2 |

### Visual Prompts for YOLOe

YOLOe uses these reference images as visual prompts for zero-shot detection:

| File | Description |
|------|-------------|
| `data/prompts/prompt_underride_na.jpg` | Stoughton North American underride guard |
| `data/prompts/prompt_underride_eu.jpg` | STAS European tubular underride guard |
| `data/prompts/prompt_underride_aebs.jpg` | Underride guard cropped from AEBS soft target |

### Labeled Validation Set

25 images selected from `datasets/rear_trucks/images/val/` with the largest truck rear bounding boxes for clear evaluation.

- **Format:** YOLO `.txt` — one line per box: `<class_id> <cx> <cy> <w> <h>` (normalized)
- **Class 0:** `truck_rear`
- **Class 1:** `underride_guard` *(to be annotated via label_images.py or MATLAB)*
- **Location:** `data/labeled/images/` + `data/labeled/labels/`

### BDD100K Filtered Dataset

The training dataset in `datasets/rear_trucks/` was filtered from BDD100K:
- **Train:** 4,595 images
- **Val:** 646 images
- **Filtering:** aspect ratio ≤ 1.0, min height ≥ 80 px, rear-view trucks only

---

## Models

| Model | Path | Size | Details |
|-------|------|------|---------|
| YOLO26l | `models/yolo26l/best.pt` | ~53 MB | 190 layers, 24.7M params, 86.1 GFLOPs |
| YOLOv5 TFLite | `models/yolov5/best-fp16.tflite` | ~88 MB | FP16 quantized, vehicle orientation |
| YOLOe | `models/yoloe/yoloe-26l-seg.pt` | — | Zero-shot segmentation with visual prompts |

### YOLO26l Training Results (300 epochs)

```
Class     Images  Instances    P       R     mAP50  mAP50-95
  all      2749      3535    0.973   0.940   0.982    0.897
Speed: 0.1ms preprocess, 10.2ms inference, 0.4ms postprocess per image
```

---

## Running Inference

### YOLO26l (best.pt)

```bash
source venv/bin/activate

# Run on validation set
yolo detect predict \
  model=models/yolo26l/best.pt \
  source=datasets/rear_trucks/images/val/ \
  imgsz=640 \
  conf=0.5 \
  save=True

# Or use the script
bash scripts/test_predictions.sh
```

### YOLOv5 (TFLite)

```bash
source venv/bin/activate
cd yolov5

python detect.py \
  --weights ../models/yolov5/best-fp16.tflite \
  --source ../datasets/rear_trucks/images/val \
  --img 640 \
  --conf 0.25 \
  --project ../runs/detect \
  --name yolov5_eval \
  --save-txt
```

### YOLOe — Two-Stage Underride Detection

```bash
source venv/bin/activate
python scripts/annotate_guard_prompt.py data/prompts/prompt_underride_na.jpg
python scripts/underride_detection.py --source data/labeled/images/ --conf 0.25
```

The script loads visual prompts from `data/prompts/`, runs detection on the full image, then crops the bottom 35% of each detection for refined underride guard confirmation.

---

## Labeling New Images

```bash
source venv/bin/activate

# Label all images in data/raw/
python scripts/label_images.py

# Or label a specific directory
python scripts/label_images.py --source data/raw/bdd100k_samples/
```

**Controls:**
| Key | Action |
|-----|--------|
| `1` | Select class `truck_rear` |
| `2` | Select class `underride_guard` |
| Click+Drag | Draw bounding box |
| `s` | Save annotation + annotated image |
| `n` | Skip to next image |
| `u` | Undo last box |
| `q` | Quit and print summary |

Labels are saved in YOLO format to `data/labeled/labels/` and annotated images to `data/labeled/images/`.

---

## Running the Model Comparison

```bash
source venv/bin/activate
python scripts/run_comparison.py --conf 0.25 --iou-threshold 0.5
```

**Outputs:**
- `results/yolo26l/detections.json` — per-image detection details
- `results/yolov5/detections.json`
- `results/yoloe/detections.json`
- `results/summary.csv` — combined CSV with columns: `image, model, class, iou, conf, detected`

---

## Quantitative Evaluation in MATLAB

### Image Labeler (optional — for manual label refinement)

```matlab
% Navigate to matlab/ folder in MATLAB
cd matlab
setup_image_labeler   % Opens Image Labeler with 25 images + label definitions
% ... label images in GUI ...
% Export -> To Workspace -> name "gTruth"
export_labels         % Converts to YOLO format .txt files
```

### Generating Evaluation Plots

```matlab
cd matlab
quantitative_eval     % Reads results/summary.csv, generates plots
```

**Plots generated (saved to `results/`):**
- `mean_iou_comparison.png` — Mean IoU per model per class (grouped bar chart)
- `precision_recall_chart.png` — Precision & Recall per model
- `detection_rate_chart.png` — Detection rate per model
- `f1_comparison.png` — F1 score comparison

---

## Results

*To be filled after running `scripts/run_comparison.py` and `matlab/quantitative_eval.m`.*

| Model | Class | Mean IoU | Precision | Recall | F1 |
|-------|-------|----------|-----------|--------|-----|
| YOLO26l | truck_rear | | | | |
| YOLOv5 | truck_rear | | | | |
| YOLOe | underride_guard | | | | |

---

## Notes

- **BDD100K** is a dashcam dataset; underride guards are only detectable at close range behind trucks.
- **YOLOe** uses visual prompts (zero-shot) — no fine-tuning required, but performance depends on prompt quality.
- **AEBS soft target images** should be included in the validation set for domain coverage.
- The YOLO26l model was trained for 300 epochs (~86 hours) on a Jetson Orin with 62 GB VRAM.
- The `datasets/rear_trucks/` folder is kept for potential re-training and validation runs.
- MATLAB evaluation is done externally; the `matlab/` folder contains all necessary scripts and images.



=================================================================
Summary
  Images processed   : 20
  Truck rears found  : 25
  Crops checked      : 25
  Guards detected    : 3
  Images with guards : 3
    - b1d0a191-65deaeef.jpg
    - b1e0c01d-dd9e6e2f.jpg
    - b3cc736f-b7a66149.jpg