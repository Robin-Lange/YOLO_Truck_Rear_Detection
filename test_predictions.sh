#!/bin/bash
# Test the model on sample validation images

source venv/bin/activate

echo "Running inference on sample validation images..."
yolo detect predict \
  model=KICSafe_Research/yolo11n_optimized/weights/best.pt \
  source=datasets/rear_trucks/images/val/ \
  imgsz=640 \
  device=0 \
  conf=0.5 \
  save=True \
  max_det=10 \
  line_width=2

echo ""
echo "✅ Predictions saved to: runs/detect/predict/"
echo "Open the folder to see detected rear trucks!"
