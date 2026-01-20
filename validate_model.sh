#!/bin/bash
# Validate the trained model and save prediction visualizations

source venv/bin/activate

echo "Running validation with visualizations..."
yolo detect val \
  model=KICSafe_Research/yolo11n_optimized/weights/best.pt \
  data=datasets/rear_trucks/data.yaml \
  imgsz=640 \
  batch=16 \
  device=0 \
  save_json=True \
  save_hybrid=True \
  plots=True

echo ""
echo "Validation results saved!"
echo "Check: runs/detect/val/ for prediction images"
