#!/bin/bash

sudo jetson_clocks
echo ""
echo "========================================="
echo "Starting YOLO Training"
echo "========================================="

# Activate virtual environment
source venv/bin/activate

# BALANCED TRAINING COMMAND - PREVENTS OVERCURRENT THROTTLING
# Optimized to avoid power issues while maintaining performance
# - batch: 16 (balanced GPU load, prevents overcurrent)
# - workers: 6 (reduced CPU load to lower total system power)
# - mosaic: 0.8 (reduced augmentation intensity)
# - rect: True (reduces compute load per batch)
# - multi_scale: False (disabled to reduce power spikes)

yolo detect train \
  model=yolo26l.pt \
  data=datasets/rear_trucks/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=KICSafe_Research \
  name=yolo11n_optimized \
  workers=10 \
  mosaic=0.8 \
  cache=True \
  amp=True \
  rect=True \
  patience=20

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
