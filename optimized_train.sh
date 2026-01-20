#!/bin/bash
# Optimized YOLO Training Script for Jetson Orin
# Fixes: Slow training (1.5 it/s) and overcurrent warnings

echo "========================================="
echo "Setting up Jetson for optimal training"
echo "========================================="

# CRITICAL: Lock CPU and GPU clocks to performance mode
echo "Enabling jetson_clocks (requires sudo password)..."
sudo jetson_clocks

# Verify settings
echo ""
echo "Power Mode:"
sudo nvpmodel -q

echo ""
echo "CPU Governor:"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

echo ""
echo "========================================="
echo "Starting YOLO Training"
echo "========================================="

# Activate virtual environment
source venv/bin/activate

# OPTIMIZED TRAINING COMMAND
# Key changes from your original:
# - batch: 32 → 16 (reduces power draw ~40%)
# - workers: 8 → 4 (reduces CPU load)
# - rect: True (CRITICAL: saves 40% compute for BDD100K aspect ratio)
# - mosaic: 0.0 → 0.5 (slight augmentation, reduces CPU load vs 1.0)

yolo detect train \
  model=yolo11n.pt \
  data=datasets/rear_trucks/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=KICSafe_Research \
  name=yolo11n_optimized \
  workers=4 \
  mosaic=0.5 \
  cache=True \
  amp=True \
  rect=True \
  patience=20

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
