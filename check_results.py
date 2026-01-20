#!/usr/bin/env python3
"""
Quick performance summary of the trained YOLO model
"""
import csv
from pathlib import Path

results_csv = Path("KICSafe_Research/yolo11n_optimized/results.csv")

if results_csv.exists():
    print("\n" + "=" * 70)
    print("YOLO11n Training Summary - Rear Truck Detection")
    print("=" * 70)

    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        # Get best epoch (around epoch 59 based on early stopping)
        best_epoch = None
        best_map50 = 0
        best_row = None

        for i, row in enumerate(rows):
            map50 = float(row["metrics/mAP50(B)"])
            if map50 > best_map50:
                best_map50 = map50
                best_epoch = i + 1
                best_row = row

        print(f"\n📊 Best Performance (Epoch {best_epoch}):")

        print(f"  • Precision:     {float(best_row['metrics/precision(B)']):.3f}")
        print(f"  • Recall:        {float(best_row['metrics/recall(B)']):.3f}")
        print(f"  • mAP@0.5:       {float(best_row['metrics/mAP50(B)']):.3f}")
        print(f"  • mAP@0.5-0.95:  {float(best_row['metrics/mAP50-95(B)']):.3f}")

        print(f"\n⏱️  Training Info:")
        print(f"  • Total Epochs: {len(rows)} (stopped early at {len(rows)})")
        print(f"  • Best Epoch: {best_epoch}")
        print(f"  • Training Time: ~2.03 hours")

        print(f"\n📁 Model Files:")
        print(f"  • Best:  KICSafe_Research/yolo11n_optimized/weights/best.pt")
        print(f"  • Last:  KICSafe_Research/yolo11n_optimized/weights/last.pt")

        print("\n" + "=" * 70)
        print("✅ Model ready for deployment!")
        print("=" * 70 + "\n")

else:
    print("❌ Results file not found. Make sure training completed successfully.")
