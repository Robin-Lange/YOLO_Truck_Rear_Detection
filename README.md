# carissma_camera_detection

Yolo26 https://docs.ultralytics.com/models/yolo26/
BDD110K dataset https://bair.berkeley.edu/blog/2018/05/30/bdd/ 
KICSafe Project https://www.thi.de/en/forschung/carissma/c-isafe-navi/projekte/kicsafe/

source venv/bin/activate


How to install pytorch
pip install typeguard
pip install pyyaml
python3 -m pip install torch==2.8.0 torchvision==0.23.0 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
sudo apt-get remove python3-torchaudio
python3 -c "import torch; print(f'Version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(torch.__file__)"

pip install 'numpy<2'

source venv/bin/activate

Add for vehicle orientation
pip install tensorflow

then in folder run


Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    300/300      5.89G     0.2292    0.07607   0.001205          5        640: 100% ━━━━━━━━━━━━ 1277/1277 1.3it/s 16:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 172/172 2.9it/s 58.7s
                   all       2749       3535      0.971      0.945      0.979      0.895

300 epochs completed in 85.886 hours.
Optimizer stripped from /mnt/nova_ssd/workspaces/carissma_camera_detection/runs/yolo26l_truck_rear_final_run/weights/last.pt, 53.0MB
Optimizer stripped from /mnt/nova_ssd/workspaces/carissma_camera_detection/runs/yolo26l_truck_rear_final_run/weights/best.pt, 53.0MB

Validating /mnt/nova_ssd/workspaces/carissma_camera_detection/runs/yolo26l_truck_rear_final_run/weights/best.pt...
Ultralytics 8.4.6 🚀 Python-3.10.12 torch-2.8.0 CUDA:0 (Orin, 62841MiB)
YOLO26l summary (fused): 190 layers, 24,746,511 parameters, 0 gradients, 86.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 172/172 3.7it/s 46.1s
                   all       2749       3535      0.973       0.94      0.982      0.897
Speed: 0.1ms preprocess, 10.2ms inference, 0.0ms loss, 0.4ms postprocess per image
Results saved to /mnt/nova_ssd/workspaces/carissma_camera_detection/runs/yolo26l_truck_rear_final_run
💡 Learn more at https://docs.ultralytics.com/modes/train
VS Code: view Ultralytics VS Code Extension ⚡ at https://docs.ultralytics.com/integrations/vscode

python detect.py --weights best.pt --source ../datasets/rear_trucks/images/val --img 640 --conf 0.5 --project runs/detect --name bdd_filtered_test --classes 0 --save-txt

image 646/646 /mnt/nova_ssd/workspaces/carissma_camera_detection/datasets/rear_trucks/images/val/fe1f55fa-19ba3600.jpg: 384x640 1 truck_rear, 52.4ms
Speed: 2.9ms preprocess, 52.5ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
Results saved to /mnt/nova_ssd/workspaces/carissma_camera_detection/runs/detect/runs/detect/bdd_filtered_test
385 labels saved to /mnt/nova_ssd/workspaces/carissma_camera_detection/runs/detect/runs/detect/bdd_filtered_test/labels

after defining custom yaml file

python detect.py \
  --weights best-fp16.tflite \
  --source ../datasets/rear_trucks/images/val \
  --img 640 \
  --conf 0.5 \
  --data data/vehicle_orientation.yaml \
  --project runs/detect \
  --name vehicle_orientation_tflite_test2 \
  --classes 6 \
  --save-txt

Speed: 1.7ms pre-process, 5006.8ms inference, 4.3ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs/detect/vehicle_orientation_tflite_test32
331 labels saved to runs/detect/vehicle_orientation_tflite_test32/labels



PS C:\Users\0507r\Downloads\model_comp> python compare_detections.py

══════════════════════════════════════════════════════════════════════
  DETECTION COMPARISON  –  646 images
──────────────────────────────────────────────────────────────────────
  Model A  YOLO26  (BDD trained) : C:\Users\0507r\Downloads\model_comp\bdd_filtered_test\labels
  Model B  TFLite  (VehicleOri)  : C:\Users\0507r\Downloads\model_comp\vehicle_orientation_tflite_test32\labels
  Output                         : C:\Users\0507r\Downloads\model_comp\comparison_output
══════════════════════════════════════════════════════════════════════


══════════════════════════════════════════════════════════════════════
  RESULTS  (646 images)
──────────────────────────────────────────────────────────────────────
  Both detected       (agreement  ✓)             296  ( 45.8%)  ██████████████████████
  YOLO26 only         (TFLite missed  ✗)          89  ( 13.8%)  ██████
  TFLite only         (YOLO26 missed  ✗)          35  (  5.4%)  ██
  Neither detected    (no visible truck rear)    226  ( 35.0%)  █████████████████
──────────────────────────────────────────────────────────────────────
  Agreement   :  522 / 646  (80.8%)
  Disagreement:  124 / 646  (19.2%)

  Proxy recall  (among 420 images with ≥1 detection):
    YOLO26 (BDD)    : 0.917  →  385 detections
    TFLite (VehOri) : 0.788  →  331 detections

  Raw detection count cross-check:
    YOLO26 label files  : 385  (log reported 385)
    TFLite label files  : 331  (log reported 331)

  CSV saved → C:\Users\0507r\Downloads\model_comp\comparison_output\comparison_report.csv
  Vis saved → C:\Users\0507r\Downloads\model_comp\comparison_output\side_by_side  (124 images rendered)
══════════════════════════════════════════════════════════════════════
