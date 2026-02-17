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
python detect.py --weights best-fp16.tflite --source ../datasets/rear_trucks/images/val --img 640 --conf 0.25 --project runs/detect --name bdd_filtered_test --classes 6 --save-txt

then in folder runs



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
