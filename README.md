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


yolo detect train \
  model=yolo11n.pt \
  data=datasets/rear_trucks/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=32 \
  device=0 \
  project=KICSafe_Research \
  name=yolo11n_stable \
  workers=8 \
  mosaic=0.0 \
  cache=True \
  amp=True
