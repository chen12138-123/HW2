# MLLM Task 2: Road Vehicle Detection and Multi-Object Tracking

This repository contains the code for Task 2 of the Deep Learning and Spatial Intelligence homework: road-scene object detection, video multi-object tracking, occlusion analysis, and line-crossing counting.

## Project Structure

```text
MLLM/
  scripts/
    get_data.py
    train_yolo.py
    test_track.py
    summarize_detection_results.py
    extract_video_frames.py
  analysis/
    detection_summary.csv
    detection_summary.md
  reports/
    task2_report.tex
    figures/
```

Large local directories, such as `trafic_data/`, test videos, tracking outputs, training runs, and model weights, are not committed to GitHub because of file size. The report contains the final metrics and selected visualizations.

## Environment

Recommended server environment:

```text
OS: Ubuntu 20.04/22.04
GPU: NVIDIA RTX 4090
Python: 3.10
CUDA: 12.1
Framework: PyTorch + Ultralytics
```

Create a Conda environment:

```bash
conda create -n mllm-yolo python=3.10 -y
conda activate mllm-yolo
```

Install PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

Check the environment:

```bash
python - <<'PY'
import torch
import ultralytics

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("ultralytics:", ultralytics.__version__)
PY
```

## Dataset

The detector is trained on Road Vehicle Images Dataset.

Download with KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download(
    "ashfakyeafi/road-vehicle-images-dataset",
    output_dir="/mnt/disk/maoqijun/MLLM/trafic_data",
)
print(path)
```

The expected dataset structure is:

```text
trafic_data/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  data_1.yaml
```

Edit `trafic_data/data_1.yaml` for the server path:

```yaml
path: /mnt/disk/maoqijun/MLLM/trafic_data
train: train/images
val: valid/images

nc: 21
names:
  - ambulance
  - army vehicle
  - auto rickshaw
  - bicycle
  - bus
  - car
  - garbagevan
  - human hauler
  - minibus
  - minivan
  - motorbike
  - pickup
  - policecar
  - rickshaw
  - scooter
  - suv
  - taxi
  - three wheelers -CNG-
  - truck
  - van
  - wheelbarrow
```

## Training

Run a 1-epoch smoke test first:

```bash
python -m ultralytics detect train \
  model=yolo11n.pt \
  data=/mnt/disk/maoqijun/MLLM/trafic_data/data_1.yaml \
  epochs=1 \
  imgsz=640 \
  batch=8 \
  workers=8 \
  device=0 \
  project=/mnt/disk/maoqijun/MLLM/runs/task2 \
  name=debug_yolo11n
```

Train YOLOv8s baseline:

```bash
python -m ultralytics detect train \
  model=yolov8s.pt \
  data=/mnt/disk/maoqijun/MLLM/trafic_data/data_1.yaml \
  epochs=80 \
  imgsz=640 \
  batch=32 \
  workers=8 \
  device=0 \
  patience=20 \
  optimizer=SGD \
  lr0=0.01 \
  cos_lr=True \
  project=/mnt/disk/maoqijun/MLLM/runs/task2 \
  name=yolov8s_e80_img640_b32_lr001
```

Train YOLO11s:

```bash
python -m ultralytics detect train \
  model=yolo11s.pt \
  data=/mnt/disk/maoqijun/MLLM/trafic_data/data_1.yaml \
  epochs=80 \
  imgsz=640 \
  batch=32 \
  workers=8 \
  device=0 \
  patience=20 \
  optimizer=SGD \
  lr0=0.01 \
  cos_lr=True \
  project=/mnt/disk/maoqijun/MLLM/runs/task2 \
  name=yolo11s_e80_img640_b32_lr001
```

If the server cannot access GitHub to download pretrained weights, download the `.pt` file locally and upload it to:

```text
/mnt/disk/maoqijun/MLLM/weights/
```

Then use the local path:

```bash
model=/mnt/disk/maoqijun/MLLM/weights/yolo11s.pt
```

## Validation

Validate the best checkpoint:

```bash
python -m ultralytics detect val \
  model=/mnt/disk/maoqijun/MLLM/runs/task2/yolov8s_e80_img640_b32_lr001/weights/best.pt \
  data=/mnt/disk/maoqijun/MLLM/trafic_data/data_1.yaml \
  imgsz=640 \
  batch=32 \
  device=0 \
  project=/mnt/disk/maoqijun/MLLM/runs/task2 \
  name=val_yolov8s_best
```

Training outputs include:

```text
args.yaml
results.csv
results.png
confusion_matrix.png
PR_curve.png
F1_curve.png
weights/best.pt
weights/last.pt
```

## Test Video Preprocessing

For a 4K 30 FPS source video, crop a 20-second 1080p clip:

```bash
ffmpeg -ss 00:00:05 -t 20 \
  -i input_4k.mp4 \
  -vf "scale=-2:1080" \
  -r 30 \
  -c:v libx264 \
  -preset medium \
  -crf 20 \
  -pix_fmt yuv420p \
  -an \
  /mnt/disk/maoqijun/MLLM/videos/test_main_1080p_20s.mp4
```

## Video Tracking

Run multi-object tracking with BoT-SORT:

```bash
python -u scripts/test_track.py \
  --model /mnt/disk/maoqijun/MLLM/runs/task2/yolov8s_e80_img640_b32_lr001/weights/best.pt \
  --source /mnt/disk/maoqijun/MLLM/videos/test_main_1080p_20s.mp4 \
  --tracker botsort \
  --conf 0.25 \
  --iou 0.5 \
  --imgsz 640 \
  --device 0 \
  --project /mnt/disk/maoqijun/MLLM/runs/task2_track \
  --name test_main_botsort
```

Try ByteTrack for comparison:

```bash
python -u scripts/test_track.py \
  --model /mnt/disk/maoqijun/MLLM/runs/task2/yolov8s_e80_img640_b32_lr001/weights/best.pt \
  --source /mnt/disk/maoqijun/MLLM/videos/test_main_1080p_20s.mp4 \
  --tracker bytetrack \
  --conf 0.25 \
  --device 0 \
  --project /mnt/disk/maoqijun/MLLM/runs/task2_track \
  --name test_main_bytetrack
```

## Line-Crossing Counting

The line-crossing counter uses tracking IDs and box center points. A target is counted once when the same `track_id` moves from one side of the virtual line to the other side.

Example command:

```bash
python -u scripts/test_track.py \
  --model /mnt/disk/maoqijun/MLLM/runs/task2/yolov8s_e80_img640_b32_lr001/weights/best.pt \
  --source /mnt/disk/maoqijun/MLLM/videos/test_2.mp4 \
  --output /mnt/disk/maoqijun/MLLM/runs/task2_count/test_2_botsort_conf025_count.mp4 \
  --tracker botsort \
  --conf 0.25 \
  --iou 0.5 \
  --line-orientation vertical \
  --line-ratio 0.5 \
  --direction negative \
  --device 0
```

Use a horizontal line if vehicles mainly move vertically in the image, and a vertical line if vehicles mainly move horizontally.

## Occlusion and ID Switch Analysis

Open the tracking result video and choose a short segment with occlusion, dense traffic, or vehicle crossing. Extract 3-4 consecutive frames:

```bash
mkdir -p /mnt/disk/maoqijun/MLLM/runs/task2_track/occlusion_frames

ffmpeg -ss 00:00:08 \
  -i /mnt/disk/maoqijun/MLLM/runs/task2_track/test_main_botsort/test_main_1080p_20s.mp4 \
  -vf "select='between(n,0,3)'" \
  -vsync 0 \
  /mnt/disk/maoqijun/MLLM/runs/task2_track/occlusion_frames/frame_%03d.jpg
```

In the report, analyze whether the same vehicle keeps its original tracking ID or gets a new ID after occlusion.

## Download Results from Server

Find output videos:

```bash
find /mnt/disk/maoqijun/MLLM/runs/task2_track -type f \( -name "*.mp4" -o -name "*.avi" \)
find /mnt/disk/maoqijun/MLLM/runs/task2_count -type f \( -name "*.mp4" -o -name "*.avi" \)
```

Download to local machine:

```bash
scp maoqijun@SERVER_HOST:/mnt/disk/maoqijun/MLLM/runs/task2_count/test_main_count.mp4 ~/Downloads/
```

## Model Weights

The trained model weights are not committed to this repository. Download them from:

```text
Weights are stored separately because they are large binary artifacts.
```

Recommended checkpoint for final tracking:

```text
runs/task2/yolov8s_e80_img640_b32_lr001/weights/best.pt
```

## Report

The report source is:

```text
reports/task2_report.tex
```

Compile:

```bash
cd reports
xelatex task2_report.tex
xelatex task2_report.tex
```

The final submission should be a PDF report. The report should include:

- model structure, dataset introduction, and result summary;
- train/validation split, network, batch size, learning rate, optimizer, epochs, loss, and metrics;
- wandb/swanlab screenshots or Ultralytics `results.png` for train/validation loss and mAP curves;
- tracking visualization with bounding boxes, classes, confidence scores, and tracking IDs;
- occlusion/ID-switch analysis with 3-4 consecutive frames;
- line-crossing counting visualization and final count.
