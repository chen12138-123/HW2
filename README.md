# HW2: Deep Learning and Spatial Intelligence (Task 1 / Task 2 / Task 3)

This repository contains the implementations and reproducible experiments for three homework tasks:

- Task 1: Flowers102 image classification (ResNet fine-tuning, pretrained vs. scratch ablation, SE/CBAM attention).
- Task 2: Road vehicle detection + multi-object tracking + analysis + line-crossing counting (Ultralytics/YOLO).
- Task 3: Semantic segmentation on Stanford Background Dataset (U-Net from scratch + Dice loss engineering).

## Project Structure

```text
HW2/
  task1/
  task2/
  task3/
```

## Task 1: Flowers102 Classification (Fine-tuning ResNet)

### Environment

Recommended:

```text
Python: 3.10+
Framework: PyTorch + torchvision
```

Install:

```bash
pip install torch torchvision numpy pillow matplotlib
```

Optional logging (for training curves):

```bash
pip install wandb
pip install swanlab
```

### Dataset

The code uses `torchvision.datasets.Flowers102` and downloads the dataset automatically under `task1/data/` on first run.

### Training

Run in `task1/`:

```bash
cd task1

# Pretrained fine-tuning (baseline)
python train.py --model resnet34 --pretrained --epochs 40 --lr_head 3e-4 --lr_backbone 3e-5

# Scratch ablation (no pretrained)
python train.py --model resnet34 --no_pretrained --epochs 10 --lr_head 3e-4 --lr_backbone 3e-4

# Attention variant (CBAM)
python train.py --model resnet34_cbam --pretrained --epochs 40 --lr_head 3e-4 --lr_backbone 3e-5
```

### Hyper-parameter Sweeps

Run a batch of experiments and export a summary JSON:

```bash
python run_experiments.py --name_prefix exp
```

### Curves / Qualitative Visualization

Plot curves from `history.json`:

```bash
python plot_curves.py --run_dir "c:\Desktop\HW2\task1\runs\<your_run_dir>"
```

Generate qualitative predictions (GT vs Pred):

```bash
python visualize_samples.py --weights "c:\Desktop\HW2\task1\runs\<your_run_dir>\best.pt" --out "c:\Desktop\HW2\task1\runs\<your_run_dir>\qualitative.png" --device cuda
```

### Outputs

Each run is saved under:

```text
task1/runs/<logger_name>_<model>_<pre|scratch>_seed<seed>_img<image_size>_e<epochs>/
```

Files:

```text
config.json
history.json
loss_curve.png
acc_curve.png
qualitative.png
test_metrics.json
best.pt / last.pt (if you keep weights locally)
```

## Task 2: Vehicle Detection + Tracking (Ultralytics/YOLO)

This section provides a complete overview for GitHub (dataset / training / tracking / report).

### Project Structure

```text
task2/
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

### Environment

Recommended server environment:

```text
OS: Ubuntu 20.04/22.04
GPU: NVIDIA RTX 4090
Python: 3.10
CUDA: 12.1
Framework: PyTorch + Ultralytics
```

Create a conda environment:

```bash
conda create -n mllm-yolo python=3.10 -y
conda activate mllm-yolo
```

Install PyTorch (CUDA):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install Ultralytics (and common utilities):

```bash
pip install ultralytics opencv-python numpy pandas matplotlib
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

### Dataset

The detector is trained on Road Vehicle Images Dataset.

Example download (KaggleHub):

```python
import kagglehub

path = kagglehub.dataset_download(
    "ashfakyeafi/road-vehicle-images-dataset",
    output_dir="/path/to/trafic_data",
)
print(path)
```

The expected dataset structure:

```text
trafic_data/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  data_1.yaml
```

### Training / Validation

Run a smoke test first:

```bash
python -m ultralytics detect train \
  model=yolo11n.pt \
  data=/path/to/trafic_data/data_1.yaml \
  epochs=1 \
  imgsz=640 \
  batch=8 \
  workers=8 \
  device=0
```

Train a baseline model (example):

```bash
python -m ultralytics detect train \
  model=yolov8s.pt \
  data=/path/to/trafic_data/data_1.yaml \
  epochs=80 \
  imgsz=640 \
  batch=32 \
  workers=8 \
  device=0 \
  patience=20 \
  optimizer=SGD \
  lr0=0.01 \
  cos_lr=True
```

Validate:

```bash
python -m ultralytics detect val \
  model=/path/to/runs/.../weights/best.pt \
  data=/path/to/trafic_data/data_1.yaml \
  imgsz=640 \
  batch=32 \
  device=0
```

Training outputs typically include:

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

### Test Video Preprocessing

Example: crop a 20-second 1080p clip from a 4K 30 FPS source video:

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
  output_1080p_20s.mp4
```

### Tracking / Counting / Analysis

Run multi-object tracking with BoT-SORT:

```bash
python -u scripts/test_track.py \
  --model /path/to/runs/.../weights/best.pt \
  --source /path/to/test_video.mp4 \
  --tracker botsort \
  --conf 0.25 \
  --iou 0.5 \
  --imgsz 640 \
  --device 0 \
  --project /path/to/task2_track_runs \
  --name test_main_botsort
```

Try ByteTrack for comparison:

```bash
python -u scripts/test_track.py \
  --model /path/to/runs/.../weights/best.pt \
  --source /path/to/test_video.mp4 \
  --tracker bytetrack \
  --conf 0.25 \
  --device 0 \
  --project /path/to/task2_track_runs \
  --name test_main_bytetrack
```

Line-crossing counting example:

```bash
python -u scripts/test_track.py \
  --model /path/to/runs/.../weights/best.pt \
  --source /path/to/test_video.mp4 \
  --output /path/to/out_count.mp4 \
  --tracker botsort \
  --conf 0.25 \
  --iou 0.5 \
  --line-orientation vertical \
  --line-ratio 0.5 \
  --direction negative \
  --device 0
```

Occlusion / ID-switch analysis: choose a short segment in the tracking result video, extract 3–4 consecutive frames, and analyze whether the same vehicle keeps its original ID after occlusion.

### Download Results

Find output videos:

```bash
find /path/to/task2_track_runs -type f \( -name "*.mp4" -o -name "*.avi" \)
```

Download to local machine (example):

```bash
scp USER@SERVER_HOST:/path/to/output.mp4 ~/Downloads/
```

### Model Weights

Model weights are typically not committed to GitHub due to file size. Upload `best.pt` to separate storage and share a download link if required by the submission.

### Report

The report source is:

```text
task2/reports/task2_report.tex
```

Compile:

```bash
cd task2/reports
xelatex task2_report.tex
xelatex task2_report.tex
```
 
The final submission report should include:

- model structure, dataset introduction, and result summary;
- train/validation split, network, batch size, learning rate, optimizer, epochs, loss, and metrics;
- learning curves and detection visualizations (e.g., PR curves / confusion matrix / result images);
- tracking visualization with bounding boxes, classes, confidence scores, and tracking IDs;
- occlusion/ID-switch analysis with 3–4 consecutive frames;
- line-crossing counting visualization and the final counts.

## Task 3: Semantic Segmentation (U-Net from Scratch + Dice Loss)

### Environment

Recommended:

```text
Python: 3.10+
Framework: PyTorch
```

Install:

```bash
pip install torch torchvision numpy pillow matplotlib
```

Optional logging (for training curves):

```bash
pip install wandb
pip install swanlab
```

### Dataset

The code automatically downloads and extracts Stanford Background Dataset:

```text
http://dags.stanford.edu/data/iccv09Data.tar.gz
```

The labels are stored as `labels/*.regions.txt` (integer matrix per pixel). Negative values are treated as unknown and ignored in metrics/loss.

### Outputs

Each run is stored under:

```text
task3/runs/<logger_name>_<loss>_seed<seed>_img<img_size>_bc<base_channels>/
```

Typical files:

```text
config.json
history.json
best_metrics.json
loss_curve.png
miou_curve.png
acc_curve.png
qualitative.png (if generated)
best.pt / last.pt (if you keep weights locally)
```

### Training

Run in `task3/`:

```bash
cd task3

# CE / Dice / CE+Dice comparisons
python train.py --loss ce --epochs 50 --batch_size 8 --img_size 256 --device cuda --logger none
python train.py --loss dice --epochs 50 --batch_size 8 --img_size 256 --device cuda --logger none
python train.py --loss ce_dice --epochs 50 --batch_size 8 --img_size 256 --device cuda --logger none
```

### Run All 3 Loss Configurations

```bash
python run_experiments.py --epochs 50 --batch_size 8 --img_size 256 --logger none --name_prefix exp
```

### Logging Curves (wandb / swanlab)

wandb:

```bash
python train.py --loss ce_dice --logger wandb --logger_project hw2-task3-seg --logger_name unet
```

swanlab:

```bash
python train.py --loss ce_dice --logger swanlab --logger_project hw2-task3-seg --logger_name unet
```

### Resume Training

If training is interrupted, resume from a previous run directory:

```bash
python train.py --resume "c:\Desktop\HW2\task3\runs\<your_run_dir>" --device cuda --epochs 400
```

### Curves / Qualitative Visualization

Plot curves (raw + EMA smoothing + best point annotation):

```bash
python plot_curves.py --run_dir "c:\Desktop\HW2\task3\runs\<your_run_dir>"
```

Generate qualitative visualization:

```bash
python visualize_samples.py --weights "c:\Desktop\HW2\task3\runs\<your_run_dir>\best.pt" --out "c:\Desktop\HW2\task3\runs\<your_run_dir>\qualitative.png" --img_size 256 --device cuda
```

## Notes on Weights

Large weight files (e.g., `best.pt`, `last.pt`) are typically not committed to GitHub due to size. Upload them to a separate storage (Google Drive/Baidu Pan/etc.) and include the download links in your report submission if required.
