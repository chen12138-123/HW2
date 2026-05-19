# HW2：深度学习与空间智能（Task1 / Task2 / Task3）

本仓库包含课程作业的三个任务实现与复现实验脚本，并提供自动生成的实验报告 PDF。

## 目录结构

```text
HW2/
  task1/   # Flowers102 分类：ResNet 微调 / 预训练消融 / 注意力模块对比
  task2/   # 道路车辆检测 + 多目标跟踪（Ultralytics/YOLO），含 LaTeX 报告
  task3/   # Stanford Background 语义分割：从零实现 U-Net + Dice Loss
  Task1_Report*.pdf
  Task3_Report*.pdf
```

## 环境准备（Task1/Task3）

建议 Python 3.10+。

```bash
pip install torch torchvision numpy pillow matplotlib reportlab
```

可选（用于记录训练曲线）：

```bash
pip install wandb
pip install swanlab
```

## Task 1：Flowers102 分类（ResNet 微调）

进入目录：

```bash
cd task1
```

训练示例（微调 / 从零训练消融 / 注意力对比）：

```bash
python train.py --model resnet34 --pretrained --epochs 40 --lr_head 3e-4 --lr_backbone 3e-5
python train.py --model resnet34 --no_pretrained --epochs 10 --lr_head 3e-4 --lr_backbone 3e-4
python train.py --model resnet34_cbam --pretrained --epochs 40 --lr_head 3e-4 --lr_backbone 3e-5
```

输出在 `task1/runs/<run_name>/`，核心文件：
- `best.pt`：按验证集 Accuracy 选取的最优权重
- `history.json`：每个 epoch 的 train/val 指标
- `loss_curve.png` / `acc_curve.png`：训练曲线
- `qualitative.png`：测试集定性可视化

更详细说明见 [task1/README.md](file:///c:/Desktop/HW2/task1/README.md)。

## Task 2：检测 + 跟踪（Ultralytics/YOLO）

Task2 的训练/评测与报告位于 `task2/`，其依赖与运行方式与 Task1/3 不同（需要 Ultralytics、数据集下载等）。

请直接阅读 [task2/README.md](file:///c:/Desktop/HW2/task2/README.md) 获取完整复现步骤，以及 `task2/reports/task2_report.tex` 的编译说明。

## Task 3：语义分割（U-Net from scratch + Dice Loss）

进入目录：

```bash
cd task3
```

训练三组对比（CE / Dice / CE+Dice）：

```bash
python train.py --loss ce --epochs 50 --batch_size 8 --img_size 256 --logger none
python train.py --loss dice --epochs 50 --batch_size 8 --img_size 256 --logger none
python train.py --loss ce_dice --epochs 50 --batch_size 8 --img_size 256 --logger none
```

数据集会自动下载并解压到 `task3/data/`（iccv09Data.tar.gz）。

恢复训练（断电/关闭终端后继续）：

```bash
python train.py --resume "c:\Desktop\HW2\task3\runs\<your_run_dir>" --device cuda --epochs 400 --save_best_on acc
```

绘制曲线（更“复杂”的版本：raw + EMA + best 点标注）：

```bash
python plot_curves.py --run_dir "c:\Desktop\HW2\task3\runs\<your_run_dir>"
```

生成定性可视化：

```bash
python visualize_samples.py --weights "c:\Desktop\HW2\task3\runs\<your_run_dir>\best.pt" --out "c:\Desktop\HW2\task3\runs\<your_run_dir>\qualitative.png" --img_size 256 --device cuda
```

更详细说明见 [task3/README.md](file:///c:/Desktop/HW2/task3/README.md)。
