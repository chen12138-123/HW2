# 任务3：U-Net 语义分割（从零训练 + Dice Loss 工程）

本目录实现了作业任务3的全部代码要求：
- 不使用任何预训练权重，使用基础 API 手写 U-Net（含编码器/解码器/Skip Connection）
- 使用 Stanford Background Dataset（ICCV09 / iccv09Data）进行像素级训练
- 手动实现 Dice Loss，并支持三种损失配置对比：CE / Dice / CE+Dice
- 在验证集上计算并对比 mIoU，同时记录验证集像素 Accuracy
- 可选接入 wandb 或 swanlab 记录曲线（用于报告截图）

## 环境

建议 Python 3.10+。

安装依赖（示例）：

```bash
pip install torch numpy pillow
pip install wandb   # 可选
pip install swanlab # 可选
```

## 数据集

代码会自动下载并解压 Stanford Background Dataset（iccv09Data.tar.gz）到 `--data_dir` 目录下。

默认下载源：
- http://dags.stanford.edu/data/iccv09Data.tar.gz

数据格式要点（官方说明）：
- `labels/*.regions.txt` 为每像素语义类别整数矩阵，负数表示 unknown

## 训练单个配置

在本目录下运行：

```bash
python train.py --loss ce --epochs 50 --batch_size 8 --img_size 256 --logger none
python train.py --loss dice --epochs 50 --batch_size 8 --img_size 256 --logger none
python train.py --loss ce_dice --epochs 50 --batch_size 8 --img_size 256 --logger none
```

输出目录：
- 默认 `runs/<logger_name>_<loss>_seed<seed>_img<img_size>_bc<base_channels>/`
- `best.pt`：按验证集 mIoU 选出的最佳权重
- `last.pt`：最后一个 epoch 的权重
- `history.json`：每个 epoch 的 train/val 指标（loss、mIoU、acc）

## 一键跑三组实验并汇总

```bash
python run_experiments.py --epochs 50 --batch_size 8 --img_size 256 --logger none --name_prefix exp
```

会生成：
- `runs/exp_summary.json`（包含三种损失配置各自的 best_val_miou 与 run_dir）

## 记录曲线（报告截图）

wandb：

```bash
python train.py --loss ce_dice --logger wandb --logger_project hw2-task3-seg --logger_name unet
```

swanlab：

```bash
python train.py --loss ce_dice --logger swanlab --logger_project hw2-task3-seg --logger_name unet
```

记录字段（每个 epoch）：
- train_loss / val_loss
- train_miou / val_miou
- train_acc / val_acc

