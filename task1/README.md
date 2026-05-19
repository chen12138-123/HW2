# 任务1：Flowers102 微调（ResNet Baseline / 预训练消融 / 注意力模块对比）

本目录覆盖 PDF 中任务1的全部要求：
- 数据集：102 Category Flower Dataset（Flowers102）
- Baseline：ResNet-18/ResNet-34，替换输出层为 102 类
- 微调：使用 ImageNet 预训练初始化；输出层随机初始化；其余参数用更小学习率微调（分组学习率）
- 超参分析：提供批量跑不同 epoch 与学习率组合的脚本并导出结果
- 预训练消融：`--pretrained` vs `--no_pretrained`
- 注意力机制：在 Baseline 上手动加入 SE / CBAM（stage 级特征注意力）并对比 Accuracy
- 可选 wandb / swanlab 记录曲线（用于报告截图）

## 环境

建议 Python 3.10+。

依赖（示例）：

```bash
pip install torch torchvision numpy
pip install wandb   # 可选
pip install swanlab # 可选
```

## 训练单个实验

在本目录下运行：

```bash
python train.py --model resnet18 --pretrained --epochs 30 --lr_head 3e-4 --lr_backbone 3e-5
python train.py --model resnet18 --no_pretrained --epochs 30 --lr_head 3e-4 --lr_backbone 3e-4
python train.py --model resnet18_se --pretrained --epochs 30 --lr_head 3e-4 --lr_backbone 3e-5
python train.py --model resnet18_cbam --pretrained --epochs 30 --lr_head 3e-4 --lr_backbone 3e-5
```

输出目录：
- `runs/<name>_<model>_<pre|scratch>_seed<seed>_img<img>_e<epochs>/`
- `best.pt`：按验证集 top1 Accuracy 选出的最佳权重
- `history.json`：每个 epoch 的 train/val loss 与 accuracy（可直接画曲线/或给 wandb/swanlab）
- `test_metrics.json`：最终 test 集评估

## 超参组合实验（训练步数/epoch + 学习率组合）

```bash
python run_experiments.py --name_prefix exp
```

会生成：
- `runs/exp_summary.json`（每个实验的 best_val_acc 与 run_dir）

## 记录曲线（报告截图）

wandb：

```bash
python train.py --model resnet18 --pretrained --logger wandb --logger_project hw2-task1-flowers --logger_name r18
```

swanlab：

```bash
python train.py --model resnet18 --pretrained --logger swanlab --logger_project hw2-task1-flowers --logger_name r18
```

记录字段（每个 epoch）：
- train_loss / val_loss
- train_acc / val_acc

## 报告/提交对照清单（你需要在报告PDF里给齐）

- 模型结构：ResNet Baseline + 注意力结构（SE/CBAM）说明
- 数据集说明：Flowers102（train/val/test 官方划分）
- 实验设置：batch size、learning rate（head/backbone）、优化器、epoch、评价指标（Accuracy）
- 超参对比：不同 epoch / 学习率组合对验证集 Accuracy 的影响（可用 `exp_summary.json` 汇总）
- 预训练消融：pretrained vs scratch 的对比结论
- 注意力对比：Baseline vs SE vs CBAM 的对比结论
- 可视化截图：训练/验证 loss 曲线 + 验证集 Accuracy 曲线（wandb/swanlab）
- 代码链接：你的 public GitHub repo
- 权重链接：best.pt 上传到网盘的下载地址（报告中给链接）

