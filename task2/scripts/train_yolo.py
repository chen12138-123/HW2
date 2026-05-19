import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 训练脚本参数配置")
    
    # 模型与数据路径
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="预训练模型路径或模型定义文件")
    parser.add_argument("--data", type=str, default="MLLM/data/trafic_data/data.yaml", help="数据集配置文件路径")
    
    # 常用超参数
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="使用设备, 如 0 或 0,1,2,3 或 cpu")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数")
    
    # 优化器相关
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam", "AdamW", "RMSProp"], help="优化器类型")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--cos_lr", action="store_true", default=True, help="是否使用余弦学习率调度")
    parser.add_argument("--patience", type=int, default=20, help="EarlyStopping 等待轮数")
    
    # 保存与日志
    parser.add_argument("--project", type=str, default="", help="项目名称")
    parser.add_argument("--name", type=str, default="yolov8s_e80_img640_b32_lr001", help="本次实验保存的名称")
    parser.add_argument("--verbose", action="store_true", default=True, help="是否打印详细日志")

    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=args.cos_lr,
        patience=args.patience,
        project=args.project,
        name=args.name,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()