from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_one(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parent / "data"))
    p.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "runs"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--logger", type=str, default="none", choices=["none", "wandb", "swanlab"])
    p.add_argument("--logger_project", type=str, default="hw2-task3-seg")
    p.add_argument("--name_prefix", type=str, default="exp")
    return p.parse_args()


def best_miou(run_dir: Path) -> float:
    p = run_dir / "best_metrics.json"
    if not p.exists():
        return float("nan")
    obj = json.loads(p.read_text(encoding="utf-8"))
    return float(obj.get("val", {}).get("miou", float("nan")))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    train_py = root / "train.py"

    losses = ["ce", "dice", "ce_dice"]
    run_dirs: dict[str, str] = {}
    for loss in losses:
        name = f"{args.name_prefix}"
        cmd = [
            sys.executable,
            str(train_py),
            "--data_dir",
            args.data_dir,
            "--out_dir",
            args.out_dir,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--img_size",
            str(args.img_size),
            "--base_channels",
            str(args.base_channels),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--loss",
            loss,
            "--logger",
            args.logger,
            "--logger_project",
            args.logger_project,
            "--logger_name",
            name,
        ]
        run_one(cmd)
        run_dir = (
            Path(args.out_dir)
            / f"{name}_{loss}_seed{args.seed}_img{args.img_size}_bc{args.base_channels}"
        )
        run_dirs[loss] = str(run_dir)

    summary = {loss: {"run_dir": run_dirs[loss], "best_val_miou": best_miou(Path(run_dirs[loss]))} for loss in losses}
    summary_path = Path(args.out_dir) / f"{args.name_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

