from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def run_one(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    p.add_argument("--data_dir", type=str, default=str(root / "data"))
    p.add_argument("--out_dir", type=str, default=str(root / "runs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--logger", type=str, default="none", choices=["none", "wandb", "swanlab"])
    p.add_argument("--logger_project", type=str, default="hw2-task1-flowers")
    p.add_argument("--name_prefix", type=str, default="exp")
    return p.parse_args()


def read_metric(run_dir: Path, key: str) -> float:
    p = run_dir / "best_metrics.json"
    if not p.exists():
        return float("nan")
    obj = json.loads(p.read_text(encoding="utf-8"))
    return float(obj.get(key, float("nan")))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    train_py = root / "train.py"

    amp_flags = []
    if args.amp:
        amp_flags = ["--amp"]
    if args.no_amp:
        amp_flags = ["--no_amp"]

    models = ["resnet18", "resnet18_se", "resnet18_cbam"]
    pretrained_opts = [True, False]

    lr_pairs = [
        (3e-4, 3e-5),
        (1e-3, 1e-4),
        (3e-4, 1e-4),
    ]
    epochs_list = [15, 30]

    summary = []
    for model, pretrained, (lr_head, lr_backbone), epochs in itertools.product(models, pretrained_opts, lr_pairs, epochs_list):
        name = args.name_prefix
        cmd = [
            sys.executable,
            str(train_py),
            "--data_dir",
            args.data_dir,
            "--out_dir",
            args.out_dir,
            "--seed",
            str(args.seed),
            "--model",
            model,
            "--image_size",
            str(args.image_size),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--epochs",
            str(epochs),
            "--lr_head",
            str(lr_head),
            "--lr_backbone",
            str(lr_backbone),
            "--device",
            args.device,
            "--logger",
            args.logger,
            "--logger_project",
            args.logger_project,
            "--logger_name",
            name,
            *amp_flags,
        ]
        cmd += ["--pretrained"] if pretrained else ["--no_pretrained"]
        run_one(cmd)

        run_dir = (
            Path(args.out_dir)
            / f"{name}_{model}_{'pre' if pretrained else 'scratch'}_seed{args.seed}_img{args.image_size}_e{epochs}"
        )
        best_val_acc = read_metric(run_dir, "best_val_acc")
        summary.append(
            {
                "run_dir": str(run_dir),
                "model": model,
                "pretrained": pretrained,
                "epochs": epochs,
                "lr_head": lr_head,
                "lr_backbone": lr_backbone,
                "best_val_acc": best_val_acc,
            }
        )

    out_path = Path(args.out_dir) / f"{args.name_prefix}_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

