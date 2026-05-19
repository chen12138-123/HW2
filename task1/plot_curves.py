from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    history_path = run_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"history.json not found: {history_path}")

    hist = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [h["epoch"] for h in hist]
    train_loss = [h["train_loss"] for h in hist]
    val_loss = [h["val_loss"] for h in hist]
    train_acc = [h["train_acc"] for h in hist]
    val_acc = [h["val_acc"] for h in hist]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("top1_acc")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "acc_curve.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()

