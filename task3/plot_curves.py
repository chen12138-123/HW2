from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    return p.parse_args()


def _ema(xs: list[float], alpha: float = 0.2) -> list[float]:
    if not xs:
        return []
    alpha = float(alpha)
    alpha = min(max(alpha, 0.0), 1.0)
    out = [float(xs[0])]
    for x in xs[1:]:
        out.append(alpha * float(x) + (1.0 - alpha) * out[-1])
    return out


def _best_idx(xs: list[float], mode: str) -> int:
    if not xs:
        return 0
    if mode == "min":
        return min(range(len(xs)), key=lambda i: float(xs[i]))
    return max(range(len(xs)), key=lambda i: float(xs[i]))


def _plot_metric(
    *,
    run_name: str,
    epochs: list[int],
    train: list[float],
    val: list[float],
    ylabel: str,
    out_path: Path,
    best_mode: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, train, label="train (raw)", linewidth=1.0, alpha=0.35)
    ax.plot(epochs, val, label="val (raw)", linewidth=1.0, alpha=0.35)

    train_s = _ema(train, alpha=0.22)
    val_s = _ema(val, alpha=0.22)
    ax.plot(epochs, train_s, label="train (EMA)", linewidth=2.2)
    ax.plot(epochs, val_s, label="val (EMA)", linewidth=2.2)

    bi = _best_idx(val, best_mode)
    be = epochs[bi] if epochs else 0
    bv = float(val[bi]) if val else float("nan")
    ax.scatter([be], [bv], s=55, zorder=5, label=f"best val @e{be}")
    ax.axvline(be, linestyle="--", linewidth=1.2, alpha=0.6)

    dy = 0.015 * (max(val) - min(val) + 1e-8) if val else 0.0
    ax.text(be, bv + dy, f"e{be}\n{bv:.4f}", ha="left", va="bottom", fontsize=9)

    ax.set_title(f"{run_name} | {ylabel}", fontsize=12, pad=10)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.minorticks_on()
    ax.grid(True, which="major", linewidth=0.8, alpha=0.55)
    ax.grid(True, which="minor", linewidth=0.5, alpha=0.25)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


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
    train_miou = [h["train_miou"] for h in hist]
    val_miou = [h["val_miou"] for h in hist]
    train_acc = [h["train_acc"] for h in hist]
    val_acc = [h["val_acc"] for h in hist]

    run_name = run_dir.name
    _plot_metric(
        run_name=run_name,
        epochs=epochs,
        train=train_loss,
        val=val_loss,
        ylabel="loss",
        out_path=run_dir / "loss_curve.png",
        best_mode="min",
    )
    _plot_metric(
        run_name=run_name,
        epochs=epochs,
        train=train_miou,
        val=val_miou,
        ylabel="mIoU",
        out_path=run_dir / "miou_curve.png",
        best_mode="max",
    )
    _plot_metric(
        run_name=run_name,
        epochs=epochs,
        train=train_acc,
        val=val_acc,
        ylabel="pixel_acc",
        out_path=run_dir / "acc_curve.png",
        best_mode="max",
    )


if __name__ == "__main__":
    main()
