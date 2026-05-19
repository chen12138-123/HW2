from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from contextlib import nullcontext
from torch.utils.data import DataLoader

from src.datasets.stanford_background import (
    CLASS_NAMES,
    ensure_stanford_background_dataset,
    load_or_create_split,
    StanfordBackgroundDataset,
)
from src.losses.dice import DiceLossConfig
from src.metrics.segmentation import (
    AverageMeter,
    confusion_matrix_from_logits,
    metrics_from_confusion,
)
from src.models.unet import UNetConfig
from src.utils import ensure_dir, get_device, init_logger, save_json, set_seed


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str
    out_dir: str
    resume: Optional[str] = None
    seed: int = 42
    img_size: int = 256
    num_classes: int = 8
    base_channels: int = 64
    norm: Literal["bn", "gn"] = "bn"
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 50
    augment: bool = True
    lr: float = 3e-4
    weight_decay: float = 1e-4
    onecycle: bool = True
    max_lr: Optional[float] = None
    grad_clip_norm: float = 0.0
    amp: bool = True
    device: str = "auto"
    loss: Literal["ce", "dice", "ce_dice"] = "ce"
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    ce_class_weight_mode: Literal["none", "freq", "inv_freq"] = "none"
    ignore_index: int = 255
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    logger: str = "none"
    logger_project: str = "hw2-task3-seg"
    logger_name: str = "run"
    save_best_on: Literal["miou", "acc", "loss"] = "miou"
    stop_acc: Optional[float] = None


def _autocast_ctx(enabled: bool, device: torch.device):
    if not enabled or device.type != "cuda":
        return nullcontext()
    try:
        from torch.amp import autocast as autocast_v2

        return autocast_v2(device_type="cuda")
    except Exception:
        from torch.cuda.amp import autocast as autocast_v1

        return autocast_v1()


def _grad_scaler(enabled: bool, device: torch.device):
    if not enabled or device.type != "cuda":
        return None
    try:
        from torch.amp import GradScaler as GradScalerV2

        return GradScalerV2("cuda", enabled=True)
    except Exception:
        from torch.cuda.amp import GradScaler as GradScalerV1

        return GradScalerV1(enabled=True)


def build_loss(cfg: "TrainConfig", ce_weights: Optional[torch.Tensor]) -> nn.Module:
    ce = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index, weight=ce_weights)
    dice = DiceLossConfig(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index).build()

    class _Loss(nn.Module):
        def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if cfg.loss == "ce":
                return cfg.ce_weight * ce(logits, target)
            if cfg.loss == "dice":
                return cfg.dice_weight * dice(logits, target)
            if cfg.loss == "ce_dice":
                return cfg.ce_weight * ce(logits, target) + cfg.dice_weight * dice(logits, target)
            raise ValueError(f"Unknown loss: {cfg.loss}")

    return _Loss()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scaler,
    scheduler,
    device: torch.device,
    amp: bool,
    num_classes: int,
    ignore_index: int,
    grad_clip_norm: float,
) -> dict:
    model.train()
    loss_meter = AverageMeter()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if amp and scaler is not None:
            with _autocast_ctx(True, device):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), n=x.shape[0])
        cm += confusion_matrix_from_logits(logits.detach(), y, num_classes=num_classes, ignore_index=ignore_index)

    m = metrics_from_confusion(cm)
    return {"loss": loss_meter.avg, "miou": m["miou"], "acc": m["acc"]}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp: bool,
    num_classes: int,
    ignore_index: int,
) -> dict:
    model.eval()
    loss_meter = AverageMeter()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        with _autocast_ctx(amp, device):
            logits = model(x)
            loss = loss_fn(logits, y)

        loss_meter.update(loss.item(), n=x.shape[0])
        cm += confusion_matrix_from_logits(logits, y, num_classes=num_classes, ignore_index=ignore_index)

    m = metrics_from_confusion(cm)
    return {"loss": loss_meter.avg, "miou": m["miou"], "acc": m["acc"], "iou_per_class": m["iou_per_class"]}


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parent / "data"))
    p.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "runs"))
    p.add_argument("--resume", type=str, default=None, help="Resume training from a run_dir (preferred) or a checkpoint .pt file.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--augment", action="store_true", help="Enable train-time color jitter + random resized crop.")
    p.add_argument("--no_aug", action="store_true", help="Disable train-time color jitter + random resized crop (resize only).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--onecycle", action="store_true")
    p.add_argument("--no_onecycle", action="store_true")
    p.add_argument("--max_lr", type=float, default=None)
    p.add_argument("--grad_clip_norm", type=float, default=0.0)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--norm", type=str, default="bn", choices=["bn", "gn"])
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "dice", "ce_dice"])
    p.add_argument("--ce_weight", type=float, default=1.0)
    p.add_argument("--dice_weight", type=float, default=1.0)
    p.add_argument("--ce_class_weight_mode", type=str, default="none", choices=["none", "freq", "inv_freq"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.0)
    p.add_argument("--logger", type=str, default="none", choices=["none", "wandb", "swanlab"])
    p.add_argument("--logger_project", type=str, default="hw2-task3-seg")
    p.add_argument("--logger_name", type=str, default="run")
    p.add_argument("--save_best_on", type=str, default="miou", choices=["miou", "acc", "loss"])
    p.add_argument("--stop_acc", type=float, default=None, help="Early stop if val pixel accuracy >= this value.")
    args = p.parse_args()

    amp = True
    if args.amp:
        amp = True
    if args.no_amp:
        amp = False

    onecycle = True
    if args.onecycle:
        onecycle = True
    if args.no_onecycle:
        onecycle = False

    augment = True
    if args.augment:
        augment = True
    if args.no_aug:
        augment = False

    return TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        resume=args.resume,
        seed=args.seed,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        augment=augment,
        lr=args.lr,
        weight_decay=args.weight_decay,
        onecycle=onecycle,
        max_lr=args.max_lr,
        grad_clip_norm=args.grad_clip_norm,
        base_channels=args.base_channels,
        norm=args.norm,
        loss=args.loss,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        ce_class_weight_mode=args.ce_class_weight_mode,
        amp=amp,
        device=args.device,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        logger=args.logger,
        logger_project=args.logger_project,
        logger_name=args.logger_name,
        save_best_on=args.save_best_on,
        stop_acc=args.stop_acc,
    )


def _cfg_from_dict(d: dict) -> TrainConfig:
    allowed = {f.name for f in fields(TrainConfig)}
    kwargs = {k: v for k, v in (d or {}).items() if k in allowed}
    if "data_dir" not in kwargs or "out_dir" not in kwargs:
        raise ValueError("Checkpoint config missing required keys: data_dir/out_dir")
    return TrainConfig(**kwargs)


def _resolve_resume_paths(resume: str) -> tuple[Path, Path]:
    p = Path(resume)
    if p.is_dir():
        run_dir = p
        ckpt_path = run_dir / "last.pt"
        if not ckpt_path.exists():
            ckpt_path = run_dir / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No last.pt/best.pt found under: {run_dir}")
        return run_dir, ckpt_path
    if not p.exists():
        raise FileNotFoundError(f"Resume path not found: {p}")
    return p.parent, p


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    resume_ckpt = None
    start_epoch = 1

    if cfg.resume:
        run_dir, ckpt_path = _resolve_resume_paths(cfg.resume)
        resume_ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(resume_ckpt, dict) and "config" in resume_ckpt:
            loaded = _cfg_from_dict(resume_ckpt["config"])
            loaded = replace(
                loaded,
                device=cfg.device,
                out_dir=str(run_dir.parent),
                epochs=max(int(loaded.epochs), int(cfg.epochs)),
                stop_acc=cfg.stop_acc if cfg.stop_acc is not None else loaded.stop_acc,
                save_best_on=cfg.save_best_on if cfg.save_best_on is not None else loaded.save_best_on,
            )
            cfg = loaded
        if isinstance(resume_ckpt, dict) and "epoch" in resume_ckpt:
            start_epoch = int(resume_ckpt["epoch"]) + 1
        if not (run_dir / "config.json").exists():
            save_json(run_dir / "config.json", asdict(cfg))
    else:
        out_root = ensure_dir(cfg.out_dir)
        run_dir = ensure_dir(out_root / f"{cfg.logger_name}_{cfg.loss}_seed{cfg.seed}_img{cfg.img_size}_bc{cfg.base_channels}")
        save_json(run_dir / "config.json", asdict(cfg))

    dataset_root = ensure_stanford_background_dataset(cfg.data_dir)
    split = load_or_create_split(dataset_root, seed=cfg.seed, val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio)

    train_ds = StanfordBackgroundDataset(
        dataset_root=dataset_root,
        split="train",
        stems=split.train,
        image_size=(cfg.img_size, cfg.img_size),
        random_horizontal_flip=True,
        augment=cfg.augment,
        ignore_index=cfg.ignore_index,
    )
    val_ds = StanfordBackgroundDataset(
        dataset_root=dataset_root,
        split="val",
        stems=split.val,
        image_size=(cfg.img_size, cfg.img_size),
        random_horizontal_flip=False,
        ignore_index=cfg.ignore_index,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = UNetConfig(in_channels=3, num_classes=cfg.num_classes, base_channels=cfg.base_channels, norm=cfg.norm).build().to(device)
    if isinstance(resume_ckpt, dict) and "model_state" in resume_ckpt:
        model.load_state_dict(resume_ckpt["model_state"])
    ce_weights = None
    if cfg.ce_class_weight_mode != "none" and cfg.loss in ("ce", "ce_dice"):
        counts = np.zeros((cfg.num_classes,), dtype=np.int64)
        for stem in split.train:
            p = dataset_root / "labels" / f"{stem}.regions.txt"
            arr = np.loadtxt(p, dtype=np.int64)
            valid = arr[arr >= 0]
            if valid.size == 0:
                continue
            mn = int(valid.min())
            if mn == 1 and int(valid.max()) == cfg.num_classes:
                valid = valid - 1
            valid = valid[(valid >= 0) & (valid < cfg.num_classes)]
            if valid.size == 0:
                continue
            binc = np.bincount(valid.reshape(-1), minlength=cfg.num_classes)
            counts += binc.astype(np.int64)

        counts = np.maximum(counts, 1)
        if cfg.ce_class_weight_mode == "freq":
            w = counts.astype(np.float64)
        else:
            w = 1.0 / counts.astype(np.float64)
        w = w / (w.mean() + 1e-12)
        ce_weights = torch.tensor(w, dtype=torch.float32, device=device)

    loss_fn = build_loss(cfg, ce_weights=ce_weights).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if isinstance(resume_ckpt, dict) and "optimizer_state" in resume_ckpt:
        try:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        except Exception:
            pass
    scaler = _grad_scaler(cfg.amp, device)
    if scaler is not None and isinstance(resume_ckpt, dict) and "scaler_state" in resume_ckpt and resume_ckpt["scaler_state"] is not None:
        try:
            scaler.load_state_dict(resume_ckpt["scaler_state"])
        except Exception:
            pass
    scheduler = None
    if cfg.onecycle:
        steps_per_epoch = len(train_loader)
        max_lr = cfg.max_lr if cfg.max_lr is not None else cfg.lr * 10
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=1e4,
            anneal_strategy="cos",
        )
        if isinstance(resume_ckpt, dict) and "scheduler_state" in resume_ckpt and resume_ckpt["scheduler_state"] is not None:
            try:
                scheduler.load_state_dict(resume_ckpt["scheduler_state"])
            except Exception:
                pass

    logger = init_logger(
        backend=cfg.logger,
        project=cfg.logger_project,
        name=f"{cfg.logger_name}_{cfg.loss}",
        config=asdict(cfg),
        out_dir=run_dir,
    )

    best_score = -math.inf
    best_metrics_path = run_dir / "best_metrics.json"
    if isinstance(resume_ckpt, dict) and "best_score" in resume_ckpt:
        try:
            best_score = float(resume_ckpt["best_score"])
        except Exception:
            best_score = -math.inf
    elif best_metrics_path.exists():
        try:
            import json as _json

            bm = _json.loads(best_metrics_path.read_text(encoding="utf-8"))
            best_score = float(bm.get("best_score", best_score))
        except Exception:
            pass

    history = []
    hist_path = run_dir / "history.json"
    if hist_path.exists():
        try:
            import json as _json

            history = _json.loads(hist_path.read_text(encoding="utf-8"))
        except Exception:
            history = []

    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
            amp=cfg.amp,
            num_classes=cfg.num_classes,
            ignore_index=cfg.ignore_index,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        va = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            amp=cfg.amp,
            num_classes=cfg.num_classes,
            ignore_index=cfg.ignore_index,
        )

        record = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_miou": tr["miou"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_miou": va["miou"],
            "val_acc": va["acc"],
        }
        history.append(record)
        save_json(run_dir / "history.json", history)

        logger.log(record, step=epoch)
        print(
            f"epoch {epoch:03d} | "
            f"train loss {tr['loss']:.4f} miou {tr['miou']:.4f} acc {tr['acc']:.4f} | "
            f"val loss {va['loss']:.4f} miou {va['miou']:.4f} acc {va['acc']:.4f}"
        )

        if cfg.save_best_on == "miou":
            score = va["miou"]
        elif cfg.save_best_on == "acc":
            score = va["acc"]
        else:
            score = -va["loss"]
        if score > best_score:
            best_score = score
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_score": best_score,
                "config": asdict(cfg),
                "class_names": CLASS_NAMES,
            }
            torch.save(ckpt, run_dir / "best.pt")
            save_json(
                run_dir / "best_metrics.json",
                {"epoch": epoch, "best_score": best_score, "val": va},
            )

        last_ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_score": best_score,
            "config": asdict(cfg),
            "class_names": CLASS_NAMES,
        }
        torch.save(last_ckpt, run_dir / "last.pt")

        if cfg.stop_acc is not None and float(va["acc"]) >= float(cfg.stop_acc):
            print(f"early stop: val_acc {va['acc']:.4f} >= {cfg.stop_acc:.4f}")
            break

    logger.finish()


if __name__ == "__main__":
    main()
