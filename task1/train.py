from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.flowers102 import Flowers102Config, build_dataloaders
from src.metrics import AverageMeter, top1_accuracy
from src.models.resnet_models import ModelConfig
from src.utils import ensure_dir, get_device, init_logger, save_json, set_seed


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


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str
    out_dir: str
    seed: int = 42

    model: str = "resnet18"
    pretrained: bool = True
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4

    epochs: int = 30
    lr_head: float = 3e-4
    lr_backbone: float = 3e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    onecycle: bool = True
    max_lr_head: Optional[float] = None
    max_lr_backbone: Optional[float] = None
    grad_clip_norm: float = 0.0

    amp: bool = True
    device: str = "auto"

    logger: str = "none"
    logger_project: str = "hw2-task1-flowers"
    logger_name: str = "run"


def parse_args() -> TrainConfig:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=str(root / "data"))
    p.add_argument("--out_dir", type=str, default=str(root / "runs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet18_se", "resnet34_se", "resnet18_cbam", "resnet34_cbam"],
    )
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr_head", type=float, default=3e-4)
    p.add_argument("--lr_backbone", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--onecycle", action="store_true")
    p.add_argument("--no_onecycle", action="store_true")
    p.add_argument("--max_lr_head", type=float, default=None)
    p.add_argument("--max_lr_backbone", type=float, default=None)
    p.add_argument("--grad_clip_norm", type=float, default=0.0)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--logger", type=str, default="none", choices=["none", "wandb", "swanlab"])
    p.add_argument("--logger_project", type=str, default="hw2-task1-flowers")
    p.add_argument("--logger_name", type=str, default="run")
    args = p.parse_args()

    pretrained = True
    if args.pretrained:
        pretrained = True
    if args.no_pretrained:
        pretrained = False

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

    return TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        model=args.model,
        pretrained=pretrained,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        onecycle=onecycle,
        max_lr_head=args.max_lr_head,
        max_lr_backbone=args.max_lr_backbone,
        grad_clip_norm=args.grad_clip_norm,
        amp=amp,
        device=args.device,
        logger=args.logger,
        logger_project=args.logger_project,
        logger_name=args.logger_name,
    )


def _step(
    model: nn.Module,
    batch,
    loss_fn: nn.Module,
    device: torch.device,
    amp: bool,
):
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    with _autocast_ctx(amp, device):
        logits = model(x)
        loss = loss_fn(logits, y)
    return logits, loss, y, x.shape[0]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scaler,
    scheduler,
    device: torch.device,
    amp: bool,
    grad_clip_norm: float,
) -> dict:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits, loss, y, bs = _step(model, batch, loss_fn, device, amp)
        if amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), n=bs)
        acc_meter.update(top1_accuracy(logits.detach(), y), n=bs)

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp: bool,
) -> dict:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for batch in loader:
        logits, loss, y, bs = _step(model, batch, loss_fn, device, amp)
        loss_meter.update(loss.item(), n=bs)
        acc_meter.update(top1_accuracy(logits, y), n=bs)
    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    out_root = ensure_dir(cfg.out_dir)
    run_dir = ensure_dir(
        out_root
        / f"{cfg.logger_name}_{cfg.model}_{'pre' if cfg.pretrained else 'scratch'}_seed{cfg.seed}_img{cfg.image_size}_e{cfg.epochs}"
    )
    save_json(run_dir / "config.json", asdict(cfg))

    dl_cfg = Flowers102Config(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    train_loader, val_loader, test_loader = build_dataloaders(dl_cfg)

    model = ModelConfig(model=cfg.model, num_classes=102, pretrained=cfg.pretrained).build().to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing).to(device)

    backbone, head = ModelConfig.split_params_for_finetune(model)
    optimizer = optim.AdamW(
        [
            {"params": backbone, "lr": cfg.lr_backbone},
            {"params": head, "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )

    scheduler = None
    if cfg.onecycle:
        steps_per_epoch = len(train_loader)
        max_lr_backbone = cfg.max_lr_backbone if cfg.max_lr_backbone is not None else cfg.lr_backbone * 10
        max_lr_head = cfg.max_lr_head if cfg.max_lr_head is not None else cfg.lr_head * 10
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[max_lr_backbone, max_lr_head],
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=1e4,
            anneal_strategy="cos",
        )

    scaler = _grad_scaler(cfg.amp, device)
    logger = init_logger(
        backend=cfg.logger,
        project=cfg.logger_project,
        name=f"{cfg.logger_name}_{cfg.model}_{'pre' if cfg.pretrained else 'scratch'}",
        config=asdict(cfg),
        out_dir=run_dir,
    )

    best = -math.inf
    history = []
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scaler,
            scheduler,
            device,
            cfg.amp,
            cfg.grad_clip_norm,
        )
        va = evaluate(model, val_loader, loss_fn, device, cfg.amp)
        rec = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
        }
        history.append(rec)
        save_json(run_dir / "history.json", history)
        logger.log(rec, step=epoch)

        print(
            f"epoch {epoch:03d} | "
            f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.4f}"
        )

        if va["acc"] > best:
            best = va["acc"]
            ckpt = {
                "epoch": epoch,
                "best_val_acc": best,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": asdict(cfg),
            }
            torch.save(ckpt, run_dir / "best.pt")
            save_json(run_dir / "best_metrics.json", {"epoch": epoch, "best_val_acc": best, "val": va})

        torch.save({"epoch": epoch, "model_state": model.state_dict(), "config": asdict(cfg)}, run_dir / "last.pt")

    te = evaluate(model, test_loader, loss_fn, device, cfg.amp)
    save_json(run_dir / "test_metrics.json", te)
    logger.log({"test_acc": te["acc"], "test_loss": te["loss"]})
    logger.finish()
    print(f"test | loss {te['loss']:.4f} acc {te['acc']:.4f}")


if __name__ == "__main__":
    main()
