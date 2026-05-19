from __future__ import annotations

from dataclasses import dataclass

import torch


@torch.no_grad()
def confusion_matrix_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"logits must be NCHW, got shape={tuple(logits.shape)}")
    if target.ndim != 3:
        raise ValueError(f"target must be NHW, got shape={tuple(target.shape)}")

    pred = torch.argmax(logits, dim=1)
    valid = target != ignore_index
    pred = pred[valid].to(torch.int64)
    tgt = target[valid].to(torch.int64)
    if pred.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=logits.device)

    idx = tgt * num_classes + pred
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


@torch.no_grad()
def metrics_from_confusion(cm: torch.Tensor, eps: float = 1e-6) -> dict:
    cm_f = cm.to(torch.float32)
    tp = torch.diag(cm_f)
    fp = torch.sum(cm_f, dim=0) - tp
    fn = torch.sum(cm_f, dim=1) - tp
    denom = tp + fp + fn
    iou = tp / torch.clamp(denom, min=eps)
    miou = torch.mean(iou).item() if iou.numel() else 0.0

    total = torch.sum(cm_f).item()
    correct = torch.sum(tp).item()
    acc = (correct / total) if total > 0 else 0.0
    return {"miou": float(miou), "acc": float(acc), "iou_per_class": iou.cpu().tolist()}


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

