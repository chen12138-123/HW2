from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"logits must be NCHW, got shape={tuple(logits.shape)}")
    if target.ndim != 3:
        raise ValueError(f"target must be NHW, got shape={tuple(target.shape)}")

    n, c, h, w = logits.shape
    if c != num_classes:
        raise ValueError(f"logits channels={c} != num_classes={num_classes}")
    if target.shape != (n, h, w):
        raise ValueError(f"target shape={tuple(target.shape)} != (N,H,W)={(n,h,w)}")

    valid = target != ignore_index
    if valid.sum() == 0:
        return logits.sum() * 0.0

    probs = F.softmax(logits, dim=1)
    target_clamped = torch.where(valid, target, torch.zeros_like(target))
    one_hot = F.one_hot(target_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()
    mask = valid.unsqueeze(1).float()

    probs = probs * mask
    one_hot = one_hot * mask

    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    cardinality = torch.sum(probs + one_hot, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


@dataclass(frozen=True)
class DiceLossConfig:
    num_classes: int
    ignore_index: int = 255
    eps: float = 1e-6

    def build(self) -> nn.Module:
        cfg = self

        class _DiceLoss(nn.Module):
            def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return dice_loss_from_logits(
                    logits,
                    target,
                    num_classes=cfg.num_classes,
                    ignore_index=cfg.ignore_index,
                    eps=cfg.eps,
                )

        return _DiceLoss()

