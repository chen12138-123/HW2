from __future__ import annotations

import torch


@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    correct = (pred == targets).sum().item()
    total = targets.numel()
    return float(correct / total) if total > 0 else 0.0


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

