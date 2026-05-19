from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class Flowers102Config:
    data_dir: str
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4


def build_transforms(image_size: int, is_train: bool):
    import torchvision.transforms as T

    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if is_train:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
                normalize,
            ]
        )
    return T.Compose(
        [
            T.Resize(int(image_size * 1.15)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            normalize,
        ]
    )


def build_flowers102_dataset(data_dir: str, split: Literal["train", "val", "test"], image_size: int):
    from torchvision.datasets import Flowers102

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    ds = Flowers102(root=str(root), split=split, download=True, transform=build_transforms(image_size, is_train=(split == "train")))
    return ds


def build_dataloaders(cfg: Flowers102Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = build_flowers102_dataset(cfg.data_dir, "train", cfg.image_size)
    val_ds = build_flowers102_dataset(cfg.data_dir, "val", cfg.image_size)
    test_ds = build_flowers102_dataset(cfg.data_dir, "test", cfg.image_size)

    common = dict(num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader

