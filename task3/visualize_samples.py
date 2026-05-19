from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from src.datasets.stanford_background import CLASS_NAMES, StanfordBackgroundDataset, ensure_stanford_background_dataset, load_or_create_split
from src.models.unet import UNetConfig


PALETTE = [
    (135, 206, 235),
    (34, 139, 34),
    (128, 64, 128),
    (124, 252, 0),
    (0, 191, 255),
    (178, 34, 34),
    (160, 82, 45),
    (255, 215, 0),
]


def colorize_mask(mask: np.ndarray, ignore_index: int = 255) -> Image.Image:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(PALETTE[: len(CLASS_NAMES)]):
        rgb[mask == i] = c
    rgb[mask == ignore_index] = (0, 0, 0)
    return Image.fromarray(rgb, mode="RGB")


def overlay(image_rgb: Image.Image, mask_rgb: Image.Image, alpha: float = 0.45) -> Image.Image:
    img = image_rgb.convert("RGBA")
    m = mask_rgb.convert("RGBA")
    return Image.blend(img, m, alpha=alpha).convert("RGB")


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(3, 1, 1)
    return x * std + mean


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=str(root / "data"))
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--out", type=str, default=str(root / "runs" / "qualitative.png"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_root = ensure_stanford_background_dataset(args.data_dir)
    split = load_or_create_split(Path(dataset_root), seed=args.seed, val_ratio=0.2, test_ratio=0.0)
    val_ds = StanfordBackgroundDataset(
        dataset_root=Path(dataset_root),
        split="val",
        stems=split.val,
        image_size=(args.img_size, args.img_size),
        random_horizontal_flip=False,
        ignore_index=255,
    )

    device = torch.device(args.device if args.device else "cuda")
    ckpt = torch.load(args.weights, map_location=device)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    base_channels = int(cfg.get("base_channels", args.base_channels))
    norm = str(cfg.get("norm", "bn"))
    model = UNetConfig(in_channels=3, num_classes=len(CLASS_NAMES), base_channels=base_channels, norm=norm).build().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n = min(args.num_samples, len(val_ds))
    indices = list(range(n))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for row, idx in enumerate(indices):
        item = val_ds[idx]
        x = item["image"].unsqueeze(0).to(device)
        y = item["mask"].cpu().numpy().astype(np.int64)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)

        img_t = denormalize_imagenet(item["image"].to(torch.float32)).clamp(0.0, 1.0)
        img_np = (img_t.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_np, mode="RGB")
        gt_rgb = colorize_mask(y)
        pr_rgb = colorize_mask(pred)
        gt_ov = overlay(img, gt_rgb)
        pr_ov = overlay(img, pr_rgb)

        for col, (title, im) in enumerate(
            [
                ("Image", img),
                ("GT Mask", gt_rgb),
                ("Pred Mask", pr_rgb),
                ("Overlay (Pred)", pr_ov),
            ]
        ):
            axes[row, col].imshow(im)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
