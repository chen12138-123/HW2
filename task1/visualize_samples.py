from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

from src.datasets.flowers102 import Flowers102Config, build_dataloaders
from src.models.resnet_models import ModelConfig


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=str(root / "data"))
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--out", type=str, default=str(root / "runs" / "qualitative.png"))
    p.add_argument("--model", type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet18_se", "resnet34_se", "resnet18_cbam", "resnet34_cbam"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_samples", type=int, default=12)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dl_cfg = Flowers102Config(data_dir=args.data_dir, image_size=args.image_size, batch_size=args.batch_size, num_workers=args.num_workers)
    _, _, test_loader = build_dataloaders(dl_cfg)

    device = torch.device(args.device if args.device else "cuda")
    model = ModelConfig(model=args.model, num_classes=102, pretrained=False).build().to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    import matplotlib.pyplot as plt
    import numpy as np

    images: List[Image.Image] = []
    titles: List[str] = []

    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        x_np = (x.detach().cpu().permute(0, 2, 3, 1).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        x_np = (x_np * 255.0).clip(0, 255).astype(np.uint8)

        for i in range(x_np.shape[0]):
            img = Image.fromarray(x_np[i], mode="RGB")
            images.append(img)
            titles.append(f"GT={int(y[i].item())+1} Pred={int(pred[i].item())+1}")
            if len(images) >= args.num_samples:
                break
        if len(images) >= args.num_samples:
            break

    n = len(images)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")
        if idx < n:
            axes[r, c].imshow(images[idx])
            axes[r, c].set_title(titles[idx], fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

