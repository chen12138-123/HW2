import json
import os
import random
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


STANFORD_BG_URL = "http://dags.stanford.edu/data/iccv09Data.tar.gz"

CLASS_NAMES = (
    "sky",
    "tree",
    "road",
    "grass",
    "water",
    "building",
    "mountain",
    "foreground",
)


@dataclass(frozen=True)
class StanfordBackgroundSplit:
    train: list[str]
    val: list[str]
    test: list[str]


def _download(url: str, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return

    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        pct = 100.0 * downloaded / total_size
        print(f"\rDownloading {url} -> {dst_path.name}: {pct:6.2f}%", end="")

    urllib.request.urlretrieve(url, tmp_path, reporthook=reporthook)
    print()
    tmp_path.replace(dst_path)


def _extract(tar_gz_path: Path, dst_dir: Path) -> None:
    marker = dst_dir / ".extracted"
    if marker.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_gz_path, "r:gz") as tf:
        tf.extractall(dst_dir)
    marker.write_text("ok", encoding="utf-8")


def _find_dataset_root(root_dir: Path) -> Path:
    candidates = [
        root_dir,
        root_dir / "iccv09Data",
        root_dir / "data" / "iccv09Data",
    ]
    for c in candidates:
        if (c / "images").exists() and (c / "labels").exists():
            return c
    for c in root_dir.rglob("iccv09Data"):
        if (c / "images").exists() and (c / "labels").exists():
            return c
    raise FileNotFoundError(f"Could not find iccv09Data root under: {root_dir}")


def _read_regions_txt(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int64)


def _infer_label_offset(label_paths: list[Path]) -> int:
    for p in label_paths[: min(8, len(label_paths))]:
        arr = _read_regions_txt(p)
        valid = arr[arr >= 0]
        if valid.size == 0:
            continue
        mn = int(valid.min())
        mx = int(valid.max())
        if mn == 0:
            return 0
        if mn == 1 and mx == len(CLASS_NAMES):
            return 1
    return 0


def _build_split(
    dataset_root: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> StanfordBackgroundSplit:
    images_dir = dataset_root / "images"
    label_dir = dataset_root / "labels"
    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {images_dir}")

    stems: list[str] = []
    for ip in image_paths:
        stem = ip.stem
        if (label_dir / f"{stem}.regions.txt").exists():
            stems.append(stem)
    if not stems:
        raise FileNotFoundError(f"No image/label pairs found under: {dataset_root}")

    rng = random.Random(seed)
    rng.shuffle(stems)

    n = len(stems)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_test = max(0, min(n_test, n))
    n_val = max(0, min(n_val, n - n_test))

    test = stems[:n_test]
    val = stems[n_test : n_test + n_val]
    train = stems[n_test + n_val :]
    return StanfordBackgroundSplit(train=train, val=val, test=test)


def ensure_stanford_background_dataset(root_dir: str) -> Path:
    root_dir_path = Path(root_dir)
    root_dir_path.mkdir(parents=True, exist_ok=True)

    tar_path = root_dir_path / "iccv09Data.tar.gz"
    _download(STANFORD_BG_URL, tar_path)
    _extract(tar_path, root_dir_path)
    return _find_dataset_root(root_dir_path)


def load_or_create_split(
    dataset_root: Path,
    seed: int = 42,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
) -> StanfordBackgroundSplit:
    split_path = dataset_root / f"split_seed{seed}_v{val_ratio}_t{test_ratio}.json"
    if split_path.exists():
        obj = json.loads(split_path.read_text(encoding="utf-8"))
        return StanfordBackgroundSplit(train=obj["train"], val=obj["val"], test=obj["test"])

    split = _build_split(dataset_root, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    split_path.write_text(
        json.dumps({"train": split.train, "val": split.val, "test": split.test}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return split


def _resize_pair(
    img: Image.Image,
    mask: np.ndarray,
    size_hw: Tuple[int, int],
) -> Tuple[Image.Image, np.ndarray]:
    h, w = size_hw
    img_r = img.resize((w, h), resample=Image.BILINEAR)
    mask_img = Image.fromarray(mask.astype(np.int32), mode="I")
    mask_r = mask_img.resize((w, h), resample=Image.NEAREST)
    return img_r, np.array(mask_r, dtype=np.int64)


def _hflip_pair(img: Image.Image, mask: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
    return img.transpose(Image.FLIP_LEFT_RIGHT), np.ascontiguousarray(mask[:, ::-1])


def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x - mean) / std


def _random_resized_crop_pair(
    img: Image.Image,
    mask: np.ndarray,
    out_hw: Tuple[int, int],
    scale: Tuple[float, float] = (0.85, 1.0),
    ratio: Tuple[float, float] = (0.9, 1.1),
) -> Tuple[Image.Image, np.ndarray]:
    import torchvision.transforms as T
    import torchvision.transforms.functional as F

    h, w = out_hw
    i, j, th, tw = T.RandomResizedCrop.get_params(img, scale=scale, ratio=ratio)
    img = F.resized_crop(img, i, j, th, tw, size=[h, w], interpolation=Image.BILINEAR)
    mask_img = Image.fromarray(mask.astype(np.int32), mode="I")
    mask_img = F.resized_crop(mask_img, i, j, th, tw, size=[h, w], interpolation=Image.NEAREST)
    return img, np.array(mask_img, dtype=np.int64)


def _color_jitter(img: Image.Image) -> Image.Image:
    import torchvision.transforms as T

    cj = T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.03)
    return cj(img)


class StanfordBackgroundDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split: Literal["train", "val", "test"],
        stems: list[str],
        image_size: Optional[Tuple[int, int]] = (256, 256),
        random_horizontal_flip: bool = False,
        augment: bool = True,
        ignore_index: int = 255,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.stems = stems
        self.image_size = image_size
        self.random_horizontal_flip = random_horizontal_flip
        self.augment = augment
        self.ignore_index = ignore_index

        self.images_dir = dataset_root / "images"
        self.labels_dir = dataset_root / "labels"

        label_paths = [self.labels_dir / f"{s}.regions.txt" for s in self.stems]
        self.label_offset = _infer_label_offset(label_paths)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        img_path_jpg = self.images_dir / f"{stem}.jpg"
        img_path_png = self.images_dir / f"{stem}.png"
        img_path = img_path_jpg if img_path_jpg.exists() else img_path_png
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found for stem={stem}")

        label_path = self.labels_dir / f"{stem}.regions.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Label not found for stem={stem}")

        img = Image.open(img_path)
        mask = _read_regions_txt(label_path)
        mask = mask.astype(np.int64)

        ignore = mask < 0
        if self.label_offset:
            mask = np.where(mask >= 0, mask - self.label_offset, mask)
        mask = np.where(ignore, self.ignore_index, mask)

        if self.split == "train" and self.image_size is not None:
            if self.augment:
                img = _color_jitter(img)
                img, mask = _random_resized_crop_pair(img, mask, self.image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1))
            else:
                img, mask = _resize_pair(img, mask, self.image_size)
            if self.random_horizontal_flip and random.random() < 0.5:
                img, mask = _hflip_pair(img, mask)
        else:
            if self.image_size is not None:
                img, mask = _resize_pair(img, mask, self.image_size)

        x = _normalize(_image_to_tensor(img))
        y = torch.from_numpy(mask.astype(np.int64))
        return {"image": x, "mask": y, "stem": stem}

