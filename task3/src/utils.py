from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def now_compact() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class Logger:
    backend: str = "none"
    run: Any = None

    def log(self, d: dict, step: Optional[int] = None) -> None:
        if self.backend == "wandb":
            if step is not None:
                self.run.log(d, step=step)
            else:
                self.run.log(d)
        elif self.backend == "swanlab":
            if step is not None:
                self.run.log(d, step=step)
            else:
                self.run.log(d)

    def finish(self) -> None:
        if self.backend == "wandb" and self.run is not None:
            self.run.finish()
        if self.backend == "swanlab" and self.run is not None:
            self.run.finish()


def init_logger(
    backend: str,
    project: str,
    name: str,
    config: dict,
    out_dir: Path,
) -> Logger:
    backend = (backend or "none").lower()
    if backend == "wandb":
        try:
            import wandb
        except Exception:
            return Logger(backend="none", run=None)
        run = wandb.init(project=project, name=name, config=config, dir=str(out_dir))
        return Logger(backend="wandb", run=run)

    if backend == "swanlab":
        try:
            import swanlab
        except Exception:
            return Logger(backend="none", run=None)
        run = swanlab.init(project=project, experiment_name=name, config=config, logdir=str(out_dir))
        return Logger(backend="swanlab", run=run)

    return Logger(backend="none", run=None)

