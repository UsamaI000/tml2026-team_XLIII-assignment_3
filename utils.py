from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int):
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


@dataclass
class EarlyStoppingConfig:
    patience: int = 15
    min_epochs: int = 20
    min_delta: float = 1e-4
    ema_alpha: float = 0.3
    clean_acc_floor: float = 0.55
    max_overfit_gap: float = 0.35


class RobustEarlyStopping:
    """Early stopping with smoothing and clean-accuracy safety gate.

    score should be validation clean accuracy for clean training, or the assignment
    proxy score 0.5*clean + 0.5*robust for PGD training.
    """

    def __init__(self, cfg: EarlyStoppingConfig):
        self.cfg = cfg
        self.best_epoch = -1
        self.best_value = -1e9
        self.best_ema = None
        self.bad_epochs = 0

    def step(self, epoch: int, score: float, val_clean_acc: float, train_acc: Optional[float] = None) -> tuple[bool, bool]:
        if self.best_ema is None:
            ema = score
        else:
            ema = self.cfg.ema_alpha * score + (1 - self.cfg.ema_alpha) * self.best_ema
        self.best_ema = ema

        # Safety gate: prefer not to save models that are likely to fail server clean gate.
        eligible = val_clean_acc >= self.cfg.clean_acc_floor
        if train_acc is not None:
            eligible = eligible and ((train_acc - val_clean_acc) <= self.cfg.max_overfit_gap)

        improved = eligible and (ema > self.best_value + self.cfg.min_delta)
        if improved:
            self.best_value = ema
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            if epoch >= self.cfg.min_epochs:
                self.bad_epochs += 1

        should_stop = epoch >= self.cfg.min_epochs and self.bad_epochs >= self.cfg.patience
        return improved, should_stop


def save_state_dict(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def append_history(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
