from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(p: str | Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else ROOT / p


def load_yaml(path: str | Path) -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, obj: Any) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_csv(path: str | Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def resolve_device(value: str) -> torch.device:
    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(value)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return dev


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_sd(values):
    x = np.asarray(list(values), dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1) if len(x) > 1 else 0.0)
