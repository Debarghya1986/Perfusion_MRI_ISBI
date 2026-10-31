"""Small helpers for the single project-level YAML configuration file."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch
import yaml


def load_config(path: str | Path = "config.yml") -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level YAML object must be a mapping: {path}")
    return cfg


def require_section(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, dict):
        raise KeyError(f"Missing mapping section '{name}' in config.yml")
    return dict(value)


def namespace(*mappings: Dict[str, Any], **extra: Any) -> SimpleNamespace:
    merged: Dict[str, Any] = {}
    for mapping in mappings:
        merged.update(mapping)
    merged.update(extra)
    return SimpleNamespace(**merged)


def resolve_device(value: Any) -> str:
    if value is None or str(value).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(value)
