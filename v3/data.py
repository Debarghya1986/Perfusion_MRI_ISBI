from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import resolve_path


def list_split(root: str | Path, split: str):
    return sorted((resolve_path(root) / split).glob("sample_*.npz"))


def load_case(path: str | Path) -> Dict:
    with np.load(path, allow_pickle=False) as d:
        out = {k: np.asarray(d[k]) for k in d.files}
    clean = np.asarray(out["clean"], np.float32)  # [T,H,W]
    out["h_gt"] = np.transpose(clean, (1, 2, 0)).astype(np.float32)
    out["mask"] = np.asarray(out["brain_mask"], np.float32)
    out["h_scale"] = float(np.asarray(out["h_scale"]).reshape(()))
    out["residue_family"] = str(np.asarray(out["residue_family"]).reshape(()))
    out["shape_model"] = str(np.asarray(out["shape_model"]).reshape(()))
    return out


class DenoiserDataset(Dataset):
    """AWGN denoiser dataset using only clean V3 h volumes."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        sigma_min: float,
        sigma_max: float,
        fixed_sigma: float | None = None,
        seed: int = 0,
    ):
        self.files = list_split(root, split)
        if not self.files:
            raise RuntimeError(f"No cases in {resolve_path(root) / split}")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.fixed_sigma = None if fixed_sigma is None else float(fixed_sigma)
        self.seed = int(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=False) as d:
            clean = np.asarray(d["clean"], np.float32)
        clean = torch.from_numpy(clean).unsqueeze(0)  # [1,T,H,W]
        if self.fixed_sigma is None:
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.rand(())
            noisy = clean + sigma * torch.randn_like(clean)
        else:
            sigma = torch.tensor(self.fixed_sigma, dtype=torch.float32)
            g = torch.Generator().manual_seed(self.seed + idx)
            noisy = clean + sigma * torch.randn(clean.shape, generator=g)
        return clean, noisy, sigma
