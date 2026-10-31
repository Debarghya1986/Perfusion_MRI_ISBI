"""Dataset loaders for AWGN pretraining of CCD3D on synthetic h-bar volumes.

Training task:
    h_bar + epsilon_h -> h_bar,
    epsilon_h ~ N(0, sigma_h^2).

Training samples return ``(clean, sigma)`` so the supplied CCD3D trainer creates
fresh AWGN every epoch. Validation samples return ``(clean, noisy, sigma)`` with
fixed deterministic noise, which makes validation PSNR comparable across epochs.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RotkopfHbarDataset(Dataset):
    def __init__(
        self,
        root="data/rotkopf_synthetic",
        split="train",
        sigma_min=0.0,
        sigma_max=25.0 / 255.0,
        fixed_sigma=None,
        augment=False,
        deterministic_noisy=False,
        seed=0,
    ):
        self.root = Path(root)
        self.split = split
        self.files = sorted((self.root / split).glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {self.root / split}. Run 01_generate_rotkopf.py first.")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.fixed_sigma = None if fixed_sigma is None else float(fixed_sigma)
        self.augment = bool(augment)
        self.deterministic_noisy = bool(deterministic_noisy)
        self.seed = int(seed)

    def __len__(self):
        return len(self.files)

    def _sigma(self):
        if self.fixed_sigma is not None:
            return self.fixed_sigma
        return float(torch.empty(1).uniform_(self.sigma_min, self.sigma_max).item())

    @staticmethod
    def _augment(x, code):
        # x [C,T,H,W]. Only spatial H/W are transformed; time is untouched.
        x = torch.rot90(x, k=code % 4, dims=(-2, -1))
        if code >= 4:
            x = torch.flip(x, dims=(-1,))
        return x.contiguous()

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as d:
            clean = torch.from_numpy(np.asarray(d["clean"], dtype=np.float32)).unsqueeze(0)  # [1,T,H,W]
        if self.augment:
            # V3 is rectangular (66x77). Restrict the original V1/V2
            # dihedral augmentation to transformations that preserve HxW:
            # identity, 180-degree rotation, horizontal flip, vertical flip.
            code = (0, 2, 4, 6)[int(torch.randint(0, 4, (1,)).item())]
            clean = self._augment(clean, code)
        sigma = torch.tensor(self._sigma(), dtype=torch.float32)

        if not self.deterministic_noisy:
            # Trainer will synthesize fresh AWGN using this sigma.
            return clean, sigma

        g = torch.Generator().manual_seed(self.seed + idx)
        noisy = clean + sigma * torch.randn(clean.shape, generator=g, dtype=clean.dtype)
        return clean, noisy, sigma


def create_dataloaders(
    data_root="data/rotkopf_synthetic",
    batch_size=4,
    num_workers=2,
    sigma_min=0.0,
    sigma_max=25.0 / 255.0,
    val_sigma=25.0 / 255.0,
    seed=0,
):
    train_ds = RotkopfHbarDataset(
        data_root, "train", sigma_min, sigma_max,
        fixed_sigma=None, augment=True, deterministic_noisy=False, seed=seed,
    )
    val_ds = RotkopfHbarDataset(
        data_root, "val", val_sigma, val_sigma,
        fixed_sigma=val_sigma, augment=False, deterministic_noisy=True,
        seed=seed + 1_000_000,
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, persistent_workers=(num_workers > 0), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=(num_workers > 0), drop_last=False,
    )
    return train_loader, val_loader


__all__ = ["RotkopfHbarDataset", "create_dataloaders"]
