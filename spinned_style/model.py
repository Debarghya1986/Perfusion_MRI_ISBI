"""SPINNED-style DSC-MRI deconvolution network.

This is a paper-faithful *adaptation* to the ISBI_2027 synthetic dataset, not
official author code.  The 2024 SPINNED paper establishes the key structure:
AIF + tissue concentration -> MLP -> tissue response function, with a
physics/reconvolution term.  No public official repository was located during
the benchmark audit, and the accessible paper sources did not expose enough
implementation detail to claim an exact architecture reproduction.

Our target is the flow-scaled TRF h(t) used by the project, length T=45.
"""

from __future__ import annotations
from typing import Sequence

import torch
from torch import Tensor, nn


class SPINNEDStyleMLP(nn.Module):
    """Voxel-wise MLP mapping [AIF(t), C_t(t)] -> h(t)."""

    def __init__(
        self,
        timepoints: int = 45,
        hidden_dims: Sequence[int] = (256, 256, 128),
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.timepoints = int(timepoints)
        if self.timepoints < 2:
            raise ValueError("timepoints must be >= 2.")

        dims = [2 * self.timepoints] + [int(v) for v in hidden_dims] + [self.timepoints]
        if any(v <= 0 for v in dims):
            raise ValueError("All layer widths must be positive.")

        act_name = activation.lower()
        if act_name == "silu":
            act_cls = nn.SiLU
        elif act_name == "relu":
            act_cls = nn.ReLU
        elif act_name == "gelu":
            act_cls = nn.GELU
        else:
            raise ValueError("activation must be 'silu', 'relu', or 'gelu'.")

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, aif: Tensor, concentration: Tensor) -> Tensor:
        """Parameters are (..., T) tensors; output is (..., T)."""
        if aif.shape[-1] != self.timepoints or concentration.shape[-1] != self.timepoints:
            raise ValueError(
                f"Expected final dimension T={self.timepoints}; got "
                f"{aif.shape[-1]} and {concentration.shape[-1]}."
            )
        if aif.shape[:-1] != concentration.shape[:-1]:
            aif = torch.broadcast_to(aif, concentration.shape)
        x = torch.cat([aif, concentration], dim=-1)
        return self.net(x)


def reconvolve_rowwise(h: Tensor, A: Tensor) -> Tensor:
    """Compute C = h A^T for batched rows.

    h: (N,T)
    A: (T,T) or (N,T,T)
    """
    if A.ndim == 2:
        return h @ A.transpose(-1, -2)
    if A.ndim == 3:
        return torch.bmm(h.unsqueeze(1), A.transpose(1, 2)).squeeze(1)
    raise ValueError("A must have shape (T,T) or (N,T,T).")


__all__ = ["SPINNEDStyleMLP", "reconvolve_rowwise"]
