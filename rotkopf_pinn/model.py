"""Vectorized voxel-wise PINN for the Rotkopf et al. (2024) DSC benchmark.

Paper-derived structure
-----------------------
- one scalar time input;
- four hidden layers, 32 units each;
- one scalar TRF output;
- Swish/SiLU after every hidden layer;
- independent voxel-wise networks fitted in parallel;
- convolution MSE + elastic-net + temporal-TV regularization.

Two implementation details are important here:

1) Batched Xavier initialization
   ``weights`` has shape [V, Din, Dout]. Calling ``nn.init.xavier_uniform_``
   directly on that 3-D tensor uses V in PyTorch's fan calculation and makes
   every voxel network far too weakly initialized. We therefore initialize
   each batched matrix with the Xavier bound for a *single* [Din,Dout] MLP.

2) Per-voxel output scaling
   Rotkopf et al.'s fitting-loop caption explicitly includes a fitted scaling
   factor. We represent that as one trainable multiplicative scalar per voxel.
   It is initialized to one. The paper does not specify its initialization, so
   this is an explicit implementation choice rather than a claimed exact detail.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class VoxelMLPBank(nn.Module):
    def __init__(
        self,
        n_voxels: int,
        hidden_width: int = 32,
        hidden_layers: int = 4,
        use_output_scale: bool = True,
    ):
        super().__init__()

        self.n_voxels = int(n_voxels)
        self.hidden_width = int(hidden_width)
        self.hidden_layers = int(hidden_layers)
        self.use_output_scale = bool(use_output_scale)

        dims = [1] + [self.hidden_width] * self.hidden_layers + [1]

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for din, dout in zip(dims[:-1], dims[1:]):
            w = nn.Parameter(
                torch.empty(
                    self.n_voxels,
                    int(din),
                    int(dout),
                )
            )
            b = nn.Parameter(
                torch.zeros(
                    self.n_voxels,
                    int(dout),
                )
            )

            # Xavier-uniform for each independent [Din,Dout] voxel matrix.
            # Since all voxel matrices at a layer have the same shape, filling
            # the whole batched tensor with this bound is equivalent to
            # independently Xavier-initializing every voxel network.
            bound = math.sqrt(6.0 / (float(din) + float(dout)))
            with torch.no_grad():
                w.uniform_(-bound, bound)

            self.weights.append(w)
            self.biases.append(b)

        if self.use_output_scale:
            self.output_scale = nn.Parameter(
                torch.ones(
                    self.n_voxels,
                    1,
                )
            )
        else:
            self.register_parameter(
                "output_scale",
                None,
            )

    def forward(self, t: Tensor) -> Tensor:
        """Evaluate independent voxel TRFs.

        Parameters
        ----------
        t : Tensor, shape [T] or [T,1]

        Returns
        -------
        Tensor, shape [V,T]
        """
        if t.ndim == 1:
            t = t[:, None]

        if t.ndim != 2 or t.shape[-1] != 1:
            raise ValueError(
                f"Expected t with shape [T] or [T,1], got {tuple(t.shape)}."
            )

        x = t[None, :, :].expand(
            self.n_voxels,
            -1,
            -1,
        )

        for i, (w, b) in enumerate(
            zip(self.weights, self.biases)
        ):
            x = torch.einsum(
                "vti,vio->vto",
                x,
                w,
            ) + b[:, None, :]

            if i < len(self.weights) - 1:
                x = torch.nn.functional.silu(x)

        h = x[..., 0]

        if self.output_scale is not None:
            h = h * self.output_scale

        return h


def rotkopf_loss(
    h: Tensor,
    C: Tensor,
    A: Tensor,
    lambda_en: float = 0.5,
    lambda_tv: float = 0.5,
):
    """Rotkopf loss with independent voxel-wise objectives fitted in parallel.

    The equations in Rotkopf et al. average over *time* for each voxel.
    We therefore build one objective per voxel and sum those objectives for
    backpropagation. Because model parameters are independent across voxels,
    this gives each voxel network the same gradient it would obtain if fitted
    separately, while still allowing fully parallel GPU optimization.

    Returns
    -------
    total_for_backward : scalar Tensor
        Sum of voxel objectives.
    conv_mean, en_mean, tv_mean : scalar Tensor
        Mean diagnostic values across voxels.
    """
    if h.ndim != 2 or C.ndim != 2:
        raise ValueError("h and C must have shape [V,T].")
    if h.shape != C.shape:
        raise ValueError(f"h shape {h.shape} != C shape {C.shape}.")
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != h.shape[1]:
        raise ValueError(
            f"A must be square [T,T] with T={h.shape[1]}; got {A.shape}."
        )

    C_pred = h @ A.transpose(-1, -2)

    loss_conv_v = torch.mean(
        (C_pred - C) ** 2,
        dim=1,
    )

    loss_en_v = (
        torch.mean(
            torch.abs(h),
            dim=1,
        )
        + torch.mean(
            h ** 2,
            dim=1,
        )
    )

    if h.shape[1] > 1:
        loss_tv_v = torch.mean(
            torch.abs(
                h[:, 1:] - h[:, :-1]
            ),
            dim=1,
        )
    else:
        loss_tv_v = torch.zeros_like(loss_conv_v)

    total_v = (
        loss_conv_v
        + float(lambda_en) * loss_en_v
        + float(lambda_tv) * loss_tv_v
    )

    total_for_backward = torch.sum(total_v)

    return (
        total_for_backward,
        torch.mean(loss_conv_v),
        torch.mean(loss_en_v),
        torch.mean(loss_tv_v),
    )


__all__ = [
    "VoxelMLPBank",
    "rotkopf_loss",
]
