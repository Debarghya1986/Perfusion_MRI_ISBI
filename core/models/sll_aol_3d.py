"""SLL/AOL reparameterized 3D convolution.

The layer implements the convolutional specialization of

    W = W_tilde R^{-1/2},
    R = Diag(Diag(q)^{-1} |W_tilde^T W_tilde| q),

using the translation-invariant row-sum bound of the 3D convolution kernel.
The resulting convolution is nonexpansive in the Euclidean norm.  A finite
zero-padded convolution is a restriction of the corresponding full operator
and therefore remains nonexpansive.

The public ``weight`` attribute is the reparameterized kernel.  The trainable
unconstrained kernel is available through
``module.parametrizations.weight.original``.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization


Size3D = Union[int, Tuple[int, int, int]]


class SLLAOLConv3DRescaling(nn.Module):
    """AOL diagonal rescaling for a ``(Co, Ci, Kt, Kh, Kw)`` kernel."""

    def __init__(
        self,
        q: nn.Parameter,
        eps: float = 1e-6,
        log_q_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.q = q
        self.eps = float(eps)
        self.log_q_clip = float(log_q_clip)
        if self.eps <= 0:
            raise ValueError("eps must be strictly positive.")
        if self.log_q_clip <= 0:
            raise ValueError("log_q_clip must be strictly positive.")

    def forward(self, weight: Tensor) -> Tensor:
        if weight.ndim != 5:
            raise ValueError(
                "SLLAOLConv3DRescaling expects a 5D Conv3d kernel, "
                f"received shape {tuple(weight.shape)}."
            )

        _, in_channels, kt, kh, kw = weight.shape
        if self.q.numel() != in_channels:
            raise ValueError(
                f"q has {self.q.numel()} entries but the kernel has "
                f"{in_channels} input channels."
            )

        # The normalization is evaluated in at least float32 even inside an
        # AMP region.  Computing the Gram correlations in float16 can
        # underestimate an absolute row sum through rounding, which is the
        # wrong direction for a certified upper bound.
        with torch.autocast(device_type=weight.device.type, enabled=False):
            work_dtype = (
                torch.float32
                if weight.dtype in (torch.float16, torch.bfloat16)
                else weight.dtype
            )
            work_weight = weight.to(dtype=work_dtype)

            # ``weight_tp`` is interpreted by conv3d as a batch of Ci inputs
            # and simultaneously as Ci filters.  The result contains all 3D
            # cross-correlations of the input-channel blocks of W^T W.
            weight_tp = work_weight.transpose(0, 1)
            gram_corr = F.conv3d(
                weight_tp,
                weight_tp,
                padding=(kt - 1, kh - 1, kw - 1),
            )

            # Only ratios q_j/q_i matter.  Centering log(q) removes a null
            # direction. Row and column factors deliberately use different
            # axes; using the same view would cancel q entirely. Clipping
            # merely selects another strictly positive diagonal vector and
            # prevents exponent overflow.
            work_q = self.q.to(dtype=work_dtype)
            log_q = (work_q - work_q.mean()).clamp(
                -self.log_q_clip, self.log_q_clip
            )
            q_row_inv = torch.exp(-log_q).reshape(-1, 1, 1, 1, 1)
            q_col = torch.exp(log_q).reshape(1, -1, 1, 1, 1)

            r = (q_row_inv * gram_corr.abs() * q_col).sum(
                dim=(1, 2, 3, 4)
            )
            inv_sqrt_r = torch.rsqrt(r + self.eps).reshape(
                1, -1, 1, 1, 1
            )
            scaled = work_weight * inv_sqrt_r
        return scaled.to(dtype=weight.dtype)

    def right_inverse(self, weight: Tensor) -> Tensor:
        return weight


class SLLAOLConv3d(nn.Conv3d):
    """Drop-in ``nn.Conv3d`` with an SLL/AOL nonexpansive weight.

    The layer itself supports the standard Conv3d arguments.  The CCD wrapper
    uses odd kernels, unit stride, and symmetric zero padding so that
    ``conv_transpose3d`` with the same parameters is its exact adjoint and has
    the same output shape.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3D,
        stride: Size3D = 1,
        padding: Union[Size3D, str] = 0,
        dilation: Size3D = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        initializer: str = "orthogonal",
        eps: float = 1e-6,
        log_q_clip: float = 10.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if groups != 1:
            raise NotImplementedError(
                "The present SLL/AOL channel bound is implemented for groups=1."
            )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        # q stores log-positive diagonal weights.  q=0 corresponds to q_i=1.
        self.q = nn.Parameter(
            torch.zeros(in_channels, device=self.weight.device,
                        dtype=self.weight.dtype)
        )

        if initializer == "orthogonal":
            nn.init.orthogonal_(self.weight)
        elif initializer == "dirac":
            nn.init.dirac_(self.weight)
        elif initializer == "xavier_normal":
            nn.init.xavier_normal_(self.weight)
        else:
            raise ValueError(
                "initializer must be one of "
                "{'orthogonal', 'dirac', 'xavier_normal'}"
            )

        if self.bias is not None:
            nn.init.zeros_(self.bias)

        register_parametrization(
            self,
            "weight",
            SLLAOLConv3DRescaling(
                self.q, eps=eps, log_q_clip=log_q_clip
            ),
        )


__all__ = ["SLLAOLConv3d", "SLLAOLConv3DRescaling"]
