"""Composed SLL/AOL Conv3d analysis operator and its exact adjoint."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from core.models.sll_aol_3d import SLLAOLConv3d


Size3D = Union[int, Tuple[int, int, int]]


def _triple(value: Size3D) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"Expected an int or length-3 tuple, received {value}.")
    return tuple(int(v) for v in value)


class MultiSLConv3d(nn.Module):
    """A cascade of nonexpansive SLL/AOL 3D convolutions.

    All kernels are odd and all layers use unit stride with symmetric zero
    padding.  Therefore every feature tensor keeps shape ``(T, H, W)`` and
    reversing the layers with ``conv_transpose3d`` gives the exact Euclidean
    adjoint.  Since each layer has operator norm at most one, their composition
    is nonexpansive.
    """

    def __init__(
        self,
        num_channels: Sequence[int] = (1, 64),
        size_kernels: Sequence[Size3D] = (3,),
        initializer: str = "orthogonal",
        eps: float = 1e-6,
        log_q_clip: float = 10.0,
    ) -> None:
        super().__init__()

        if len(num_channels) < 2:
            raise ValueError("num_channels must contain at least input/output entries.")
        if len(size_kernels) != len(num_channels) - 1:
            raise ValueError(
                "size_kernels must have len(num_channels)-1 entries."
            )

        self.num_channels = [int(c) for c in num_channels]
        if any(c <= 0 for c in self.num_channels):
            raise ValueError("Every entry in num_channels must be positive.")
        self.size_kernels = [_triple(k) for k in size_kernels]
        self.conv_layers = nn.ModuleList()

        for index, (cin, cout, kernel) in enumerate(
            zip(self.num_channels[:-1], self.num_channels[1:], self.size_kernels)
        ):
            if any(k <= 0 or k % 2 == 0 for k in kernel):
                raise ValueError(
                    f"Layer {index} has kernel {kernel}; odd positive kernels "
                    "are required for same-size exact-adjoint operation."
                )
            padding = tuple(k // 2 for k in kernel)
            self.conv_layers.append(
                SLLAOLConv3d(
                    in_channels=cin,
                    out_channels=cout,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                    dilation=1,
                    groups=1,
                    bias=False,
                    padding_mode="zeros",
                    initializer=initializer,
                    eps=eps,
                    log_q_clip=log_q_clip,
                )
            )

        self.padding_total = tuple(
            sum(kernel[axis] // 2 for kernel in self.size_kernels)
            for axis in range(3)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.convolution(x)

    def convolution(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(
                "MultiSLConv3d expects (B,C,T,H,W), "
                f"received {tuple(x.shape)}."
            )
        for conv in self.conv_layers:
            x = F.conv3d(
                x,
                conv.weight,
                bias=None,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
            )
        return x

    def transpose(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(
                "MultiSLConv3d.transpose expects (B,C,T,H,W), "
                f"received {tuple(x.shape)}."
            )
        for conv in reversed(self.conv_layers):
            x = F.conv_transpose3d(
                x,
                conv.weight,
                bias=None,
                stride=conv.stride,
                padding=conv.padding,
                output_padding=0,
                groups=conv.groups,
                dilation=conv.dilation,
            )
        return x

    @torch.no_grad()
    def adjoint_relative_error(
        self,
        spatial_shape: Tuple[int, int, int] = (7, 17, 19),
        batch_size: int = 2,
    ) -> Tensor:
        """Return ``|<Wx,y>-<x,W^Ty>|`` divided by the inner-product scale."""
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        x = torch.randn(
            batch_size, self.num_channels[0], *spatial_shape,
            device=device, dtype=dtype,
        )
        with parametrize.cached():
            wx = self.convolution(x)
            y = torch.randn_like(wx)
            lhs = (wx * y).sum()
            rhs = (x * self.transpose(y)).sum()
        scale = lhs.abs() + rhs.abs() + torch.finfo(dtype).eps
        return (lhs - rhs).abs() / scale


__all__ = ["MultiSLConv3d"]
