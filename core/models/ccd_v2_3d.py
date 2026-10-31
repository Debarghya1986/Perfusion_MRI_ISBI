"""Spatiotemporal SLL/AOL contractive convolutional denoiser.

Input convention: ``(batch, channels, time, height, width)``.

For fixed noise level sigma, the model has gradient-step form

    D_sigma(x) = x - gamma * grad Phi_sigma(x),

where ``grad Phi_sigma`` is implemented with the same nonexpansive analysis
operator W and exact adjoint W^T. The public sigma is normalized (e.g. 25/255)
and is multiplied by 255 internally for conditioning, exactly as in the current
lab GitHub implementation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.utils.parametrize as parametrize

from core.activations.spline_module import LinearSpline
from core.models.multi_sl_conv_3d import MultiSLConv3d


SigmaLike = Union[float, int, Tensor]


class CCD3DV2(nn.Module):
    """Contractive CCD for spatiotemporal ``(B,C,T,H,W)`` tensors.

    ``param_multi_conv['num_channels'][0]`` must equal the image channel count
    for a blind model and image channel count + 1 for a non-blind model, since
    the latter concatenates one constant sigma map.
    """

    def __init__(
        self,
        param_multi_conv: Dict,
        param_spline_activation: Dict,
        param_spline_scaling: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.blind = bool(kwargs.get("blind", False))
        self.scaling = bool(kwargs.get("scaling", False))
        self.scaling_eps = float(kwargs.get("scaling_eps", 1e-5))
        if self.scaling_eps <= 0:
            raise ValueError("scaling_eps must be strictly positive.")
        self.step_size_type = kwargs.get("step_size_type", "t2")
        if self.step_size_type not in ("t0", "t1", "t2", "t3", "default"):
            raise ValueError(
                "step_size_type must be one of "
                "{'t0', 't1', 't2', 't3', 'default'}."
            )
        self.enforce_contractivity = bool(
            kwargs.get("enforce_contractivity", True)
        )

        self.tau_min = float(kwargs.get("tau_min", 0.0102))
        self.tau_max = float(kwargs.get("tau_max", 0.135))
        init_tau = float(kwargs.get("init_tau", 0.05))
        if not 0.0 < self.tau_min <= self.tau_max:
            raise ValueError("Require 0 < tau_min <= tau_max.")
        if self.step_size_type == "t0" and self.tau_max > 0.5:
            raise ValueError(
                "step_size_type='t0' uses gamma=1 and requires tau <= 0.5."
            )
        self.tau = nn.Parameter(torch.tensor(init_tau, dtype=torch.float32))

        channel_schedule = list(param_multi_conv["num_channels"])
        self.analysis_in_channels = int(channel_schedule[0])
        self.num_channels = int(channel_schedule[-1])
        self.image_channels = (
            self.analysis_in_channels if self.blind
            else self.analysis_in_channels - 1
        )
        if self.image_channels < 1:
            raise ValueError(
                "A non-blind model needs at least one image channel plus one "
                "sigma-map channel."
            )

        activation_params = dict(param_spline_activation)
        activation_params["num_activations"] = self.num_channels
        self.activation = LinearSpline(**activation_params)
        if self.enforce_contractivity:
            self._validate_activation_contractivity()

        self.spline_scaling = None
        if self.scaling:
            if param_spline_scaling is None:
                raise ValueError(
                    "param_spline_scaling is required when scaling=True."
                )
            scaling_params = dict(param_spline_scaling)
            scaling_params["num_activations"] = self.num_channels
            self.spline_scaling = LinearSpline(**scaling_params)

        self.conv_layer = MultiSLConv3d(
            num_channels=channel_schedule,
            size_kernels=param_multi_conv["size_kernels"],
            initializer=kwargs.get("initializer", "orthogonal"),
            eps=float(kwargs.get("sll_eps", 1e-6)),
            log_q_clip=float(kwargs.get("sll_log_q_clip", 10.0)),
        )

        self.bias = None
        if kwargs.get("bias", False):
            self.bias = nn.Parameter(torch.zeros(self.num_channels))

        self.project_parameters()

    def _validate_activation_contractivity(self) -> None:
        """Verify the actual derivative bounds used by ``spline_module``.

        ``LinearSpline`` divides adjacent coefficient differences by the knot
        spacing before clipping. Therefore ``slope_min`` and ``slope_max`` are
        already derivative bounds, matching the current GitHub implementation.
        """
        slope_min = self.activation.slope_min
        slope_max = self.activation.slope_max
        if slope_min is None or slope_max is None:
            raise ValueError("Contractivity requires finite spline slope bounds.")
        slope_min = float(slope_min)
        slope_max = float(slope_max)
        tolerance = 1e-7
        if slope_min < -tolerance or slope_max > 1.0 + tolerance:
            raise ValueError(
                "The main spline must have derivative in [0,1], so require "
                "0 <= slope_min <= slope_max <= 1."
            )
        if slope_min > slope_max:
            raise ValueError("Require slope_min <= slope_max.")

    @torch.no_grad()
    def project_parameters(self) -> None:
        """Project scalar parameters onto the contractive admissible set."""
        self.tau.clamp_(self.tau_min, self.tau_max)

    @property
    def effective_tau(self) -> Tensor:
        # The forward pass is safe even when the module is optimized outside
        # the supplied trainer and the raw parameter has not yet been projected.
        return self.tau.clamp(self.tau_min, self.tau_max)

    def _canonical_sigma(self, sigma: Optional[SigmaLike], x: Tensor) -> Tensor:
        """Return ``255 * sigma`` broadcast as ``(B,1,T,H,W)``.

        The public API receives sigma in normalized signal units used to create
        AWGN (for example ``25/255`` for data in [0,1]).  Following the current
        lab GitHub code, CCD converts it to the 0--255 conditioning convention
        exactly once inside the model.
        """
        if x.ndim != 5:
            raise ValueError(
                "CCD3DV2 expects (B,C,T,H,W), "
                f"received shape {tuple(x.shape)}."
            )
        batch, _, time, height, width = x.shape
        if sigma is None:
            if self.blind and not self.scaling:
                return x.new_zeros((batch, 1, 1, 1, 1)).expand(
                    batch, 1, time, height, width
                )
            raise ValueError("sigma is required for a non-blind or scaled CCD3D.")

        sigma_tensor = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)

        if sigma_tensor.numel() in (1, batch) and sigma_tensor.ndim < 5:
            sigma_tensor = sigma_tensor.reshape(-1, 1, 1, 1, 1)
        elif sigma_tensor.ndim != 5:
            raise ValueError(
                "sigma must be a scalar, one scalar per sample, or a 5D "
                "(B,1,T,H,W) map."
            )

        if sigma_tensor.shape[0] not in (1, batch):
            raise ValueError(
                f"sigma batch dimension {sigma_tensor.shape[0]} is not "
                f"broadcastable to batch size {batch}."
            )
        if sigma_tensor.shape[1] != 1:
            raise ValueError("sigma must have exactly one channel.")

        target = (batch, 1, time, height, width)
        try:
            sigma_tensor = sigma_tensor.expand(target)
        except RuntimeError as exc:
            raise ValueError(
                f"sigma shape {tuple(sigma_tensor.shape)} cannot broadcast to {target}."
            ) from exc

        if torch.any(sigma_tensor < 0):
            raise ValueError("Noise standard deviation sigma must be nonnegative.")
        return sigma_tensor * 255.0

    def _analysis_input(self, x: Tensor, sigma_map: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(
                "CCD3DV2 expects (B,C,T,H,W), "
                f"received shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.image_channels:
            raise ValueError(
                f"Expected {self.image_channels} image channels, "
                f"received {x.shape[1]}."
            )
        return x if self.blind else torch.cat((x, sigma_map), dim=1)

    def get_scaling(self, sigma_map: Tensor) -> Tensor:
        if self.spline_scaling is None:
            raise RuntimeError("Scaling was not enabled for this model.")
        sigma_features = sigma_map.expand(
            sigma_map.shape[0], self.num_channels, *sigma_map.shape[2:]
        )
        return torch.exp(self.spline_scaling(sigma_features)) / (
            sigma_features + self.scaling_eps
        )

    def grad(self, x: Tensor, sigma: Optional[SigmaLike] = None) -> Tensor:
        sigma_map = self._canonical_sigma(sigma, x)
        y = self._analysis_input(x, sigma_map)

        # Some parametrizations compute their weights procedurally.  Caching
        # guarantees that W and W^T use the identical realized kernel.
        with parametrize.cached():
            y = self.conv_layer.convolution(y)
            if self.bias is not None:
                y = y + self.bias.reshape(1, -1, 1, 1, 1)

            scaling = None
            if self.scaling:
                scaling = self.get_scaling(sigma_map)
                y = y * scaling

            y = self.activation(y)

            if scaling is not None:
                y = y / scaling

            y = self.conv_layer.transpose(y)

        if not self.blind:
            y = y[:, : self.image_channels]
        return y + self.effective_tau * x

    def step_size(self) -> Tensor:
        tau = self.effective_tau
        if self.step_size_type == "t0":
            return torch.ones_like(tau)
        if self.step_size_type == "t1":
            return 2.0 / (1.0 + 2.0 * tau)
        if self.step_size_type == "t2":
            return 1.0 / (1.0 + 2.0 * tau)
        if self.step_size_type in ("t3", "default"):
            return 1.0 / (1.0 + tau)
        raise ValueError(
            "step_size_type must be one of "
            "{'t0', 't1', 't2', 't3', 'default'}."
        )

    def forward(self, x: Tensor, sigma: Optional[SigmaLike] = None) -> Tensor:
        return x - self.step_size() * self.grad(x, sigma=sigma)

    def integrate_activation(
        self,
        x: Tensor,
        sigma_map: Optional[Tensor] = None,
    ) -> Tensor:
        if self.scaling:
            if sigma_map is None:
                raise ValueError("sigma_map is required when scaling=True.")
            scaling = self.get_scaling(sigma_map)
            x = x * scaling
            y = self.activation.integrate(x)
            return y / scaling.square()
        return self.activation.integrate(x)

    def potential(self, x: Tensor, sigma: Optional[SigmaLike] = None) -> Tensor:
        sigma_map = self._canonical_sigma(sigma, x)
        y = self._analysis_input(x, sigma_map)

        with parametrize.cached():
            y = self.conv_layer.convolution(y)
            if self.bias is not None:
                y = y + self.bias.reshape(1, -1, 1, 1, 1)
            y = self.integrate_activation(y, sigma_map=sigma_map)

        potential_main = y.sum(dim=tuple(range(1, y.ndim)))
        reg_term = 0.5 * self.effective_tau * x.flatten(1).square().sum(dim=1)
        return potential_main + reg_term

    @torch.no_grad()
    def contraction_factor(self) -> Tensor:
        tau = self.effective_tau
        gamma = self.step_size()
        value = 1.0 - 2.0 * gamma * tau * (1.0 + tau) / (1.0 + 2.0 * tau)
        return value.clamp_min(0.0).sqrt()


CCD = CCD3DV2

__all__ = ["CCD3DV2", "CCD"]
