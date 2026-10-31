"""Build CCD3D models directly from the project YAML model section.

The project config now exposes the full analysis-channel schedule and both
spline parameter dictionaries directly.  Nothing about the W1/W2/W3 widths,
main activation grid, or noise-scaling grid is hidden in this helper.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping


_SUPPORTED_VARIANTS = {"ccd", "ccd2"}


def _require_mapping(parent: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"denoiser_model.{key} must be a mapping.")
    return deepcopy(value)


def make_model_config(model_section: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert ``denoiser_model`` from config.yml into a checkpointable config."""
    if not isinstance(model_section, Mapping):
        raise TypeError("model_section must be a mapping.")

    model_variant = str(model_section.get("model_variant", "ccd2")).lower()
    if model_variant not in _SUPPORTED_VARIANTS:
        raise ValueError(
            f"model_variant must be one of {sorted(_SUPPORTED_VARIANTS)}."
        )

    param_multi_conv = _require_mapping(model_section, "param_multi_conv")
    param_spline_activation = _require_mapping(
        model_section, "param_spline_activation"
    )
    param_spline_scaling = _require_mapping(
        model_section, "param_spline_scaling"
    )
    kwargs = _require_mapping(model_section, "kwargs")

    channels = list(param_multi_conv.get("num_channels", ()))
    kernels = list(param_multi_conv.get("size_kernels", ()))
    if len(channels) < 2:
        raise ValueError(
            "param_multi_conv.num_channels must contain input and output widths."
        )
    if len(kernels) != len(channels) - 1:
        raise ValueError(
            "param_multi_conv.size_kernels must have "
            "len(num_channels)-1 entries."
        )
    if any(int(c) <= 0 for c in channels):
        raise ValueError("All analysis-channel widths must be positive.")

    # Today's design choice: W receives the one-channel h volume only.
    # Sigma is still supplied to the model, but only through nu(sigma).
    if int(channels[0]) != 1:
        raise ValueError(
            "The current perfusion denoiser design requires "
            "param_multi_conv.num_channels[0] == 1."
        )
    if not bool(kwargs.get("blind", False)):
        raise ValueError(
            "Set kwargs.blind=true so W receives h only (no sigma input channel). "
            "Noise awareness is retained through kwargs.scaling=true."
        )
    if not bool(kwargs.get("scaling", False)):
        raise ValueError(
            "Set kwargs.scaling=true to use the paper-style nu(sigma) pathway."
        )

    return {
        "model_variant": model_variant,
        "param_multi_conv": param_multi_conv,
        "param_spline_activation": param_spline_activation,
        "param_spline_scaling": param_spline_scaling,
        "kwargs": kwargs,
    }


def build_model(model_config: Mapping[str, Any]):
    """Instantiate the selected CCD3D variant from a saved model config."""
    cfg = deepcopy(dict(model_config))
    variant = str(cfg.pop("model_variant", "ccd2")).lower()
    kwargs = cfg.pop("kwargs", {})

    if variant == "ccd":
        from core.models.ccd_3d import CCD3D as ModelClass
    elif variant == "ccd2":
        from core.models.ccd_v2_3d import CCD3DV2 as ModelClass
    else:
        raise ValueError(f"Unsupported model_variant: {variant}")

    return ModelClass(**cfg, **kwargs)


__all__ = ["make_model_config", "build_model"]
