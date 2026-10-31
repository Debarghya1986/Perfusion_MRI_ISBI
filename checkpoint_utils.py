from __future__ import annotations
from pathlib import Path
import torch
from model_config import build_model


def load_frozen_ccd3d(path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if "model_config" not in ckpt:
        raise KeyError("Inference checkpoint is missing model_config.")
    model = build_model(ckpt["model_config"]).to(device)
    state = ckpt.get("model_state", ckpt.get("model_state_dict"))
    if state is None:
        raise KeyError("Checkpoint has no model state.")
    model.load_state_dict(state)
    model.project_parameters()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def validate_sigma_255(sigma_255: float, ckpt, name="sigma"):
    lo = ckpt.get("training_sigma_min_255", None)
    hi = ckpt.get("training_sigma_max_255", None)
    if lo is None and "training_sigma_min" in ckpt:
        lo = float(ckpt["training_sigma_min"]) * 255.0
    if hi is None and "training_sigma_max" in ckpt:
        hi = float(ckpt["training_sigma_max"]) * 255.0
    if lo is not None and hi is not None:
        if not (float(lo) <= float(sigma_255) <= float(hi)):
            raise ValueError(
                f"{name}={sigma_255:g}/255 is outside the denoiser training "
                f"range [{float(lo):g},{float(hi):g}]/255."
            )
    return lo, hi


__all__ = ["load_frozen_ccd3d", "validate_sigma_255"]
