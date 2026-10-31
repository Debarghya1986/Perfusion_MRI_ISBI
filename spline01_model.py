"""CCD3DV2 with *only* its noise-scaling spline coordinate remapped.

Public sigma is the normalized AWGN standard deviation (0..25/255).
The original CCD3DV2 _canonical_sigma() returns sigma255=255*sigma.
This variant evaluates spline_scaling(sigma255 * multiplier/255), while
preserving the original denominator (sigma255 + scaling_eps).
This isolates the spline coordinate/grid ablation and leaves W, activation,
noise synthesis, tau, gamma, and RED's physical sigma convention unchanged.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import torch
from core.models.ccd_v2_3d import CCD3DV2

class CCDSpline01(CCD3DV2):
    def __init__(self, *args, spline_input_multiplier=10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.spline_input_multiplier = float(spline_input_multiplier)
        if self.spline_input_multiplier <= 0:
            raise ValueError('spline_input_multiplier must be > 0')

    def get_scaling(self, sigma_map):
        # sigma_map = 255 * normalized physical sigma, inherited convention.
        if self.spline_scaling is None:
            raise RuntimeError('Scaling spline is disabled')
        spline_coord = sigma_map * (self.spline_input_multiplier / 255.0)
        # Fail rather than silently extrapolate outside the declared spline domain.
        xmin = self.spline_scaling.x_min.to(spline_coord)
        xmax = self.spline_scaling.x_max.to(spline_coord)
        if bool(torch.any(spline_coord < xmin - 1e-5) or torch.any(spline_coord > xmax + 1e-5)):
            raise ValueError('Noise scaling coordinate outside spline range')
        v = spline_coord.expand(sigma_map.shape[0], self.num_channels, *sigma_map.shape[2:])
        return torch.exp(self.spline_scaling(v)) / (
                spline_coord + self.scaling_eps
                )


def build_spline01_model(model_cfg):
    if model_cfg.get('model_variant') != 'ccd2':
        raise ValueError('This experiment requires CCD2')
    k = deepcopy(model_cfg['kwargs'])
    multiplier = k.pop('spline_input_multiplier', None)
    if multiplier is None:
        raise ValueError('Missing spline_input_multiplier: refusing silent stock CCD model')
    return CCDSpline01(
        param_multi_conv=deepcopy(model_cfg['param_multi_conv']),
        param_spline_activation=deepcopy(model_cfg['param_spline_activation']),
        param_spline_scaling=deepcopy(model_cfg['param_spline_scaling']),
        spline_input_multiplier=float(multiplier), **k)


def load_spline01_checkpoint(path, device, expected_multiplier=None):
    # Load only checkpoints produced by this project. PyTorch pickle is not safe for untrusted files.
    ckpt = torch.load(str(Path(path)), map_location='cpu', weights_only=False)
    if ckpt.get('spline01_experiment') != 'sigma_coordinate_remapping_v1':
        raise RuntimeError('Not a spline01 experiment checkpoint')
    model_cfg = ckpt['model_config']
    scale = float(model_cfg['kwargs']['spline_input_multiplier'])
    if expected_multiplier is not None and abs(scale-float(expected_multiplier)) > 1e-10:
        raise RuntimeError('Noise multiplier differs from checkpoint')
    model=build_spline01_model(model_cfg)
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.to(device).eval()
    return model, ckpt
