"""Load an isolated V3 spline-01 experiment configuration."""
from pathlib import Path
from copy import deepcopy
import yaml
from model_config import make_model_config

def load_experiment_config(settings_path):
    f = Path(settings_path).expanduser().resolve()
    overrides = yaml.safe_load(f.read_text())
    e = overrides['experiment']
    base = (f.parent / overrides['base_config']).resolve()
    cfg = yaml.safe_load(base.read_text())
    repo = f.parent
    # Use absolute output paths to prevent relative resolution surprises.
    p = cfg['paths']
    for dst, key in [('checkpoint_dir','checkpoint_dir'),('inference_checkpoint','inference_checkpoint'),('training_output_dir','training_output_dir')]:
        p[dst] = str((repo/e[key]).resolve())
    p['checkpoint'] = p['inference_checkpoint']
    p['ccd_checkpoint'] = p['inference_checkpoint']
    mc = cfg['denoiser_model']
    if mc['model_variant'] != 'ccd2':
        raise ValueError('Expected CCD2 model')
    sc = mc['param_spline_scaling']
    sc['x_min'] = float(e['spline_x_min'])
    sc['x_max'] = float(e['spline_x_max'])
    sc['num_knots'] = int(e['spline_num_knots'])
    mc['kwargs']['spline_input_multiplier'] = float(e['spline_input_multiplier'])
    expected = cfg['ccd_expected']['scaling']
    expected.update(x_min=sc['x_min'], x_max=sc['x_max'], num_knots=sc['num_knots'])
    cfg['test_red_params']['ccd'] = {'lambda_red':float(e['lambda_red']), 'strength':float(e['prior_strength_255'])}
    hi = float(cfg['denoiser_training']['sigma_max_255']) / 255 * float(e['spline_input_multiplier'])
    if not (sc['x_min'] <= 0 <= hi <= sc['x_max']):
        raise ValueError(f'Spline input range [0,{hi:g}] not contained in [{sc["x_min"]},{sc["x_max"]}]')
    if not (cfg['denoiser_training']['sigma_min_255'] <= e['prior_strength_255'] <= cfg['denoiser_training']['sigma_max_255']):
        raise ValueError('Prior strength outside training noise range')
    return cfg, e, make_model_config(mc), repo
