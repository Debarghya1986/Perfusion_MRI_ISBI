#!/usr/bin/env python3
"""Generate paired AIF-shifted Synthetic V3 test/validation cases.

Only the gamma-variate AIF and its consequent forward measurements change.
GT response, tissue labels, geometry and perfusion maps are copied unchanged.
The default reuses each saved case's original MR-signal noise realization,
original kappa, S0 and TE. No training data or checkpoints are changed.

AIFs in the ORIGINAL generator use:
  x = t-arrival_d, z=x/tmax,
  a(t) = ymax*z**alpha*exp(alpha*(1-z)) (for x>0).
Hence the nominal peak is at t = arrival_d + tmax, not at tmax alone.

Scenarios (fixed ymax, dt=1s, duration=56s):
  delayed: arrival_d=5, tmax=4.5, alpha=3
  broad:   arrival_d=3, tmax=6.0, alpha=2
  sharp:   arrival_d=3, tmax=3.5, alpha=4
Note broad/sharp also change peak time: these are combined AIF-shape stress
conditions, not controlled width-only changes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

SCENARIOS = {
    'delayed': {'arrival_d': 5.0, 'tmax': 4.5, 'alpha': 3.0},
    'broad': {'arrival_d': 3.0, 'tmax': 6.0, 'alpha': 2.0},
    'sharp': {'arrival_d': 3.0, 'tmax': 3.5, 'alpha': 4.0},
}


def import_generator(repo: Path):
    filename = repo / '01_generate_synthetic_v3.py'
    if not filename.is_file():
        raise FileNotFoundError(f'Missing original V3 generator: {filename}')
    spec = importlib.util.spec_from_file_location('synthetic_v3_aif_source', filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='config_v3_model_consistent.yml')
    p.add_argument('--scenarios', nargs='+', choices=[*SCENARIOS, 'baseline'], default=list(SCENARIOS))
    p.add_argument('--split', choices=['test', 'val'], default='test')
    p.add_argument('--out-root', default='outputs/v3/aif_shift/datasets')
    p.add_argument('--noise-mode', choices=['paired-original', 'fresh'], default='paired-original')
    p.add_argument('--noise-seed', type=int, default=20271021)
    p.add_argument('--max-cases', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        p.error('--max-cases must be positive')

    repo = Path(__file__).resolve().parent
    cfg_path = (repo / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    dcfg = cfg['dataset']
    source_root = (repo / cfg['paths']['data_root']).resolve() / args.split
    source_files = sorted(source_root.glob('sample_*.npz'))
    if args.max_cases is not None:
        source_files = source_files[:args.max_cases]
    if not source_files:
        raise RuntimeError(f'No source cases found in {source_root}')
    out_base = (repo / args.out_root).resolve()
    gen = import_generator(repo)
    print(f'Using {len(source_files)} original {args.split} cases; denoiser training stays unchanged.')

    for condition in args.scenarios:
        out_dir = out_base / condition / args.split
        out_dir.mkdir(parents=True, exist_ok=True)
        aif_params = dict(dcfg['aif'])
        if condition != 'baseline':
            aif_params.update(SCENARIOS[condition])
        signature = {
            'condition': condition,
            'split': args.split,
            'source_data_root': str(source_root),
            'source_config': str(cfg_path),
            'aif': aif_params,
            'noise': dict(dcfg['noise']),
            'noise_mode': args.noise_mode,
            'noise_seed': args.noise_seed,
            'source_files': [x.name for x in source_files],
        }
        manifest = out_dir.parent / f'manifest_{args.split}.json'
        if manifest.exists() and not args.overwrite:
            old = json.loads(manifest.read_text())
            for key in ['condition', 'split', 'source_data_root', 'aif', 'noise', 'noise_mode', 'noise_seed']:
                if old.get(key) != signature[key]:
                    raise RuntimeError(f'{manifest} mismatch ({key}); use --overwrite only deliberately.')

        n_written = 0
        for i, src in enumerate(source_files):
            dst = out_dir / src.name
            if dst.exists() and not args.overwrite:
                continue
            with np.load(src, allow_pickle=False) as f:
                orig = {k: np.array(f[k], copy=True) for k in f.files}
            if condition == 'baseline':
                result = orig  # exact original baseline measurement
            else:
                h_net = np.transpose(orig['clean'], (1, 2, 0)).astype(np.float32)
                H, W, T = h_net.shape
                t = np.asarray(orig['t'], dtype=np.float32)
                dt = float(t[1] - t[0])
                mask = np.asarray(orig['brain_mask'], dtype=np.float32)
                h_scale = float(np.asarray(orig['h_scale']).reshape(()))
                h_phys = h_net * h_scale
                aif = gen.gamma_variate_aif(t, {'aif': aif_params})
                A = gen.conv_matrix(aif, dt)
                C_clean_phys = (h_phys.reshape(-1, T) @ A.T).reshape(H, W, T).astype(np.float32)

                if args.noise_mode == 'paired-original':
                    for key in ('signal_kappa', 'C_clean', 'C_obs', 'signal_noise_sd'):
                        if key not in orig:
                            raise RuntimeError(f'{src}: missing {key} for paired MR noise')
                    S0 = float(dcfg['noise']['S0'])
                    TE = float(dcfg['noise']['TE_s'])
                    kappa = float(np.asarray(orig['signal_kappa']).reshape(()))
                    C_old_clean = np.asarray(orig['C_clean'], np.float64) * h_scale
                    C_old_obs = np.asarray(orig['C_obs'], np.float64) * h_scale
                    eta_original = (S0 * np.exp(-kappa * TE * C_old_obs)
                                    - S0 * np.exp(-kappa * TE * C_old_clean))
                    S_new_clean = S0 * np.exp(-kappa * TE * C_clean_phys.astype(np.float64))
                    S_new_obs = np.maximum(S_new_clean + eta_original, 1e-4)
                    C_obs_phys = -np.log(S_new_obs / S0) / (kappa * TE)
                    C_obs_phys[mask == 0] = 0.0
                    C_obs_phys = C_obs_phys.astype(np.float32)
                    sigma_S = float(np.asarray(orig['signal_noise_sd']).reshape(()))
                else:
                    rng = np.random.default_rng(args.noise_seed + i + (1_000_000 if args.split == 'val' else 0))
                    C_obs_phys, kappa, sigma_S = gen.signal_domain_noise(
                        C_clean_phys, mask, {'noise': dcfg['noise']}, rng)

                result = dict(orig)
                result.update({
                    'aif': aif.astype(np.float32),
                    'A': A.astype(np.float32),
                    'C_clean': (C_clean_phys / h_scale).astype(np.float32),
                    'C_obs': (C_obs_phys / h_scale).astype(np.float32),
                    'signal_kappa': np.asarray(kappa, dtype=np.float32),
                    'signal_noise_sd': np.asarray(sigma_S, dtype=np.float32),
                    'aif_shift_scenario': np.asarray(condition),
                })
                for key in ('clean', 'cbf_gt', 'cbv_gt', 'mtt_gt', 'brain_mask', 'tissue_label'):
                    if not np.array_equal(result[key], orig[key]):
                        raise AssertionError(f'Unexpected change in ground truth: {key}')
            np.savez_compressed(dst, **result)
            n_written += 1
            if (i+1) % 20 == 0 or i+1 == len(source_files):
                print(f'  {condition}: {i+1}/{len(source_files)} checked', flush=True)

        signature['source_files'] = sorted(f.name for f in out_dir.glob('sample_*.npz'))
        manifest.write_text(json.dumps(signature, indent=2))
        with np.load(out_dir / source_files[0].name, allow_pickle=False) as first:
            print(f'{condition}: files={len(signature["source_files"])} '
                  f'A={first["A"].shape} peak={float(first["aif"].max()):.5f} '
                  f'newly_generated={n_written}')
        print(f'Saved to {out_dir}')


if __name__ == '__main__':
    main()
