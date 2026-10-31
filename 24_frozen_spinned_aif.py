#!/usr/bin/env python3
"""Frozen nominal-AIF SPINNED-style direct-fit network across four AIFs.
No retraining, no physics-consistency refinement, no benchmark overwrites.
Run from ~/ISBI_2027_Final/ alongside synthetic_utils.py.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from synthetic_utils import (load_yaml, load_case, list_split, resolve_device,
                             resolve_repo_path, load_spinned_model,
                             reconstruct_spinned, evaluate_case)

CONDITIONS = ('baseline', 'delayed', 'broad', 'sharp')
METRICS = ('cbf_absrel_percent', 'cbv_absrel_percent', 'mtt_absrel_percent', 'h_nrmse')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No rows to write')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def data_paths(repo, cfg, condition):
    if condition == 'baseline':
        root = resolve_repo_path(cfg['paths']['data_root'])
    else:
        root = repo / 'outputs/v3/aif_shift/datasets' / condition
    return list_split(root, 'test')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config_v3_model_consistent.yml')
    ap.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    ap.add_argument('--max-cases', type=int, default=None)
    ap.add_argument('--out-root', default='outputs/v3/aif_extension_tests/spinned_frozen')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent
    cfg = load_yaml(repo / args.config)
    device = resolve_device(cfg.get('device', 'auto'))
    cfg['methods']['spinned']['enabled'] = True
    checkpoint = resolve_repo_path(cfg['methods']['spinned']['checkpoint'])
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Frozen SPINNED-style checkpoint absent: {checkpoint}')
    model = load_spinned_model(cfg, device)
    model.eval()
    outroot = (repo / args.out_root).resolve()
    print('FROZEN model:', checkpoint, '| device:', device, flush=True)
    for condition in args.conditions:
        paths = data_paths(repo, cfg, condition)
        if args.max_cases is not None:
            paths = paths[:args.max_cases]
        if not paths:
            raise RuntimeError(f'No cases for {condition}')
        out = outroot / condition / (f'smoke_{len(paths)}' if args.max_cases is not None else 'full')
        if (out / 'per_case.csv').exists() and not args.overwrite:
            raise FileExistsError(f'{out}/per_case.csv exists; use --overwrite intentionally')
        rows = []
        for i, file in enumerate(paths):
            case = load_case(file)
            start = time.perf_counter()
            pred = reconstruct_spinned(case, model, device)
            if device.type == 'cuda':
                import torch
                torch.cuda.synchronize(device)
            runtime = time.perf_counter() - start
            if pred.shape != case['h_gt'].shape or not np.isfinite(pred).all():
                raise RuntimeError(f'Invalid prediction for {file.name}: {pred.shape}')
            metrics = evaluate_case(pred, case, eps_relative=float(cfg['evaluation']['eps_relative']))
            rows.append({'case_index':i, 'case_file':file.name,
                         'family':str(np.asarray(case['residue_family']).item()),
                         'condition': condition, 'method':'spinned_style_frozen',
                         'runtime_seconds':runtime, **metrics})
            if (i+1) % 10 == 0 or i+1 == len(paths):
                print(f'{condition}: {i+1}/{len(paths)}', flush=True)
        write_csv(out/'per_case.csv', rows)
        summary = {'condition': condition, 'method':'spinned_style_frozen', 'n_cases':len(paths),
                   'checkpoint': str(checkpoint), 'inference':'direct_fit_no_refinement',
                   'training':'none_in_this_experiment'}
        for metric in METRICS:
            v=np.array([float(r[metric]) for r in rows])
            summary[metric+'_mean']=float(v.mean())
            summary[metric+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.0
        summary['map_mean_absrel_percent']=float(np.mean([summary[x+'_mean'] for x in METRICS[:3]]))
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print(condition, 'map average:', f"{summary['map_mean_absrel_percent']:.3f}%", 'hNRMSE:',
              f"{summary['h_nrmse_mean']:.5f}", 'saved', out)


if __name__ == '__main__':
    main()
