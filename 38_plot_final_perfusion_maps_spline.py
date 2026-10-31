#!/usr/bin/env python3
"""Compact perfusion maps, ONE column wide: GT | RED-CCD (h>=0) | oSVD.

Three rows: CBF, CBV, MTT. Same GT-derived color scale within every row.
GT-defined core outlines. No map smoothing, interpolation or altered metrics.
Only saved predictions are read; no RED reconstruction or training is run.
Exports unique 3x3 PNG and PDF filenames, preserving all earlier figures.
"""
from __future__ import annotations
import argparse
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

p = import_module('35_paper_figure_io')
from synthetic_utils import perfusion_maps_from_h

METHODS = ('GT', 'ccd', 'osvd')
METRICS = ('CBF', 'CBV', 'MTT')


def gt_mask_crop(mask: np.ndarray, pad: int):
    r, c = np.where(mask)
    if r.size == 0:
        raise ValueError('Empty GT brain mask')
    y0, y1 = max(int(r.min())-pad, 0), min(int(r.max())+pad+1, mask.shape[0])
    x0, x1 = max(int(c.min())-pad, 0), min(int(c.max())+pad+1, mask.shape[1])
    return np.s_[y0:y1, x0:x1]


def absrel_percent(x, y, mask, eps):
    numer = np.abs(np.asarray(x, dtype=np.float64)[mask] -
                   np.asarray(y, dtype=np.float64)[mask])
    denom = np.maximum(np.abs(np.asarray(y, dtype=np.float64)[mask]), eps)
    return 100. * float(np.mean(numer/denom))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ccd-version', choices=['previous', 'spline01', 'spline01_tuned'], default='spline01_tuned')
    ap.add_argument('--case-index', type=int, default=4)
    ap.add_argument('--family', choices=p.FAMILIES, default='biexponential')
    ap.add_argument('--dpi', type=int, default=350)
    ap.add_argument('--color-lo', type=float, default=1.0)
    ap.add_argument('--color-hi', type=float, default=99.0)
    ap.add_argument('--crop-padding', type=int, default=1)
    ap.add_argument('--no-tight-crop', action='store_true')
    ap.add_argument('--no-core-outline', action='store_true')
    ap.add_argument('--width-inch', type=float, default=3.5,
                    help='Single IEEE column width in inches (default 3.5)')
    ap.add_argument('--height-inch', type=float, default=3.55)
    args = ap.parse_args()
    if not 0 <= args.color_lo < args.color_hi <= 100:
        ap.error('Need 0 <= color-lo < color-hi <= 100')
    if args.crop_padding < 0 or args.width_inch <= 0 or args.height_inch <= 0:
        ap.error('Invalid cropping or figure size')

    cfg = p.load_config()
    idx = p.choose_case(cfg, args.family, args.case_index)
    case = p.get_case(cfg, 'baseline', idx)
    mask = np.asarray(case['mask']) > 0
    labels = np.asarray(case['tissue_label'])
    lesion = (labels == 3) & mask
    crop = (np.s_[:, :] if args.no_tight_crop else
            gt_mask_crop(mask, args.crop_padding))
    cropped_mask, cropped_core = mask[crop], lesion[crop]
    dt = float(np.diff(np.asarray(case['t'], dtype=float))[0])
    hscale = float(case['h_scale'])
    gt = {'CBF': np.asarray(case['cbf_gt']),
          'CBV': np.asarray(case['cbv_gt']),
          'MTT': np.asarray(case['mtt_gt'])}
    estimates = {}
    errors = {}
    eps = float(cfg['evaluation'].get('eps_relative', 1e-8))
    for method in METHODS[1:]:
        h = p.get_prediction(cfg, 'baseline', method, idx, case, args.ccd_version)
        cbf, cbv, mtt = perfusion_maps_from_h(
            h, h_scale=hscale, dt=dt, mask=case['mask'],
            nonnegative_for_maps=True)
        estimates[method] = dict(zip(METRICS, (cbf, cbv, mtt)))
        errors[method] = {
            metric: absrel_percent(estimates[method][metric], gt[metric], mask, eps)
            for metric in METRICS
        }

    fig = plt.figure(figsize=(args.width_inch, args.height_inch))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, .04],
                          left=.065, right=.920, bottom=.055, top=.93,
                          wspace=.085, hspace=.29)
    for i, metric in enumerate(METRICS):
        vmin, vmax = np.percentile(gt[metric][mask],
                                   [args.color_lo, args.color_hi])
        vmin, vmax = float(vmin), float(vmax)
        if vmax <= vmin:
            vmax = vmin + 1.e-5
        norm = Normalize(vmin=vmin, vmax=vmax)
        for j, method in enumerate(METHODS):
            ax = fig.add_subplot(gs[i, j])
            arr = gt[metric] if method == 'GT' else estimates[method][metric]
            shown = np.ma.array(arr[crop], mask=~cropped_mask)
            im = ax.imshow(shown, cmap='viridis', norm=norm,
                           interpolation='nearest', origin='upper')
            if cropped_core.any() and not args.no_core_outline:
                ax.contour(cropped_core.astype(float), levels=[.5],
                           colors=['white'], linewidths=.43)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                head = ('GT' if method == 'GT' else
                        'RED-CCD' if method == 'ccd' else 'oSVD (20%)')
                ax.set_title(head, fontsize=7.0, weight='semibold', pad=2.0)
            if j == 0:
                ax.set_ylabel(metric, fontsize=8.3, weight='semibold', labelpad=2.2)
            else:
                ax.set_xlabel(f'AbsRel {errors[method][metric]:.2f}%',
                              fontsize=6.4, labelpad=1.4)
        cax = fig.add_subplot(gs[i, 3])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=5.8, length=1.5, pad=1.5)
        cb.set_ticks(np.linspace(vmin, vmax, 3))
        cb.ax.set_yticklabels([f'{v:.0f}' if metric != 'CBV' else f'{v:.1f}'
                               for v in np.linspace(vmin, vmax, 3)])
        # Units are specified in the LaTeX caption to avoid cramped colorbar titles.

    stem = (f'maps_{args.family}_case_{idx:06d}_{args.ccd_version}'
            '_gt_ccd_osvd_single_column')
    output = p.outdir() / f'{stem}.png'
    p.save_both(fig, output, args.dpi)
    print('CASE:', case['_source_name'], '|', args.family)
    print('Saved 3x3 maps at %.2f x %.2f inches' %
          (args.width_inch, args.height_inch))
    print('GT-derived color limits per parameter; GT-derived lesion contour.')
    for metric in METRICS:
        print(f'{metric}: RED-CCD {errors["ccd"][metric]:.3f}% | '
              f'oSVD {errors["osvd"][metric]:.3f}%')


if __name__ == '__main__':
    main()
