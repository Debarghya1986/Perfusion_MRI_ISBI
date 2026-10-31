#!/usr/bin/env python3
"""Nominal h(t) comparisons (including original RED-BM4D) in old paper style.

Same old-paper layout: 1x2 figure, original Matplotlib default color order,
GT solid blue / CCD dashed orange / comparator dash-dot green,
inside-axes legends (upper right), physical h(t), labels on BOTH axes,
light grid and a single top-centered title.

Input/output conventions are unchanged from the previous script 36.
No reconstruction, checkpoint training, or map manipulation is performed.
"""
from __future__ import annotations

import argparse
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = import_module('35_paper_figure_io')

# The original plotting function uses default Matplotlib's first three colors.
# Explicitly pin their color values so the appearance is reproducible even if
# an external mplstyle modifies the active color cycle.
BLUE = '#1f77b4'
ORANGE = '#ff7f0e'
GREEN = '#008080'  # teal as in the selected old-paper plots

DISPLAY = {
    'tv': 'RED-TV',
    'frindel': 'Frindel-ST',
    'osvd': 'oSVD (20%)',
    'bm4d': 'RED-BM4D',  # Original unconstrained reconstruction
}
FAMILY_TITLE = {'biexponential': 'Bi-exponential', 'pharmacokinetic': 'Pharmacokinetic'}


def original_style_curve(ax, times, gt, ccd, other, comparator):
    """Replicate 07_visual_reports.py: plot_curves and axes annotations."""
    ax.plot(times, gt, label='Ground truth', color=BLUE, linewidth=2.6)
    ax.plot(times, ccd, label='RED-CCD', color=ORANGE, linewidth=2.0, linestyle='--')
    ax.plot(times, other, label=DISPLAY[comparator], color=GREEN,
            linewidth=1.8, linestyle='-.')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$h(t)$, physical scale')
    ax.grid(alpha=0.20)
    ax.legend(loc='upper right', fontsize=13)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ccd-version', choices=['previous', 'spline01', 'spline01_tuned'], default='spline01_tuned')
    parser.add_argument('--case-biexp', type=int, default=4)
    parser.add_argument('--case-pk', type=int, default=31)
    parser.add_argument('--region', choices=['core', 'gm', 'wm'], default='core')
    parser.add_argument('--baselines', nargs='+', choices=list(DISPLAY), default=list(DISPLAY))
    parser.add_argument('--dpi', type=int, default=350)
    args = parser.parse_args()

    cfg = p.load_config()
    indices = {
        'biexponential': p.choose_case(cfg, 'biexponential', args.case_biexp),
        'pharmacokinetic': p.choose_case(cfg, 'pharmacokinetic', args.case_pk),
    }
    data = {}
    for fam, index in indices.items():
        case = p.get_case(cfg, 'baseline', index)
        y, x = p.get_voxel(case, args.region)
        scale = float(case['h_scale'])  # Original paper uses physical-scale h(t).
        d = {
            't': np.asarray(case['t'], dtype=np.float64),
            'GT': np.asarray(case['h_gt'][y, x, :], dtype=np.float64) * scale,
            'ccd': p.get_prediction(cfg, 'baseline', 'ccd', index, case, args.ccd_version)[y, x, :].astype(np.float64) * scale,
        }
        for method in args.baselines:
            d[method] = p.get_prediction(cfg, 'baseline', method, index, case, args.ccd_version)[y, x, :].astype(np.float64) * scale
        data[fam] = d

    output = p.outdir()
    for method in args.baselines:
        # This is the original 07_visual_reports.py figsize / subplot geometry.
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharex=True)
        for ax, (family, d) in zip(axes, data.items()):
            original_style_curve(ax, d['t'], d['GT'], d['ccd'], d[method], method)
            ax.set_title(FAMILY_TITLE[family], fontsize=14, fontweight='bold')
        fig.suptitle('RED-CCD vs ' + DISPLAY[method], fontsize=15, fontweight='bold')
        fig.tight_layout()  # Matches original; no figure-level legend or annotations.
        path = output / f'nominal_{args.ccd_version}_ccd_vs_{method}_{args.region}_oldpaperstyle.png'
        p.save_both(fig, path, args.dpi)
    print('Finished: CCD is nonnegative; BM4D is the original unconstrained baseline.')


if __name__ == '__main__':
    main()
