#!/usr/bin/env python3
"""GT / projected RED-CCD / frozen SPINNED-style for separate AIF conditions.

Plot formatting is intentionally IDENTICAL to
36_plot_nominal_three_baselines.py and the manuscript's 07_visual_reports.py:
  - Each exported figure has two panels: Bi-exponential | Pharmacokinetic.
  - figsize=(9.4, 3.7), sharex=True, fig.tight_layout().
  - Matplotlib default text sizes and matching super-title size 11.
  - Blue solid GT (2.6), orange dashed RED-CCD (2.0),
    green dash-dot SPINNED-style (1.8).
  - Axis labels on BOTH panels; grid alpha=0.20; upper-right legend size 8.

The two AIFs (default nominal and broad) produce TWO separate PNG/PDF files.
CCD is the saved nonnegative RED reconstruction. SPINNED-style uses the
existing frozen checkpoint for ONLY the selected display cases. No training,
RED iterations, or changes to saved predictions are performed.

Requires the unchanged 35_paper_figure_io.py in the same project directory.
"""
from __future__ import annotations

import argparse
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = import_module('35_paper_figure_io')

# Copy the exact defaults explicitly used by script 36.
BLUE = '#1f77b4'
ORANGE = '#ff7f0e'
GREEN = '#008080'  # teal as in the selected old-paper plots
FAMILIES = ('biexponential', 'pharmacokinetic')
FAMILY_TITLE = {'biexponential': 'Bi-exponential',
                'pharmacokinetic': 'Pharmacokinetic'}
CONDITION_TITLE = {'baseline': 'Nominal', 'delayed': 'Delayed',
                   'broad': 'Broad', 'sharp': 'Sharp'}


def original_style_curve(ax, times, gt, ccd, spinned):
    """Same plotting operations and settings as script 36."""
    ax.plot(times, gt, label='Ground truth', color=BLUE, linewidth=2.6)
    ax.plot(times, ccd, label='RED-CCD', color=ORANGE,
            linewidth=2.0, linestyle='--')
    ax.plot(times, spinned, label='SPINNED-style', color=GREEN,
            linewidth=1.8, linestyle='-.')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$h(t)$, physical scale')
    ax.grid(alpha=0.20)
    ax.legend(loc='upper right', fontsize=13)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ccd-version', choices=['previous', 'spline01', 'spline01_tuned'],
                        default='spline01_tuned')
    parser.add_argument('--conditions', nargs=2, choices=p.CONDITIONS,
                        default=['baseline', 'broad'],
                        help='Export two separate figures: one per AIF')
    parser.add_argument('--case-biexp', type=int, default=4)
    parser.add_argument('--case-pk', type=int, default=31)
    parser.add_argument('--region', choices=['core', 'gm', 'wm'], default='core')
    parser.add_argument('--dpi', type=int, default=350)
    args = parser.parse_args()
    if len(set(args.conditions)) != 2:
        parser.error('--conditions requires two different AIF conditions')

    cfg = p.load_config()
    indices = {
        'biexponential': p.choose_case(cfg, 'biexponential', args.case_biexp),
        'pharmacokinetic': p.choose_case(cfg, 'pharmacokinetic', args.case_pk),
    }
    baseline = {fam: p.get_case(cfg, 'baseline', index)
                for fam, index in indices.items()}
    voxels = {fam: p.get_voxel(baseline[fam], args.region)
              for fam in FAMILIES}
    frozen = p.FrozenSpinned(cfg)
    curves = {}

    for condition in args.conditions:
        for family in FAMILIES:
            index = indices[family]
            case = p.get_case(cfg, condition, index)
            reference = baseline[family]
            if case['_source_name'] != reference['_source_name']:
                raise RuntimeError(
                    f'{condition}/{family}: case file does not match nominal AIF: '
                    f'{case["_source_name"]} vs {reference["_source_name"]}')
            if not np.allclose(case['h_gt'], reference['h_gt'],
                               rtol=1e-5, atol=1e-6):
                raise RuntimeError(
                    f'{condition}/{family}: GT tissue response differs from nominal')
            y, x = voxels[family]
            scale = float(case['h_scale'])
            ccd = p.get_prediction(cfg, condition, 'ccd', index,
                                   case, args.ccd_version)
            spinned = frozen.predict(case, condition, index)
            curves[(condition, family)] = {
                't': np.asarray(case['t'], dtype=np.float64),
                'GT': np.asarray(case['h_gt'][y, x, :], dtype=np.float64) * scale,
                'ccd': np.asarray(ccd[y, x, :], dtype=np.float64) * scale,
                'spinned': np.asarray(spinned[y, x, :], dtype=np.float64) * scale,
            }

    for condition in args.conditions:
        # EXACTLY the same axes geometry and figure layout as script 36.
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharex=True)
        for ax, family in zip(axes, FAMILIES):
            d = curves[(condition, family)]
            original_style_curve(ax, d['t'], d['GT'], d['ccd'], d['spinned'])
            ax.set_title(FAMILY_TITLE[family], fontsize=14, fontweight='bold')

        fig.suptitle('RED-CCD vs SPINNED-style '
                     f'({CONDITION_TITLE[condition]} AIF)', fontsize=15, fontweight='bold')
        fig.tight_layout()  # Copy script 36; no manual subplot adjustments.
        path = (p.outdir() / f'aif_{condition}_{args.ccd_version}_vs_spinned_'
                f'{args.region}_oldpaperstyle_two_panel.png')
        p.save_both(fig, path, args.dpi)

    print('Exported two separate AIF figures, each styled identically to script 36.')
    print('Each: 9.4 x 3.7 inches, two panels with their own upper-right legends.')
    print('GT/CCD/SPINNED: blue solid / orange dashed / green dash-dot.')
    print('Checkpoint frozen; no training or RED reconstruction performed.')


if __name__ == '__main__':
    main()
