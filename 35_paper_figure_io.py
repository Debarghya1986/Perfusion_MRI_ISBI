#!/usr/bin/env python3
"""Shared saved-case/prediction readers for paper-figure scripts 36-38.
Only load images for selected cases; never compute/overwrite reconstructions.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parent
CONDITIONS = ('baseline','delayed','broad','sharp')
FAMILIES = ('biexponential','pharmacokinetic')


def load_config(repo=ROOT):
    from synthetic_utils import load_yaml
    return load_yaml(repo/'config_v3_model_consistent.yml')


def paths_for_condition(cfg, condition: str, repo=ROOT):
    from synthetic_utils import list_split, resolve_repo_path
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown AIF condition: {condition}')
    root = (resolve_repo_path(cfg['paths']['data_root']) if condition == 'baseline'
            else repo/'outputs/v3/aif_shift/datasets'/condition)
    paths = list_split(root, 'test')
    if not paths:
        raise FileNotFoundError(f'No test cases at {root}/test')
    return paths


def get_case(cfg, condition: str, index: int, repo=ROOT):
    from synthetic_utils import load_case
    paths=paths_for_condition(cfg, condition, repo)
    if not 0 <= index < len(paths):
        raise IndexError(f'Case index {index} outside 0..{len(paths)-1}')
    c=load_case(paths[index]); c['_source_name']=paths[index].name
    return c


def family_of(case):
    return str(np.asarray(case['residue_family']).item())


def validate_family(case, expected, index):
    found=family_of(case)
    if expected != found:
        raise ValueError(f'Case {index} has family={found}, expected={expected}. '
                         'Choose an index from the requested response family.')


def choose_case(cfg, family, selected_index, repo=ROOT):
    if selected_index is not None:
        validate_family(get_case(cfg, 'baseline', selected_index, repo), family, selected_index)
        return selected_index
    from synthetic_utils import load_case
    paths=paths_for_condition(cfg, 'baseline', repo)
    pool=[]
    for i,p in enumerate(paths):
        c=load_case(p)
        if family_of(c)!=family: continue
        m=np.asarray(c['mask'])>0
        if m.any(): pool.append((i,float(np.asarray(c['cbf_gt'])[m].mean())))
    if not pool: raise ValueError(f'No test cases of family={family}')
    target=float(np.median([a[1] for a in pool]))
    return min(pool, key=lambda a:(abs(a[1]-target),a[0]))[0]


def get_voxel(case,region='core'):
    labels=np.asarray(case['tissue_label'])
    codes={'gm':1,'wm':2,'core':3}
    if region not in codes: raise ValueError(f'--region must be one of {tuple(codes)}')
    coords=np.argwhere(labels==codes[region]); cbf=np.asarray(case['cbf_gt'])
    if not len(coords): raise ValueError(f'No {region} tissue voxels in selected case')
    values=cbf[labels==codes[region]]; target=float(np.median(values))
    y,x=map(int,coords[np.argmin(np.abs(values-target))]);return (y,x)


def _read_npz(path,key,index,expected_shape):
    if not path.is_file(): return None
    with np.load(path,allow_pickle=False) as npz:
        if key not in npz.files:
            raise KeyError(f'{path}: key {key!r} missing, found {npz.files}')
        full=npz[key]
        if full.ndim!=4 or not (0<=index<full.shape[0]):
            raise ValueError(f'{path}[{key}] expected [N,H,W,T] including index {index}, got {full.shape}')
        one=np.asarray(full[index],dtype=np.float32)
    if one.shape!=expected_shape or not np.isfinite(one).all():
        raise ValueError(f'{path}[{key}]: prediction shape or finite check failed {one.shape} != {expected_shape}')
    print(f'  {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} [{key}] case={index}',flush=True)
    return one


def get_prediction(cfg,condition,method,index,case,ccd_version='previous',repo=ROOT):
    """method: ccd, tv, frindel, osvd, bm4d. CCD is always nonnegative. BM4D is original, unconstrained."""
    shape=np.asarray(case['h_gt']).shape
    if method=='ccd':
        if ccd_version=='previous':
            p=repo/'outputs/v3/aif_extension_tests/nonnegative_red'/condition/'ccd/full/predictions.npz'
        elif ccd_version=='spline01':
            # Earlier untuned spline experiment. Retained for reproducibility.
            p=repo/'outputs/v3/spline01/nonnegative_red'/condition/'ccd/full/predictions.npz'
        elif ccd_version=='spline01_tuned':
            # Corrected scaling denominator and nominal-validation-selected RED parameters.
            p=repo/'outputs/v3/spline01_tuned/nonnegative_red'/condition/'ccd/full/predictions.npz'
            summary=p.with_name('summary.json')
            if not summary.is_file():
                raise FileNotFoundError(f'Missing tuned experiment summary: {summary}')
            import json
            with summary.open('r') as f:
                run=json.load(f)
            if (not np.isclose(float(run.get('lambda_red', -1)), 1.6) or
                not np.isclose(float(run.get('strength', -1)), 1.0) or
                int(run.get('n_cases', -1)) != 120):
                raise RuntimeError(f'Expected tuned CCD lambda=1.6, strength=1.0, 120 cases; found {run}')
            csvpath=p.with_name('per_case.csv')
            if not csvpath.is_file():
                raise FileNotFoundError(f'Missing tuned per-case metadata: {csvpath}')
            import csv
            with csvpath.open(newline='') as f:
                rows=list(csv.DictReader(f))
            if (len(rows)!=120 or
                rows[index].get('case_file')!=case.get('_source_name')):
                raise RuntimeError(f'Case identity mismatch in {csvpath} at index {index}')
        else: raise ValueError(f'Unknown CCD version {ccd_version}')
        key='prediction'
    elif method=='tv':
        p=repo/'outputs/v3/aif_extension_tests/nonnegative_red'/condition/'tv/full/predictions.npz'
        key='prediction'
    elif method in ('frindel','osvd','bm4d'):
        if condition != 'baseline':
            raise ValueError(f'{method} supported only for baseline in these paper figures')
        p=repo/'outputs/v3/all_methods_test/predictions_final_initC.npz'
        key='red_bm4d' if method=='bm4d' else method
        if method=='bm4d':
            # Original unconstrained BM4D. Never substitute missing nonnegative BM4D.
            # The combined final archive should contain 'red_bm4d'; the individual
            # init-C prediction archive is an equally valid original-output fallback.
            if not p.is_file():
                p=repo/'outputs/v3/red_priors_test/bm4d_init_c/predictions.npz'
                key='prediction'
            else:
                with np.load(p,allow_pickle=False) as z:
                    if 'red_bm4d' not in z.files:
                        p=repo/'outputs/v3/red_priors_test/bm4d_init_c/predictions.npz'
                        key='prediction'
    else:
        raise ValueError(f'Unknown method {method}')
    a=_read_npz(p,key,index,shape)
    if a is None:
        raise FileNotFoundError(f'Missing {method} prediction: {p}. '
            'Complete the corresponding reconstruction; the plotting script does not run RED.')
    if method=='ccd' and ccd_version=='spline01_tuned' and float(np.min(a)) < -1e-6:
        raise RuntimeError('Tuned CCD prediction unexpectedly contains negative values')
    return a


class FrozenSpinned:
    """On-demand prediction for display cases only (24 saves metrics but no full .npz)."""
    def __init__(self,cfg,repo=ROOT):
        from synthetic_utils import load_spinned_model, resolve_device
        self.device=resolve_device(cfg.get('device','auto'))
        self.cfg=copy.deepcopy(cfg)
        self.cfg['methods']['spinned']['enabled']=True
        self.model=load_spinned_model(self.cfg,self.device)
        self.repo=repo

    def predict(self,case,condition,index):
        from synthetic_utils import reconstruct_spinned
        result=np.asarray(reconstruct_spinned(case,self.model,self.device),dtype=np.float32)
        if result.shape!=np.asarray(case['h_gt']).shape or not np.isfinite(result).all():
            raise ValueError(f'SPINNED output shape or finite check failed for {condition}/{index}')
        print(f'  frozen SPINNED-style inference: {condition}, case {index}',flush=True)
        return result


def outdir(repo=ROOT):
    p=repo/'outputs/v3/final_paper_candidate_figures'
    p.mkdir(parents=True,exist_ok=True)
    return p


def save_both(fig,path,dpi=300):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=dpi,facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),facecolor='white')
    plt.close(fig)
    print('SAVED:',path,'and PDF',flush=True)

COLORS={'GT':'#222222','ccd':'#087F8C','tv':'#C17432',
        'frindel':'#99569F','osvd':'#D45758','spinned':'#C17432'}
LABELS={'GT':'Ground truth','ccd':'RED-CCD (nonneg.)','tv':'RED-TV (nonneg.)',
        'frindel':'Frindel-ST','osvd':'oSVD (20%)','bm4d':'RED-BM4D (original)','spinned':'SPINNED-style (frozen)'}


def draw_curve(ax,t,h,method):
    color=COLORS[method]; ls='-' if method in ('GT','ccd') else '--'
    lw=2.25 if method=='GT' else 1.9
    ax.plot(t,h,color=color,ls=ls,lw=lw,label=LABELS[method],zorder=4 if method=='GT' else 3)


def style_curve(ax,*,xlabel=False,ylabel=False):
    ax.grid(True,alpha=.16,lw=.65)
    ax.axhline(0,color='#7f7f7f',lw=.65,alpha=.65)
    if xlabel:ax.set_xlabel('Time (s)',fontsize=10)
    if ylabel:ax.set_ylabel('$h(t)$',fontsize=10)
    ax.tick_params(labelsize=8)
