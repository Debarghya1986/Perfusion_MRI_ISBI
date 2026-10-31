#!/usr/bin/env python3
"""Nonnegative RED-CCD with the separately trained spline01 denoiser on three AIFs."""
from __future__ import annotations
import argparse,csv,json,importlib.util
from pathlib import Path
import numpy as np
import torch
from spline01_utils import load_experiment_config
from spline01_model import load_spline01_checkpoint
from synthetic_utils import (load_case, list_split, stack_cases, batch_paths,
                             evaluate_case,resolve_device)

METRICS=('cbf_absrel_percent','cbv_absrel_percent','mtt_absrel_percent','h_nrmse')
CONDITIONS=('baseline','delayed','broad')

class SplinePrior:
    def __init__(self, model, checkpoint):
        self.model=model
        self.checkpoint=checkpoint
        self.sigma_min=float(checkpoint['training_sigma_min_255'])
        self.sigma_max=float(checkpoint['training_sigma_max_255'])
    @torch.inference_mode()
    def __call__(self,x_hwt,strength):
        v=float(strength)
        if not (self.sigma_min<=v<=self.sigma_max):
            raise ValueError('RED strength outside trained spline noise range')
        inp=x_hwt.permute(0,3,1,2).unsqueeze(1).contiguous()
        sigma=torch.full((inp.shape[0],),v/255,dtype=inp.dtype,device=inp.device)
        res=self.model(inp,sigma)
        return res[:,0].permute(0,2,3,1).contiguous()

def import_projected_solver(repo):
    f=repo/'25_projected_red_aif.py'
    spec=importlib.util.spec_from_file_location('project_existing_projected_red',f)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.projected_red_batch

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config_v3_spline01.yml')
    parser.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS))
    parser.add_argument('--max-cases',type=int,default=None,help='Smoke only, defaults to all 120 held-out cases')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--max-iter',type=int,default=None,help='Smoke only; default uses original 5000')
    parser.add_argument('--overwrite',action='store_true',help='Allow replacing existing spline01 predictions')
    parser.add_argument('--dry-run',action='store_true',help='Check paths and checkpoint without reconstructing')
    args=parser.parse_args()
    if args.batch_size<1 or (args.max_cases is not None and args.max_cases<1):parser.error('Positive batch size/case count required')
    cfg,e,model_config,repo=load_experiment_config(args.config)
    ckpt_file=Path(cfg['paths']['inference_checkpoint'])
    if not ckpt_file.is_file(): raise FileNotFoundError(f'Train first: {ckpt_file}')
    device=resolve_device(cfg.get('device','auto'))
    model, ckpt=load_spline01_checkpoint(ckpt_file,device,expected_multiplier=e['spline_input_multiplier'])
    # Verify the entire learned architecture and spline specification.
    if ckpt['model_config'] != model_config:
        raise RuntimeError('Training checkpoint architecture/configuration differs from current spline experiment settings')
    prior=SplinePrior(model,ckpt)
    solve=import_projected_solver(repo)
    redcfg=cfg['red']; params=cfg['test_red_params']['ccd']
    lam=float(params['lambda_red']); strength=float(params['strength'])
    max_iter=int(args.max_iter or redcfg['max_iter'])
    for condition in args.conditions:
        data_root=(repo/cfg['paths']['data_root']) if condition=='baseline' else repo/'outputs/v3/aif_shift/datasets'/condition
        paths=list_split(data_root,'test')
        if args.max_cases is None and len(paths)!=120:
            raise RuntimeError(f'{condition}: expected 120 test cases, found {len(paths)}')
        if args.max_cases is not None:paths=paths[:args.max_cases]
        if not paths:raise FileNotFoundError(f'No test cases for {condition} under {data_root}')
        out=(repo/e['reconstruction_output_dir']/condition/'ccd'/('full' if args.max_cases is None else f'smoke_{len(paths)}'))
        if (out/'predictions.npz').exists() and not args.overwrite:
            raise FileExistsError(f'{out}/predictions.npz already exists; refusing overwrite')
        print(f'Condition={condition}, n={len(paths)}, lambda={lam:g}, strength={strength:g}, iter={max_iter}; checkpoint={ckpt_file}',flush=True)
        if args.dry_run:continue
        rows=[];preds=[]
        for group in batch_paths(paths,args.batch_size):
            b=stack_cases(group)
            prediction, infos=solve(A_np=b['A'], C_np=b['C'],
               shape_hwt=tuple(b['cases'][0]['h_gt'].shape),prior=prior,strength=strength,
               device=device,lambda_red=lam,
               alpha_numerator=float(redcfg['alpha_numerator']),max_iter=max_iter,
               min_iter=int(redcfg['min_iter']),tol=float(redcfg['tol']),
               normalize_forward=bool(redcfg.get('normalize_forward',True)))
            preds.append(prediction)
            for j,case in enumerate(b['cases']):
                m=evaluate_case(prediction[j],case,eps_relative=float(cfg['evaluation']['eps_relative']))
                rows.append({'case_index':len(rows),'case_file':group[j].name,
                    'family':str(np.asarray(case['residue_family']).item()),'condition':condition,
                    'method':'red_ccd_spline01_nonnegative',
                    'lambda_red':lam,'prior_strength':strength,
                    'initialization':'positive_part_normalized_C_obs',
                    'constraint':'projected_h_ge_0',**infos[j],**m})
            print(f'  {condition}: {len(rows)}/{len(paths)}',flush=True)
        arr=np.concatenate(preds,axis=0)
        if arr.shape[0]!=len(paths) or not np.isfinite(arr).all() or arr.min()<-1e-7:
            raise RuntimeError(f'{condition}: nonnegative predictions invalid')
        if args.max_cases is None:
            from collections import Counter
            cnt=Counter(x['family'] for x in rows)
            if cnt!={'biexponential':60,'pharmacokinetic':60}:raise RuntimeError(f'Unexpected family counts: {cnt}')
        out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'predictions.npz',prediction=arr)
        write_csv(out/'per_case.csv',rows)
        summary={'checkpoint':str(ckpt_file),'condition':condition,'n_cases':len(rows),'lambda_red':lam,
            'strength':strength,'max_iter':max_iter,'converged_fraction':float(np.mean([int(x['converged']) for x in rows]))}
        for key in METRICS:
            x=np.array([r[key] for r in rows],dtype=np.float64)
            summary[key+'_mean']=float(x.mean());summary[key+'_sd']=float(x.std(ddof=1)) if len(x)>1 else 0
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print('Saved:',out,'\n',summary,flush=True)
if __name__=='__main__': main()
