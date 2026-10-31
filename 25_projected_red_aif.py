#!/usr/bin/env python3
"""Hard h>=0 projected RED-GD for CCD, TV and optionally BM4D; same V3 parameters.
Separate comparison, NEVER overwrites previous benchmark predictions.
The only solver changes are h0=positive part of C_norm and projection
of every gradient update onto h>=0; original denoising priors are reused.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from synthetic_utils import (load_yaml, resolve_repo_path, resolve_device, list_split,
                             load_case, stack_cases, batch_paths, make_prior, evaluate_case)

CONDITIONS=('baseline','delayed','broad','sharp')
METHODS=('ccd','tv','bm4d')
METRICS=('cbf_absrel_percent','cbv_absrel_percent','mtt_absrel_percent','h_nrmse')


def dataset_paths(repo,cfg,condition):
    root=resolve_repo_path(cfg['paths']['data_root']) if condition=='baseline' else repo/'outputs/v3/aif_shift/datasets'/condition
    return list_split(root,'test')


@torch.inference_mode()
def projected_red_batch(*, A_np,C_np, shape_hwt, prior, strength, device,
                        lambda_red, alpha_numerator, max_iter, min_iter, tol,
                        normalize_forward):
    H,W,T=shape_hwt
    B=int(A_np.shape[0])
    A_raw=torch.as_tensor(A_np, dtype=torch.float32, device=device)
    C_raw=torch.as_tensor(C_np.reshape(B,H*W,T), dtype=torch.float32, device=device)
    if normalize_forward:
        scale=A_raw.abs().amax(dim=(1,2)).clamp_min(1.0e-8)
        A=A_raw/scale[:,None,None]
        C=C_raw/scale[:,None,None]
    else:
        scale=torch.ones(B,device=device)
        A=A_raw; C=C_raw
    L=torch.linalg.eigvalsh(torch.bmm(A.transpose(1,2),A)).amax(dim=1)
    alpha=float(alpha_numerator)/(L+float(lambda_red))
    x=C.clamp_min(0).clone()  # Project the H0=C_norm initialization.
    active=torch.ones(B,dtype=torch.bool,device=device)
    converged=torch.zeros(B,dtype=torch.bool,device=device)
    iterations=torch.full((B,),int(max_iter),dtype=torch.int64,device=device)
    rel_final=torch.full((B,),float('inf'),device=device)
    t0=time.perf_counter()
    for it in range(1,int(max_iter)+1):
        idx=active.nonzero(as_tuple=False).flatten()
        if idx.numel()==0: break
        old=x.index_select(0,idx)
        den=prior(old.reshape(-1,H,W,T),strength).reshape(-1,H*W,T)
        Ai=A.index_select(0,idx)
        Ci=C.index_select(0,idx)
        data_grad=torch.bmm(torch.bmm(old,Ai.transpose(1,2))-Ci,Ai)
        ai=alpha.index_select(0,idx)[:,None,None]
        new=(old-ai*(data_grad+float(lambda_red)*(old-den))).clamp_min(0.0)
        rel=torch.linalg.vector_norm((new-old).flatten(1),dim=1)/(
              torch.maximum(torch.linalg.vector_norm(new.flatten(1),dim=1),
                            torch.linalg.vector_norm(old.flatten(1),dim=1))+1.0e-12)
        x.index_copy_(0,idx,new)
        rel_final.index_copy_(0,idx,rel)
        if it>=int(min_iter):
            done=rel<float(tol)
            if bool(done.any()):
                stop_idx=idx[done]
                converged[stop_idx]=True
                active[stop_idx]=False
                iterations[stop_idx]=it
    if device.type=='cuda':torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-t0
    infos=[]
    for b in range(B):
        resid=float(torch.linalg.vector_norm(x[b]@A[b].T-C[b])/(torch.linalg.vector_norm(C[b])+1.e-12))
        infos.append({'iterations':int(iterations[b]),'converged':bool(converged[b]),
                      'rel_change':float(rel_final[b]),'L_f':float(L[b]),
                      'alpha':float(alpha[b]),'forward_scale':float(scale[b]),
                      'data_residual':resid,'effective_runtime_seconds':elapsed/B})
    return x.reshape(B,H,W,T).cpu().numpy().astype(np.float32),infos


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='config_v3_model_consistent.yml')
    ap.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS))
    ap.add_argument('--methods',nargs='+',choices=METHODS,default=['ccd','tv'],
                    help='Default CCD/TV; choose --methods bm4d for expensive BM4D experiment')
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--max-cases',type=int,default=None)
    ap.add_argument('--max-iter',type=int,default=None,help='Smoke/debug only. Publication uses original 5000.')
    ap.add_argument('--out-root',default='outputs/v3/aif_extension_tests/nonnegative_red')
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    if args.batch_size<=0:raise ValueError('--batch-size must be positive')
    repo=Path(__file__).resolve().parent
    cfg=load_yaml(repo/args.config)
    device=resolve_device(cfg.get('device','auto'))
    redcfg=cfg['red']
    max_iter=int(args.max_iter or redcfg['max_iter'])
    outroot=(repo/args.out_root).resolve()
    priors={m:make_prior(m,cfg=cfg,device=device) for m in args.methods}
    for condition in args.conditions:
        paths=dataset_paths(repo,cfg,condition)
        if args.max_cases is not None:paths=paths[:args.max_cases]
        if not paths:raise RuntimeError(f'No test data for {condition}')
        for method in args.methods:
            params=cfg['test_red_params'][method]
            lam=float(params['lambda_red']);strength=float(params['strength'])
            out=outroot/condition/method/(f'smoke_{len(paths)}' if args.max_cases is not None else 'full')
            if (out/'predictions.npz').exists() and not args.overwrite:
                raise FileExistsError(f'{out}/predictions.npz exists. Use --overwrite deliberately.')
            rows=[]; preds=[];case_index=0
            print(f'CONDITION {condition} | RED-{method.upper()} projected h>=0 | n={len(paths)} | lambda={lam}, strength={strength}',flush=True)
            for group in batch_paths(paths,args.batch_size):
                batch=stack_cases(group)
                # Match original 04_benchmark_red_priors_initC.py exactly.
                pred,infos=projected_red_batch(A_np=batch['A'],C_np=batch['C'],
                    shape_hwt=tuple(batch['cases'][0]['h_gt'].shape),prior=priors[method],
                    strength=strength,device=device,lambda_red=lam,
                    alpha_numerator=float(redcfg['alpha_numerator']),
                    max_iter=max_iter,min_iter=int(redcfg['min_iter']),
                    tol=float(redcfg['tol']),normalize_forward=bool(redcfg.get('normalize_forward',True)))
                preds.append(pred)
                for j,case in enumerate(batch['cases']):
                    metric=evaluate_case(pred[j],case,eps_relative=float(cfg['evaluation']['eps_relative']))
                    rows.append({'case_index':case_index,'case_file':group[j].name,
                        'family':str(np.asarray(case['residue_family']).item()),
                        'condition':condition,'method':'red_'+method+'_nonnegative',
                        'lambda_red':lam,'prior_strength':strength,
                        'initialization':'positive_part_normalized_C_obs',
                        'constraint':'projected_h_ge_0',**infos[j],**metric})
                    case_index+=1
                print(f'  {condition} {method}: {case_index}/{len(paths)}',flush=True)
            arr=np.concatenate(preds,axis=0)
            if arr.shape[0]!=len(paths) or not np.isfinite(arr).all() or np.min(arr)<-1.e-7:
                raise RuntimeError('Nonnegative reconstruction invalid')
            out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/'predictions.npz',prediction=arr)
            csv_write(out/'per_case.csv',rows)
            summary={'condition':condition,'method':'red_'+method+'_nonnegative',
                 'n_cases':len(paths),'lambda_red':lam,'strength':strength,'max_iter':max_iter,
                 'converged_fraction':float(np.mean([int(r['converged']) for r in rows])),
                 'checkpoint_frozen':True if method=='ccd' else None}
            for metric in METRICS:
                vals=np.array([r[metric] for r in rows],dtype=float)
                summary[metric+'_mean']=float(vals.mean())
                summary[metric+'_sd']=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
            print('Saved',out,flush=True)


if __name__=='__main__':main()
