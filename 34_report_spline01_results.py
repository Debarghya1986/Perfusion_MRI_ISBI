#!/usr/bin/env python3
"""Terminal-only 120-case overall and family-wise tables: old vs new projected CCD."""
from __future__ import annotations
import argparse,csv,math
from pathlib import Path
from collections import Counter

CONDITIONS=('baseline','delayed','broad')
FAMILIES=('biexponential','pharmacokinetic')
FIELDS=('cbf_absrel_percent','cbv_absrel_percent','mtt_absrel_percent','h_nrmse')

def read_rows(path, expected):
    if not path.is_file():raise FileNotFoundError(path)
    with path.open(newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=expected:raise ValueError(f'{path} contains {len(rows)} cases; expected {expected}')
    files=[r['case_file'] for r in rows]
    if len(files)!=len(set(files)):raise ValueError(f'{path}: repeated case_file')
    for r in rows:
        for k in FIELDS:
            v=float(r[k]);
            if not math.isfinite(v) or v<0:raise ValueError(f'{path}: invalid {k}={v}')
    return rows

def stats(a):
    m=math.fsum(a)/len(a)
    return m,math.sqrt(math.fsum((v-m)**2 for v in a)/(len(a)-1))

def section(title, content, fam):
    print('\n'+'='*135+'\n'+title+'\n'+'='*135)
    print(f"{'Method':<27}{'CBF AbsRel (%)':>24}{'CBV AbsRel (%)':>24}{'MTT AbsRel (%)':>24}{'h-NRMSE':>24}{'Mean Obj (%)':>16}")
    print('-'*135)
    for label,rows in content.items():
        rr=rows if fam is None else [r for r in rows if r['family']==fam]
        def fmt(k):
            u,s=stats([float(r[k]) for r in rr])
            digits=5 if k=='h_nrmse' else 4
            return f'{u:.{digits}f} ± {s:.{digits}f}'
        obj=stats([(sum(float(r[k]) for k in FIELDS[:3])/3) for r in rr])[0]
        print(f'{label:<27}{fmt(FIELDS[0]):>24}{fmt(FIELDS[1]):>24}{fmt(FIELDS[2]):>24}{fmt(FIELDS[3]):>24}{obj:>16.4f}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='.')
    p.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS))
    p.add_argument('--no-old',action='store_true',help='Display new variant only')
    args=p.parse_args()
    repo=Path(args.repo).resolve()
    for condition in args.conditions:
        new = repo / 'outputs/v3/spline01_tuned/nonnegative_red' / condition / 'ccd/full/per_case.csv'        
        old=repo/'outputs/v3/aif_extension_tests/nonnegative_red'/condition/'ccd/full/per_case.csv'
        content={}
        if not args.no_old:content['RED-CCD (original spline)']=read_rows(old,120)
        content['RED-CCD (spline[0,1])']=read_rows(new,120)
        for label,rr in content.items():
            cnt=Counter(r['family'] for r in rr)
            if cnt!={'biexponential':60,'pharmacokinetic':60}:
                raise RuntimeError(f'{condition}/{label}: expected family split 60/60, found {cnt}')
        if len(content)==2:
            arr=list(content.values())
            signatures=[{r['case_file']:r['family'] for r in x} for x in arr]
            if signatures[0]!=signatures[1]:raise ValueError('Old vs new use different test case identities')
        print('\n'+'#'*135+'\nAIF CONDITION: '+condition.upper()+'\n'+'#'*135)
        section('OVERALL BALANCED TEST SET (120 CASES)',content,None)
        for fam in FAMILIES:
            section(f'{fam.upper()} TEST SET (60 CASES)',content,fam)
    print('\nMean ± sample SD across cases; all metrics lower is better. Reporting only: no files written.')
if __name__=='__main__':main()
