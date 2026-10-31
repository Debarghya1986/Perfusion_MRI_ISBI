#!/usr/bin/env python3
"""Train CCD2 from scratch with [0,1] noise-scaling spline and sigma*10 coordinate.
Run in ~/ISBI_2027_Final with the files from this package in repo root.
"""
from __future__ import annotations
import argparse, json, importlib.util
from pathlib import Path
import torch
from training.trainer_ccd_3d import CCD3DTrainer, TrainerConfig
from config_utils import resolve_device
from spline01_utils import load_experiment_config
from spline01_model import build_spline01_model


def create_data(cfg,repo):
    loader_file=repo/'02_dataset_loaders.py'
    spec=importlib.util.spec_from_file_location('spline01_loaders',loader_file)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t=cfg['denoiser_training']
    return mod.create_dataloaders(
       data_root=str((repo/cfg['paths']['synthetic_data_root']).resolve()),
       batch_size=int(t['batch_size']),num_workers=int(t['num_workers']),
       sigma_min=float(t['sigma_min_255'])/255,sigma_max=float(t['sigma_max_255'])/255,
       val_sigma=float(t['val_sigma_255'])/255,seed=int(t['seed']))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='config_v3_spline01.yml')
    ap.add_argument('--epochs',type=int,default=None,help='For smoke tests only; publishable model should use full settings')
    ap.add_argument('--allow-existing',action='store_true',help='Explicitly allow training to replace existing spline01 checkpoint')
    args=ap.parse_args()
    cfg,e,mc,repo=load_experiment_config(args.config)
    p=cfg['paths']; t=cfg['denoiser_training']; opt=cfg['denoiser_optimizer']
    ckpt=Path(p['inference_checkpoint']); best=Path(p['checkpoint_dir'])/'model_best.pth'
    if (ckpt.exists() or best.exists()) and not args.allow_existing:
        raise FileExistsError(f'{ckpt} or {best} already exists; refusing overwrite. Use --allow-existing deliberately.')
    model=build_spline01_model(mc)
    dev=resolve_device(cfg.get('device','auto'))
    train_loader,val_loader=create_data(cfg,repo)
    learning=float(opt['lr']); diff=bool(opt.get('lr_diff',True))
    param_lrs={}
    for x in ['conv','q','Q','tau','gamma','activation','scaling','bias']:
        z=opt.get('lr_'+x)
        param_lrs['lr_'+x]= (None if z is None else float(z)) if diff else learning
    n_epochs=int(args.epochs if args.epochs is not None else t['epochs'])
    if n_epochs<1: raise ValueError('epochs must be positive')
    tr=TrainerConfig(epochs=n_epochs,device=str(dev),seed=int(t['seed']),
        optimizer=str(opt['optimizer']),lr=learning,weight_decay=float(opt['weight_decay']),
        **param_lrs,loss=str(t['loss']),amp=bool(t['amp']),
        grad_clip_norm=opt.get('grad_clip_norm'),scheduler=str(t['scheduler']),
        cosine_eta_min=float(t['cosine_eta_min']),validate_every=int(t['validate_every']),
        checkpoint_every=int(t['checkpoint_every'] or max(1,n_epochs//20)),
        checkpoint_dir=str(p['checkpoint_dir']),resume_from=t.get('resume_from'),
        synthesize_noise_if_missing=True, data_range=1.0,keep_last=True)
    print('Training new CCD from scratch; new spline grid',mc['param_spline_scaling'])
    print('Noise synthesis remains sigma_h in [0,25/255] on clean h volumes')
    print('Spline coordinate = normalized sigma *',e['spline_input_multiplier'])
    print('Scaling denominator = 10*sigma_normalized + scaling_eps')
    print('Checkpoint:',ckpt)
    trainer=CCD3DTrainer(model,tr,on_epoch_end=lambda m:print(
        f"epoch {int(m['epoch']):04d}/{n_epochs} train={m['train_loss']:.5e} val={m.get('val_loss',float('nan')):.5e} "
        f"valPSNR={m.get('val_psnr',float('nan')):.3f}",flush=True))
    history=trainer.fit(train_loader,val_loader)
    if not best.is_file():raise FileNotFoundError(best)
    state=torch.load(best,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model_state_dict'],strict=True)
    model.project_parameters();model.eval()
    ckpt.parent.mkdir(parents=True,exist_ok=True)
    record={
      'spline01_experiment':'sigma_coordinate_remapping_v1',
      'model_state':model.state_dict(),'model_config':mc,
      'best_val_psnr':float(state.get('best_val_psnr',float('nan'))),
      'training_sigma_min_255':float(t['sigma_min_255']),
      'training_sigma_max_255':float(t['sigma_max_255']),
      'spline_coordinate_formula':'spline(10 * sigma_normalized); denominator 10*sigma_normalized + epsilon',
      'noise_convention':'Normalized physical sigma supplied in trainer and RED, spline coordinate remapped only',
      'project_config':cfg,
    }
    torch.save(record,ckpt)
    out=Path(p['training_output_dir']);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps({
       'best_val_psnr':record['best_val_psnr'],
       'checkpoint':str(ckpt),'epochs':n_epochs,
       'spline_grid':mc['param_spline_scaling'],
       'noise_multiplier':e['spline_input_multiplier']},indent=2))
    print('Saved trained inference checkpoint:',ckpt)
if __name__=='__main__':main()
