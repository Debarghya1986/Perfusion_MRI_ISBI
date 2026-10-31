#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from v3.data import list_split, load_case
from v3.metrics import perfusion_maps_from_h
from v3.utils import load_yaml, resolve_path, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v3.yml")
    ap.add_argument("--max-per-split", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    root = resolve_path(cfg["paths"]["data_root"])
    report = {"data_root": str(root), "splits": {}}
    label_names = {1:"healthy_gray",2:"healthy_white",3:"ischemic_core"}

    for split in ("train","val","test"):
        paths = list_split(root, split)
        if args.max_per_split is not None:
            paths = paths[:args.max_per_split]
        fam = Counter(); shapes = Counter(); class_vals = defaultdict(lambda: {"cbf":[],"mtt":[],"cbv":[]})
        fwd = []; map_err = []; snr_proxy=[]
        for p in paths:
            c = load_case(p)
            fam[c["residue_family"]] += 1
            shapes[c["shape_model"]] += 1
            H,W,T = c["h_gt"].shape
            Ccheck = (c["h_gt"].reshape(-1,T) @ c["A"].T).reshape(H,W,T)
            fwd.append(float(np.linalg.norm(Ccheck-c["C_clean"])/(np.linalg.norm(c["C_clean"])+1e-12)))
            cbf,cbv,mtt = perfusion_maps_from_h(c["h_gt"],c["h_scale"],float(c["t"][1]-c["t"][0]),c["mask"])
            m=c["mask"]>0
            map_err.append(max(float(np.max(np.abs(cbf[m]-c["cbf_gt"][m]))),float(np.max(np.abs(cbv[m]-c["cbv_gt"][m]))),float(np.max(np.abs(mtt[m]-c["mtt_gt"][m])))))
            for lab,name in label_names.items():
                mm=np.asarray(c["tissue_label"])==lab
                if mm.any():
                    class_vals[name]["cbf"].extend(np.asarray(c["cbf_gt"])[mm].tolist())
                    class_vals[name]["mtt"].extend(np.asarray(c["mtt_gt"])[mm].tolist())
                    class_vals[name]["cbv"].extend(np.asarray(c["cbv_gt"])[mm].tolist())
            # Simple concentration perturbation audit; not equal to signal-domain SNR definition.
            snr_proxy.append(float(np.linalg.norm(c["C_clean"][m])/(np.linalg.norm((c["C_obs"]-c["C_clean"])[m])+1e-12)))
        cls = {}
        for name,d in class_vals.items():
            cls[name] = {k:{"mean":float(np.mean(v)),"sd":float(np.std(v,ddof=1))} for k,v in d.items() if v}
        report["splits"][split] = {
            "n_checked": len(paths), "family_counts": dict(fam), "shape_counts": dict(shapes),
            "max_discrete_forward_relative_error": max(fwd) if fwd else None,
            "max_map_identity_error": max(map_err) if map_err else None,
            "concentration_snr_proxy_mean": float(np.mean(snr_proxy)) if snr_proxy else None,
            "tissue_statistics": cls,
        }
    out = resolve_path(cfg["paths"]["output_root"]) / "verification.json"
    save_json(out, report)
    print(json.dumps(report, indent=2))
    print("Saved:", out)

if __name__ == "__main__":
    main()
