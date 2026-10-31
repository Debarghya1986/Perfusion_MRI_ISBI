#!/usr/bin/env python3
"""Validation-only hyperparameter tuning for the *corrected* spline01 CCD.

Reuses project 25's exact projected RED solver and 33's checkpoint interface.
No TEST split is ever opened. No training or GitHub modification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import itertools
import json
from pathlib import Path
import re
import sys
from collections import Counter

import numpy as np
import torch
import yaml

from spline01_utils import load_experiment_config
from spline01_model import CCDSpline01, load_spline01_checkpoint
from synthetic_utils import (batch_paths, evaluate_case, list_split, load_case,
                             resolve_device, stack_cases)

METRICS = ("cbf_absrel_percent", "cbv_absrel_percent", "mtt_absrel_percent", "h_nrmse")
FAMILIES = ("biexponential", "pharmacokinetic")


class SplinePrior:
    """Exactly the sigma convention of 33_test_spline01_nonnegative.py."""
    def __init__(self, model, checkpoint):
        self.model = model
        self.lo = float(checkpoint["training_sigma_min_255"])
        self.hi = float(checkpoint["training_sigma_max_255"])

    @torch.inference_mode()
    def __call__(self, x_hwt, strength):
        strength = float(strength)
        if not self.lo <= strength <= self.hi:
            raise ValueError(f"CCD strength {strength} outside [{self.lo}, {self.hi}] in sigma/255 units")
        inp = x_hwt.permute(0, 3, 1, 2).unsqueeze(1).contiguous()
        sigma = torch.full((inp.shape[0],), strength / 255.0, dtype=inp.dtype, device=inp.device)
        return self.model(inp, sigma)[:, 0].permute(0, 2, 3, 1).contiguous()


def load_solver(repo):
    f = repo / "25_projected_red_aif.py"
    if not f.is_file():
        raise FileNotFoundError(f"Exact projected solver missing: {f}")
    spec = importlib.util.spec_from_file_location("projected_red_spline01_tuning", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.projected_red_batch


def fingerprint(paths):
    h = hashlib.sha256()
    for path in paths:
        p = Path(path)
        h.update(p.name.encode("utf-8"))
        with p.open("rb") as fh:
            for b in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(b)
    return h.hexdigest()


def audit_denominator(confirm_training):
    """The checkpoint's old metadata cannot establish which live training source ran."""
    source = re.sub(r"\s+", "", inspect.getsource(CCDSpline01.get_scaling))
    if "spline_coord+self.scaling_eps" not in source:
        raise RuntimeError("spline01_model.py does NOT implement the corrected spline_coord denominator")
    if not confirm_training:
        raise RuntimeError(
            "Training checkpoint metadata still documents the old denominator.\n"
            "Check that the 1000-epoch training PROCESS STARTED after correcting spline01_model.py.\n"
            "If certain, add --confirm-corrected-training to this command.\n"
            "Editing a .py file while training is running does NOT change that running process."
        )


def validation_paths(repo, cfg):
    root = Path(cfg["paths"]["data_root"])
    if not root.is_absolute():
        root = repo / root
    all_paths = list_split(root, "val")
    if len(all_paths) != 128:
        raise RuntimeError(f"Expected 128 nominal validation cases, got {len(all_paths)}")
    groups = {family: [] for family in FAMILIES}
    for p in all_paths:
        case = load_case(p)
        family = str(np.asarray(case["residue_family"]).item())
        if family not in groups:
            raise RuntimeError(f"Unexpected validation family: {family} in {p}")
        groups[family].append(p)
    if [len(groups[k]) for k in FAMILIES] != [64, 64]:
        raise RuntimeError(f"Validation family mismatch: { {k:len(v) for k,v in groups.items()} }")
    return groups


def means_sd(values):
    a = np.asarray(values, dtype=np.float64)
    return (float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def run_pair(paths, *, cfg, prior, solve, device, batch_size, lam, strength, max_iter):
    rows = []
    red = cfg["red"]
    for batch in batch_paths(paths, batch_size):
        data = stack_cases(batch)
        pred, info = solve(A_np=data["A"], C_np=data["C"],
                           shape_hwt=tuple(data["cases"][0]["h_gt"].shape),
                           prior=prior, strength=float(strength), device=device,
                           lambda_red=float(lam),
                           alpha_numerator=float(red["alpha_numerator"]),
                           max_iter=max_iter, min_iter=int(red["min_iter"]),
                           tol=float(red["tol"]),
                           normalize_forward=bool(red.get("normalize_forward", True)))
        if not np.isfinite(pred).all() or pred.min() < -1e-7:
            raise RuntimeError(f"Invalid nonnegative result lambda={lam} strength={strength}")
        for case, prediction, inf in zip(data["cases"], pred, info):
            values = evaluate_case(prediction, case,
                       eps_relative=float(cfg["evaluation"]["eps_relative"]))
            rows.append({"family": str(np.asarray(case["residue_family"]).item()),
                         "converged": bool(inf["converged"]),
                         "iterations": int(inf["iterations"]),
                         **{k: float(values[k]) for k in METRICS},
                         "objective": float(values["tuning_objective"])})
    return rows


def summarize(rows, stage, lam, strength):
    ret = {"stage": stage, "lambda_red": float(lam), "strength": float(strength),
           "n_cases": len(rows)}
    for k in (*METRICS, "objective"):
        ret[k+"_mean"], ret[k+"_sd"] = means_sd([r[k] for r in rows])
    ret["converged_fraction"] = float(np.mean([r["converged"] for r in rows]))
    ret["iterations_mean"] = float(np.mean([r["iterations"] for r in rows]))
    ret["family_counts"] = dict(Counter(r["family"] for r in rows))
    for family in FAMILIES:
        subset = [r for r in rows if r["family"] == family]
        for k in (*METRICS, "objective"):
            ret[f"{family}_{k}_mean"], ret[f"{family}_{k}_sd"] = means_sd([r[k] for r in subset])
    return ret


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})
    tmp.replace(path)


def show(r):
    return (f"lambda={r['lambda_red']:.4g} strength={r['strength']:.4g} | "
            f"CBF={r['cbf_absrel_percent_mean']:.3f}±{r['cbf_absrel_percent_sd']:.3f}% "
            f"CBV={r['cbv_absrel_percent_mean']:.3f}±{r['cbv_absrel_percent_sd']:.3f}% "
            f"MTT={r['mtt_absrel_percent_mean']:.3f}±{r['mtt_absrel_percent_sd']:.3f}% | "
            f"hNRMSE={r['h_nrmse_mean']:.5f}±{r['h_nrmse_sd']:.5f} "
            f"obj={100*r['objective_mean']:.4f}% conv={100*r['converged_fraction']:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config_v3_spline01.yml")
    ap.add_argument("--stage", choices=["coarse", "final", "all"], default="all",
                    help="Run only coarse, only final after coarse, or both")
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.1, 0.4, 1.6, 3.2, 6.4])
    ap.add_argument("--strengths", nargs="+", type=float, default=[1, 3, 6, 12])
    ap.add_argument("--coarse-per-family", type=int, default=8,
                    help="Choose first N validation cases from each family. Default 8+8.")
    ap.add_argument("--finalists", type=int, default=5,
                    help="Number of coarse candidates reevaluated on all 128 validation cases")
    ap.add_argument("--min-convergence", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=None,
                    help="Only for debugging; normal search uses config max_iter")
    ap.add_argument("--out-root", default="outputs/v3/spline01/nominal_nonnegative_tuning")
    ap.add_argument("--confirm-corrected-training", action="store_true",
                    help="Confirm TRAINING STARTED after denominator correction, not just file edited afterwards")
    ap.add_argument("--overwrite", action="store_true", help="Explicitly replace old tuning output")
    args = ap.parse_args()
    if args.batch_size < 1 or not 1 <= args.coarse_per_family <= 64 or args.finalists < 1:
        ap.error("Invalid batch size, coarse family size, or finalists")
    if not (0 <= args.min_convergence <= 1) or min(args.lambdas) <= 0:
        ap.error("Invalid convergence fraction or lambda candidates")
    audit_denominator(args.confirm_corrected_training)
    cfg, exp, mc, repo = load_experiment_config(args.config)
    max_iter = int(args.max_iter or cfg["red"]["max_iter"])
    if max_iter < int(cfg["red"]["min_iter"]):
        ap.error("max_iter < min_iter")
    device = resolve_device(cfg.get("device", "auto"))
    ckpt_file = Path(cfg["paths"]["inference_checkpoint"])
    if not ckpt_file.is_file():
        raise FileNotFoundError(ckpt_file)
    model, ckpt = load_spline01_checkpoint(ckpt_file, device,
                   expected_multiplier=exp["spline_input_multiplier"])
    if ckpt["model_config"] != mc:
        raise RuntimeError("Checkpoint architecture differs from experiment config")
    prior = SplinePrior(model, ckpt)
    if any(not prior.lo <= x <= prior.hi for x in args.strengths):
        ap.error(f"Strength candidates must lie within [{prior.lo},{prior.hi}]")
    solve = load_solver(repo)
    groups = validation_paths(repo, cfg)
    coarse_paths = [p for family in FAMILIES for p in groups[family][:args.coarse_per_family]]
    full_paths = [p for family in FAMILIES for p in groups[family]]
    candidates = list(dict.fromkeys((float(a), float(b)) for a,b in itertools.product(args.lambdas,args.strengths)))
    original = (float(exp["lambda_red"]), float(exp["prior_strength_255"]))
    out = (repo / args.out_root).resolve()
    if (out / "best_validation.json").exists() and not args.overwrite:
        raise FileExistsError(f"Previous tuning exists: {out}. Use new --out-root or explicit --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    report = {"source_checkpoint": str(ckpt_file), "checkpoint_sha256": fingerprint([ckpt_file]),
              "source_solver_sha256": fingerprint([repo / '25_projected_red_aif.py']),
              "model_sha256": fingerprint([repo / 'spline01_model.py']),
              "training_denominator_confirmed_by_user": args.confirm_corrected_training,
              "checkpoint_formula_metadata": ckpt.get("spline_coordinate_formula"),
              "dataset": "nominal_AIF_validation_only", "split": "val",
              "coarse_n": len(coarse_paths), "final_n": len(full_paths),
              "coarse_candidates": candidates, "max_iter": max_iter,
              "objective": "mean of per-case (CBF, CBV, MTT) AbsRel fractions",
              "minimum_final_convergence_fraction": args.min_convergence}
    if args.stage in ("all", "coarse"):
        (out / "audit.json").write_text(json.dumps(report, indent=2))
    print("TRAINING metadata:", ckpt.get("spline_coordinate_formula"),flush=True)
    print("NOTICE: Old-formula metadata may be stale; user confirmed the training PROCESS used corrected denominator.",flush=True)
    print(f"NOMINAL AIF ONLY | val coarse={len(coarse_paths)} balanced | val final=128 balanced | GPU={device}",flush=True)
    print(f"Full projected solver max_iter={max_iter}; checkpoint frozen: {ckpt_file}",flush=True)
    if args.stage in ("all", "coarse"):
        results = []
        for i,(lam,strength) in enumerate(candidates,1):
            print(f"COARSE [{i}/{len(candidates)}] ",end="",flush=True)
            row = summarize(run_pair(coarse_paths, cfg=cfg, prior=prior, solve=solve,
                       device=device, batch_size=args.batch_size, lam=lam,
                       strength=strength, max_iter=max_iter), "coarse",lam,strength)
            results.append(row)
            print(show(row),flush=True)
            save_csv(out / "coarse_results.csv", results)
        if args.stage == "coarse":
            print("Coarse search complete. Inspect", out / "coarse_results.csv",
                  "then run with --stage final (same candidate grid and parameters).", flush=True)
            return
    else:
        coarse_csv = out / "coarse_results.csv"
        if not coarse_csv.exists():
            raise FileNotFoundError(f"First run --stage coarse: {coarse_csv}")
        existing = json.loads((out / "audit.json").read_text())
        if (existing["checkpoint_sha256"] != report["checkpoint_sha256"] or
            existing["source_solver_sha256"] != report["source_solver_sha256"] or
            existing["model_sha256"] != report["model_sha256"] or
            existing["coarse_candidates"] != [list(x) for x in report["coarse_candidates"]] or
            existing["coarse_n"] != report["coarse_n"] or
            existing["max_iter"] != report["max_iter"]):
            raise RuntimeError("Checkpoint, solver, model, or coarse grid changed: rerun coarse in NEW --out-root")
        with coarse_csv.open(newline="") as f:
            results = list(csv.DictReader(f))
        if len(results) != len(candidates):
            raise RuntimeError(f"Incomplete coarse results ({len(results)}/{len(candidates)})")
    rank = sorted(results, key=lambda x: (float(x["objective_mean"]), -float(x["converged_fraction"])))
    selected = [(float(x["lambda_red"]), float(x["strength"])) for x in rank[:args.finalists]]
    if original not in selected:
        selected.append(original)
    print("\nFINALISTS including old pair:",selected,flush=True)
    finals = []
    for i,(lam,strength) in enumerate(selected,1):
        print(f"FINAL  [{i}/{len(selected)}] ",end="",flush=True)
        row = summarize(run_pair(full_paths, cfg=cfg, prior=prior, solve=solve,
                   device=device, batch_size=args.batch_size, lam=lam,
                   strength=strength, max_iter=max_iter), "final",lam,strength)
        finals.append(row)
        print(show(row),flush=True)
        save_csv(out / "final_128_results.csv", finals)
    admissible = [x for x in finals if x["converged_fraction"] >= args.min_convergence]
    if not admissible:
        print("WARNING: no finalists meet minimum convergence! Not automatically selecting parameters.",flush=True)
        (out / "no_qualifying_candidate.json").write_text(json.dumps(finals, indent=2))
        return
    best = min(admissible,key=lambda x:(x["objective_mean"], -x["converged_fraction"]))
    (out / "best_validation.json").write_text(json.dumps(best, indent=2))
    print("\nBEST FULL NOMINAL VALIDATION:\n"+show(best),flush=True)
    for fam in FAMILIES:
        print(f"{fam}: " + " | ".join(
             f"{key}={best[f'{fam}_{key}_mean']:.5f}±{best[f'{fam}_{key}_sd']:.5f}"
             for key in ("cbf_absrel_percent","cbv_absrel_percent","mtt_absrel_percent","h_nrmse")),flush=True)
    with Path(args.config).expanduser().resolve().open() as f:
        raw = yaml.safe_load(f)
    raw["experiment"]["lambda_red"] = float(best["lambda_red"])
    raw["experiment"]["prior_strength_255"] = float(best["strength"])
    # New output root ensures old fixed-parameter TEST archives are not overwritten.
    raw["experiment"]["reconstruction_output_dir"] = "outputs/v3/spline01_tuned/nonnegative_red"
    raw["experiment"]["tuning_manifest"] = str(out / "best_validation.json")
    destination = repo / "config_v3_spline01_tuned.yml"
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {destination}; use --overwrite only if intended")
    destination.write_text(yaml.safe_dump(raw, sort_keys=False))
    print(f"Tuned config written to {destination}. No TEST cases were read or reconstructed.",flush=True)

if __name__ == "__main__":
    main()
