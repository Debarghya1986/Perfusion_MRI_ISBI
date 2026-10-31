#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import numpy as np

from frindel_spatiotemporal import reconstruct_frindel
from synthetic_utils import (
    evaluate_case,
    list_split,
    load_case,
    load_yaml,
    resolve_device,
    resolve_repo_path,
    write_csv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v3_model_consistent.yml")
    ap.add_argument("--max-cases", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = resolve_device(cfg["device"])
    paths = list_split(resolve_repo_path(cfg["paths"]["data_root"]), "val")

    if args.max_cases is not None:
        paths = paths[: int(args.max_cases)]

    cases = [load_case(p) for p in paths]
    fcfg = cfg["frindel"]

    lambda_t_values = [float(x) for x in fcfg["lambda_t_candidates"]]
    lambda_s_values = [float(x) for x in fcfg["lambda_s_candidates"]]
    delta = float(fcfg.get("delta", 0.02))

    rows = []
    best = None
    best_obj = float("inf")

    print("=" * 80)
    print(f"FRINDEL-ST tuning on {len(cases)} validation cases")
    print("Potential: convex w1(u)=sqrt(u^2+delta^2)-delta")
    print("V3 adaptation: 8-neighbour 2-D spatial system")
    print(f"delta={delta:g}")
    print("=" * 80)

    for lt in lambda_t_values:
        for ls in lambda_s_values:
            metrics = []
            infos = []

            for case in cases:
                pred, info = reconstruct_frindel(
                    case,
                    cfg,
                    device,
                    lambda_t=lt,
                    lambda_s=ls,
                    delta=delta,
                )
                metrics.append(
                    evaluate_case(
                        pred,
                        case,
                        eps_relative=float(cfg["evaluation"]["eps_relative"]),
                    )
                )
                infos.append(info)

            obj = float(np.mean([m["tuning_objective"] for m in metrics]))

            row = {
                "lambda_t": lt,
                "lambda_s": ls,
                "delta": delta,
                "objective_mean_absrel": obj,
                "objective_mean_absrel_percent": 100.0 * obj,
                "cbf_absrel_percent": float(
                    np.mean([m["cbf_absrel_percent"] for m in metrics])
                ),
                "cbv_absrel_percent": float(
                    np.mean([m["cbv_absrel_percent"] for m in metrics])
                ),
                "mtt_absrel_percent": float(
                    np.mean([m["mtt_absrel_percent"] for m in metrics])
                ),
                "h_nrmse": float(np.mean([m["h_nrmse"] for m in metrics])),
                "hq_iterations_mean": float(
                    np.mean([x["hq_iterations"] for x in infos])
                ),
                "hq_rel_change_mean": float(
                    np.mean([x["final_relative_change"] for x in infos])
                ),
                "cg_rel_residual_mean": float(
                    np.mean([x["final_cg_relative_residual"] for x in infos])
                ),
            }
            rows.append(row)

            print(
                f"lambda_t={lt:g} lambda_s={ls:g} | "
                f"obj={obj:.6f} | "
                f"CBF={row['cbf_absrel_percent']:.3f}% "
                f"CBV={row['cbv_absrel_percent']:.3f}% "
                f"MTT={row['mtt_absrel_percent']:.3f}% "
                f"hNRMSE={row['h_nrmse']:.5f}"
            )

            if obj < best_obj:
                best_obj = obj
                best = {
                    "lambda_t": lt,
                    "lambda_s": ls,
                    "delta": delta,
                    "objective_mean_absrel": obj,
                }

    outdir = resolve_repo_path(cfg["paths"]["output_root"]) / "frindel_tuning"
    outdir.mkdir(parents=True, exist_ok=True)

    write_csv(outdir / "grid_results.csv", rows)
    (outdir / "best.json").write_text(json.dumps(best, indent=2))

    print("\nBest:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
