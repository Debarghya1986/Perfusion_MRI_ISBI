#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from frindel_spatiotemporal import reconstruct_frindel
from synthetic_utils import (
    evaluate_case,
    list_split,
    load_case,
    load_yaml,
    mean_sd,
    resolve_device,
    resolve_repo_path,
    write_csv,
)


# ============================================================================
# Summary utilities
# ============================================================================

def summarize(rows):
    out = {
        "method": "frindel",
        "n_cases": len(rows),
    }

    # ------------------------------------------------------------------
    # Main quantitative metrics
    # ------------------------------------------------------------------
    for key in (
        "cbf_absrel_percent",
        "cbv_absrel_percent",
        "mtt_absrel_percent",
        "cbf_nrmse",
        "cbv_nrmse",
        "mtt_nrmse",
        "h_nrmse",
        "tuning_objective",
    ):
        mu, sd = mean_sd(
            [r[key] for r in rows]
        )

        out[key + "_mean"] = mu
        out[key + "_sd"] = sd

    # ------------------------------------------------------------------
    # Convergence audit
    # ------------------------------------------------------------------
    out["convergence_fraction"] = float(
        np.mean(
            [
                float(r["converged"])
                for r in rows
            ]
        )
    )

    # Half-quadratic outer iterations
    mu, sd = mean_sd(
        [
            r["hq_iterations"]
            for r in rows
        ]
    )

    out["hq_iterations_mean"] = mu
    out["hq_iterations_sd"] = sd

    # Final CG solve iterations
    mu, sd = mean_sd(
        [
            r["final_cg_iterations"]
            for r in rows
        ]
    )

    out["final_cg_iterations_mean"] = mu
    out["final_cg_iterations_sd"] = sd

    # Final HQ relative change
    mu, sd = mean_sd(
        [
            r["final_relative_change"]
            for r in rows
        ]
    )

    out["final_relative_change_mean"] = mu
    out["final_relative_change_sd"] = sd

    # Final CG relative residual
    mu, sd = mean_sd(
        [
            r["final_cg_relative_residual"]
            for r in rows
        ]
    )

    out["final_cg_relative_residual_mean"] = mu
    out["final_cg_relative_residual_sd"] = sd

    # Runtime per case
    mu, sd = mean_sd(
        [
            r["runtime_seconds"]
            for r in rows
        ]
    )

    out["runtime_seconds_mean"] = mu
    out["runtime_seconds_sd"] = sd

    return out


def _sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="config_v3_model_consistent.yml",
    )

    ap.add_argument(
        "--max-cases",
        type=int,
        default=None,
    )

    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    cfg = load_yaml(args.config)

    device = resolve_device(
        cfg["device"]
    )

    output_root = resolve_repo_path(
        cfg["paths"]["output_root"]
    )

    fcfg = cfg["frindel"]

    # ------------------------------------------------------------------
    # Locked validation-selected parameters
    # ------------------------------------------------------------------
    if "test_params" in fcfg:
        selected = dict(
            fcfg["test_params"]
        )

        source = (
            "config.frindel.test_params"
        )

    else:
        best_file = (
            output_root
            / "frindel_tuning"
            / "best.json"
        )

        if not best_file.exists():
            raise FileNotFoundError(
                f"{best_file} not found. "
                "Run 05_tune_frindel.py first."
            )

        selected = json.loads(
            best_file.read_text()
        )

        source = str(best_file)

        print(
            "WARNING: using tuning best.json directly. "
            "For the final held-out test, the validation-selected "
            "parameters should be frozen under "
            "config.frindel.test_params."
        )

    lambda_t = float(
        selected["lambda_t"]
    )

    lambda_s = float(
        selected["lambda_s"]
    )

    delta = float(
        selected.get(
            "delta",
            fcfg.get(
                "delta",
                0.02,
            ),
        )
    )

    # ------------------------------------------------------------------
    # Held-out test split
    # ------------------------------------------------------------------
    data_root = resolve_repo_path(
        cfg["paths"]["data_root"]
    )

    paths = list_split(
        data_root,
        "test",
    )

    if args.max_cases is not None:
        paths = paths[
            : int(args.max_cases)
        ]

    # ------------------------------------------------------------------
    # Console header
    # ------------------------------------------------------------------
    print("=" * 80)
    print("FRINDEL-ST HELD-OUT TEST")
    print("=" * 80)

    print(
        "n_cases      :",
        len(paths),
    )

    print(
        "lambda_t     :",
        lambda_t,
    )

    print(
        "lambda_s     :",
        lambda_s,
    )

    print(
        "delta        :",
        delta,
    )

    print(
        "hq_max_iter  :",
        int(
            fcfg.get(
                "hq_max_iter",
                20,
            )
        ),
    )

    print(
        "hq_tol       :",
        float(
            fcfg.get(
                "hq_tol",
                1.0e-4,
            )
        ),
    )

    print(
        "cg_max_iter  :",
        int(
            fcfg.get(
                "cg_max_iter",
                80,
            )
        ),
    )

    print(
        "cg_tol       :",
        float(
            fcfg.get(
                "cg_tol",
                1.0e-5,
            )
        ),
    )

    print(
        "param source :",
        source,
    )

    print()

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------
    rows = []
    preds = []

    hq_tol = float(
        fcfg.get(
            "hq_tol",
            1.0e-4,
        )
    )

    for i, path in enumerate(paths):

        case = load_case(path)

        # --------------------------------------------------------------
        # Accurate wall-clock timing
        # --------------------------------------------------------------
        _sync_if_cuda(device)

        start = time.perf_counter()

        pred, info = reconstruct_frindel(
            case,
            cfg,
            device,
            lambda_t=lambda_t,
            lambda_s=lambda_s,
            delta=delta,
        )

        _sync_if_cuda(device)

        runtime_seconds = (
            time.perf_counter()
            - start
        )

        # --------------------------------------------------------------
        # Evaluation
        # --------------------------------------------------------------
        metric = evaluate_case(
            pred,
            case,
            eps_relative=float(
                cfg["evaluation"][
                    "eps_relative"
                ]
            ),
        )

        family = str(
            np.asarray(
                case.get(
                    "residue_family",
                    "unknown",
                )
            ).reshape(())
        )

        converged = bool(
            float(
                info[
                    "final_relative_change"
                ]
            )
            <= hq_tol
        )

        row = {
            "case_index": i,
            "case_file": path.name,
            "family": family,

            "lambda_t": lambda_t,
            "lambda_s": lambda_s,
            "delta": delta,

            "runtime_seconds":
                float(runtime_seconds),

            "converged":
                converged,

            **metric,
            **info,
        }

        rows.append(row)
        preds.append(pred)

        # --------------------------------------------------------------
        # Per-case progress
        # --------------------------------------------------------------
        print(
            f"{i + 1:03d}/{len(paths):03d} "
            f"{path.name} | "
            f"CBF={metric['cbf_absrel_percent']:.3f}% "
            f"CBV={metric['cbv_absrel_percent']:.3f}% "
            f"MTT={metric['mtt_absrel_percent']:.3f}% | "
            f"hNRMSE={metric['h_nrmse']:.5f} | "
            f"HQ={int(info['hq_iterations'])} "
            f"conv={'Y' if converged else 'N'} | "
            f"{runtime_seconds:.2f}s"
        )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    outdir = (
        output_root
        / "frindel_test"
    )

    if args.max_cases is not None:
        outdir = (
            output_root
            / f"frindel_test_smoke_{len(paths)}"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------
    np.savez_compressed(
        outdir
        / "predictions.npz",

        prediction=np.stack(
            preds
        ).astype(
            np.float32
        ),
    )

    # ------------------------------------------------------------------
    # Save per-case results
    # ------------------------------------------------------------------
    write_csv(
        outdir
        / "per_case.csv",
        rows,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = summarize(rows)

    (
        outdir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    # Also save CSV so it resembles the RED benchmark outputs.
    write_csv(
        outdir
        / "summary_absrel.csv",
        [summary],
    )

    (
        outdir
        / "selected_params.json"
    ).write_text(
        json.dumps(
            selected,
            indent=2,
        )
    )

    # ------------------------------------------------------------------
    # Final console summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    print(
        f"CBF AbsRel : "
        f"{summary['cbf_absrel_percent_mean']:.4f}"
        f" ± "
        f"{summary['cbf_absrel_percent_sd']:.4f}%"
    )

    print(
        f"CBV AbsRel : "
        f"{summary['cbv_absrel_percent_mean']:.4f}"
        f" ± "
        f"{summary['cbv_absrel_percent_sd']:.4f}%"
    )

    print(
        f"MTT AbsRel : "
        f"{summary['mtt_absrel_percent_mean']:.4f}"
        f" ± "
        f"{summary['mtt_absrel_percent_sd']:.4f}%"
    )

    print(
        f"h-NRMSE    : "
        f"{summary['h_nrmse_mean']:.6f}"
        f" ± "
        f"{summary['h_nrmse_sd']:.6f}"
    )

    print(
        f"Converged  : "
        f"{100.0 * summary['convergence_fraction']:.1f}%"
    )

    print(
        f"HQ iters   : "
        f"{summary['hq_iterations_mean']:.1f}"
        f" ± "
        f"{summary['hq_iterations_sd']:.1f}"
    )

    print(
        f"Final CG it: "
        f"{summary['final_cg_iterations_mean']:.1f}"
        f" ± "
        f"{summary['final_cg_iterations_sd']:.1f}"
    )

    print(
        f"Runtime    : "
        f"{summary['runtime_seconds_mean']:.4f}"
        f" ± "
        f"{summary['runtime_seconds_sd']:.4f} s/case"
    )

    print(
        "\nSaved:",
        outdir,
    )


if __name__ == "__main__":
    main()