#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from synthetic_utils import (
    evaluate_case,
    list_split,
    load_case,
    load_spinned_model,
    load_yaml,
    mean_sd,
    reconstruct_meb,
    reconstruct_osvd,
    reconstruct_pinn,
    reconstruct_spinned,
    resolve_device,
    resolve_repo_path,
    write_csv,
)


PRIMARY = [
    "cbf_absrel_percent",
    "cbv_absrel_percent",
    "mtt_absrel_percent",
]

METRIC_FIELDS = [
    "cbf_absrel",
    "cbf_absrel_percent",
    "cbf_nrmse",
    "cbf_mae",
    "cbf_pearson",
    "cbv_absrel",
    "cbv_absrel_percent",
    "cbv_nrmse",
    "cbv_mae",
    "cbv_pearson",
    "mtt_absrel",
    "mtt_absrel_percent",
    "mtt_nrmse",
    "mtt_mae",
    "mtt_pearson",
    "h_nrmse",
    "tuning_objective",
    "gt_cbf_mean",
    "gt_cbv_mean",
    "gt_mtt_mean",
    "est_cbf_mean",
    "est_cbv_mean",
    "est_mtt_mean",
]

DISPLAY_NAMES = {
    "red_ccd": "RED-CCD",
    "red_bm4d": "RED-BM4D",
    "red_tv": "RED-TV",
    "frindel": "Frindel-ST",
    "osvd": "oSVD (20%)",
    "meb": "MEB",
    "pinn": "PINN",
    "spinned": "SPINNED-style",
    
}


def summarize(rows, methods):
    out = []

    for method in methods:
        rr = [
            r for r in rows
            if r["method"] == method
        ]

        if not rr:
            continue

        row = {
            "method": method,
            "display_name": DISPLAY_NAMES.get(
                method,
                method,
            ),
            "n_cases": len(rr),
        }

        for k in PRIMARY:
            mu, sd = mean_sd(
                [r[k] for r in rr]
            )
            row[k + "_mean"] = mu
            row[k + "_sd"] = sd

        for k in (
            "cbf_nrmse",
            "cbv_nrmse",
            "mtt_nrmse",
            "h_nrmse",
        ):
            mu, sd = mean_sd(
                [r[k] for r in rr]
            )
            row[k + "_mean"] = mu
            row[k + "_sd"] = sd

        mu, sd = mean_sd(
            [
                r["tuning_objective"]
                for r in rr
            ]
        )

        row[
            "objective_mean_absrel_mean"
        ] = mu
        row[
            "objective_mean_absrel_sd"
        ] = sd
        row[
            "objective_mean_absrel_percent_mean"
        ] = 100.0 * mu
        row[
            "objective_mean_absrel_percent_sd"
        ] = 100.0 * sd

        timed = [
            r["runtime_seconds"]
            for r in rr
            if np.isfinite(
                r.get(
                    "runtime_seconds",
                    np.nan,
                )
            )
        ]

        if timed:
            mu, sd = mean_sd(timed)
            row["runtime_seconds_mean"] = mu
            row["runtime_seconds_sd"] = sd
        else:
            row["runtime_seconds_mean"] = float("nan")
            row["runtime_seconds_sd"] = float("nan")

        out.append(row)

    return out


def summarize_by_family(
    rows,
    methods,
):
    out = []

    families = sorted(
        {
            str(r["family"])
            for r in rows
        }
    )

    for family in families:
        for method in methods:
            rr = [
                r for r in rows
                if (
                    r["method"] == method
                    and str(r["family"]) == family
                )
            ]

            if not rr:
                continue

            row = {
                "family": family,
                "method": method,
                "display_name": DISPLAY_NAMES.get(
                    method,
                    method,
                ),
                "n_cases": len(rr),
            }

            for k in PRIMARY:
                mu, sd = mean_sd(
                    [r[k] for r in rr]
                )
                row[k + "_mean"] = mu
                row[k + "_sd"] = sd

            mu, sd = mean_sd(
                [r["h_nrmse"] for r in rr]
            )
            row["h_nrmse_mean"] = mu
            row["h_nrmse_sd"] = sd

            mu, sd = mean_sd(
                [
                    r["tuning_objective"]
                    for r in rr
                ]
            )
            row[
                "objective_mean_absrel_mean"
            ] = mu
            row[
                "objective_mean_absrel_sd"
            ] = sd
            row[
                "objective_mean_absrel_percent_mean"
            ] = 100.0 * mu
            row[
                "objective_mean_absrel_percent_sd"
            ] = 100.0 * sd

            out.append(row)

    return out


def publication_family_rows(
    family_rows,
    family,
):
    out = []

    for r in family_rows:
        if r["family"] != family:
            continue

        out.append(
            {
                "method": r["display_name"],
                "n_cases": r["n_cases"],
                "cbf_absrel_percent_mean":
                    r["cbf_absrel_percent_mean"],
                "cbf_absrel_percent_sd":
                    r["cbf_absrel_percent_sd"],
                "cbv_absrel_percent_mean":
                    r["cbv_absrel_percent_mean"],
                "cbv_absrel_percent_sd":
                    r["cbv_absrel_percent_sd"],
                "mtt_absrel_percent_mean":
                    r["mtt_absrel_percent_mean"],
                "mtt_absrel_percent_sd":
                    r["mtt_absrel_percent_sd"],
            }
        )

    return out


def absolute_summary(
    rows,
    methods,
):
    out = []

    first = next(
        m for m in methods
        if any(
            r["method"] == m
            for r in rows
        )
    )

    base = [
        r for r in rows
        if r["method"] == first
    ]

    gt = {
        "method": "GT",
        "n_cases": len(base),
    }

    for p in (
        "cbf",
        "cbv",
        "mtt",
    ):
        mu, sd = mean_sd(
            [
                r[f"gt_{p}_mean"]
                for r in base
            ]
        )
        gt[
            f"{p}_case_mean_mean"
        ] = mu
        gt[
            f"{p}_case_mean_sd"
        ] = sd

    out.append(gt)

    for method in methods:
        rr = [
            r for r in rows
            if r["method"] == method
        ]

        if not rr:
            continue

        row = {
            "method": method,
            "n_cases": len(rr),
        }

        for p in (
            "cbf",
            "cbv",
            "mtt",
        ):
            mu, sd = mean_sd(
                [
                    r[
                        f"est_{p}_mean"
                    ]
                    for r in rr
                ]
            )
            row[
                f"{p}_case_mean_mean"
            ] = mu
            row[
                f"{p}_case_mean_sd"
            ] = sd

        out.append(row)

    return out


def load_red_predictions(
    cfg,
    methods,
    n_cases,
):
    output_root = resolve_repo_path(
        cfg["paths"]["output_root"]
    )

    result = {}

    for method in methods:
        if not method.startswith(
            "red_"
        ):
            continue

        prior_name = method.removeprefix(
            "red_"
        )

        pred_file = (
            output_root
            / "red_priors_test"
            / prior_name
            / "predictions.npz"
        )

        if not pred_file.exists():
            raise FileNotFoundError(
                f"{pred_file} not found. "
                "Run the full 120-case "
                "04_benchmark_red_priors.py "
                f"--method {prior_name} first."
            )

        with np.load(
            pred_file
        ) as d:
            if "prediction" not in d.files:
                raise KeyError(
                    f"{pred_file} does not "
                    "contain key 'prediction'."
                )

            arr = np.asarray(
                d["prediction"],
                dtype=np.float32,
            )

        if arr.ndim != 4:
            raise ValueError(
                f"{method}: expected "
                f"[N,H,W,T], got {arr.shape}."
            )

        if arr.shape[0] < n_cases:
            raise RuntimeError(
                f"{method}: only "
                f"{arr.shape[0]} saved "
                "predictions, but this run "
                f"needs {n_cases}."
            )

        result[method] = arr

        print(
            f"Loaded {method}: "
            f"{arr.shape}"
        )

    return result


def _parse_csv_value(value):
    if value is None:
        return value

    text = str(value).strip()

    if text.lower() == "true":
        return True

    if text.lower() == "false":
        return False

    try:
        return float(text)
    except ValueError:
        return text


def load_pinn_audit(
    cfg,
):
    rel = cfg["methods"].get(
        "pinn_results_csv",
        None,
    )

    if not rel:
        return None, None

    path = resolve_repo_path(rel)

    if not path.exists():
        return None, path

    rows = {}

    with path.open(
        "r",
        newline="",
    ) as f:
        for raw in csv.DictReader(f):
            parsed = {
                k: _parse_csv_value(v)
                for k, v in raw.items()
            }

            case_file = str(
                parsed["case_file"]
            )

            rows[case_file] = parsed

    return rows, path


def metric_from_pinn_audit(
    audit_row,
):
    missing = [
        k
        for k in METRIC_FIELDS
        if k not in audit_row
    ]

    if missing:
        raise KeyError(
            "PINN audit CSV is missing "
            f"metric fields: {missing}"
        )

    return {
        k: float(audit_row[k])
        for k in METRIC_FIELDS
    }


def print_result_table(
    title,
    summary,
):
    print(
        "\n"
        + "=" * 113
    )
    print(title)
    print("=" * 113)

    header = (
        f"{'Method':<16}"
        f"{'CBF AbsRel (%)':>20}"
        f"{'CBV AbsRel (%)':>20}"
        f"{'MTT AbsRel (%)':>20}"
        f"{'h-NRMSE':>20}"
        f"{'Mean Obj (%)':>17}"
    )

    print(header)
    print("-" * len(header))

    for r in summary:
        print(
            f"{r['display_name']:<16}"
            f"{r['cbf_absrel_percent_mean']:>8.4f}"
            f" ± {r['cbf_absrel_percent_sd']:<8.4f}"
            f"{r['cbv_absrel_percent_mean']:>8.4f}"
            f" ± {r['cbv_absrel_percent_sd']:<8.4f}"
            f"{r['mtt_absrel_percent_mean']:>8.4f}"
            f" ± {r['mtt_absrel_percent_sd']:<8.4f}"
            f"{r['h_nrmse_mean']:>9.5f}"
            f" ± {r['h_nrmse_sd']:<8.5f}"
            f"{r['objective_mean_absrel_percent_mean']:>12.4f}"
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default=str(
            Path(__file__).with_name(
                "config_v3_model_consistent.yml"
            )
        ),
    )

    ap.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help=(
            "Smoke-test only. Partial runs "
            "are written to a separate "
            "output directory."
        ),
    )

    ap.add_argument(
        "--force-rerun-pinn",
        action="store_true",
        help=(
            "Ignore methods.pinn_results_csv "
            "and refit PINN inside this script."
        ),
    )

    args = ap.parse_args()

    cfg = load_yaml(
        args.config
    )

    device = resolve_device(
        cfg["device"]
    )

    data_root = resolve_repo_path(
        cfg["paths"]["data_root"]
    )

    all_paths = list_split(
        data_root,
        "test",
    )

    if args.max_cases is None:
        paths = all_paths
    else:
        paths = all_paths[
            : int(
                args.max_cases
            )
        ]

    methods = list(
        cfg["methods"]["include"]
    )

    dropped = {
        "csvd",
        "tisvd",
    }

    accidentally_present = sorted(
        dropped.intersection(
            methods
        )
    )

    if accidentally_present:
        raise ValueError(
            "Remove dropped method(s) "
            "from config_v3_model_consistent.yml: "
            f"{accidentally_present}"
        )

    if bool(
        cfg["methods"][
            "spinned"
        ].get(
            "enabled",
            False,
        )
    ):
        if "spinned" not in methods:
            methods.append(
                "spinned"
            )

    if len(set(methods)) != len(
        methods
    ):
        raise ValueError(
            "Duplicate benchmark methods: "
            f"{methods}"
        )

    output_root = resolve_repo_path(
        cfg["paths"]["output_root"]
    )

    if args.max_cases is None:
        outdir = (
            output_root
            / "all_methods_test"
        )
    else:
        outdir = (
            output_root
            / (
                "all_methods_test_smoke_"
                f"{len(paths)}"
            )
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "SYNTHETIC V2 "
        "ALL-METHOD HELD-OUT TEST"
    )
    print("=" * 80)
    print("n_cases :", len(paths))
    print("device  :", device)
    print("methods :", methods)
    print("outdir  :", outdir)

    red_preds = load_red_predictions(
        cfg,
        methods,
        n_cases=len(paths),
    )

    frindel_preds = None

    if "frindel" in methods:
        frindel_file = (
            output_root
            / "frindel_test"
            / "predictions.npz"
        )

        if not frindel_file.exists():
            raise FileNotFoundError(
                f"{frindel_file} not found. "
                "Run 05_benchmark_frindel.py on the full test set first."
            )

        with np.load(frindel_file) as d:
            frindel_preds = np.asarray(
                d["prediction"],
                dtype=np.float32,
            )

        if frindel_preds.shape[0] < len(paths):
            raise RuntimeError(
                f"Frindel predictions have {frindel_preds.shape[0]} cases; "
                f"need {len(paths)}."
            )
            
            
    spinned_model = (
        load_spinned_model(
            cfg,
            device,
        )
    )

    pinn_audit = None
    pinn_audit_path = None

    if (
        "pinn" in methods
        and not args.force_rerun_pinn
    ):
        (
            pinn_audit,
            pinn_audit_path,
        ) = load_pinn_audit(
            cfg
        )

        if pinn_audit is not None:
            missing_cases = [
                p.name
                for p in paths
                if p.name
                not in pinn_audit
            ]

            if missing_cases:
                raise RuntimeError(
                    "PINN audit CSV exists but "
                    "does not contain all requested "
                    "cases. Missing examples: "
                    f"{missing_cases[:5]}"
                )

            print(
                "Reusing PINN metrics from:",
                pinn_audit_path,
            )

        else:
            print(
                "PINN audit CSV not available; "
                "PINN will be fitted in 05."
            )

    pinn_seed_base = int(
        cfg["methods"].get(
            "pinn_seed",
            0,
        )
    )

    rows = []

    # We save predictions for every method actually reconstructed here.
    # When PINN audit metrics are reused, 06_visual_reports.py reconstructs
    # only the two selected PINN cases on demand.
    prediction_store = {
        method: []
        for method in methods
        if not (
            method == "pinn"
            and pinn_audit
            is not None
        )
    }

    eps_relative = float(
        cfg["evaluation"][
            "eps_relative"
        ]
    )

    for i, path in enumerate(
        paths
    ):
        print(
            f"\nCase "
            f"{i + 1}/{len(paths)}: "
            f"{path.name}"
        )

        case = load_case(
            path
        )

        family = str(
            np.asarray(
                case.get(
                    "residue_family",
                    "unknown",
                )
            ).reshape(())
        )

        for method in methods:
            runtime_seconds = float(
                "nan"
            )
            pred = None

            if method.startswith(
                "red_"
            ):
                pred = np.asarray(
                    red_preds[
                        method
                    ][i],
                    dtype=np.float32,
                )

                metric = evaluate_case(
                    pred,
                    case,
                    eps_relative=
                        eps_relative,
                )
            elif method == "frindel":
                pred = np.asarray(
                    frindel_preds[i],
                    dtype=np.float32,
                )

                metric = evaluate_case(
                    pred,
                    case,
                    eps_relative=eps_relative,
                )
                
            elif (
                method == "pinn"
                and pinn_audit
                is not None
            ):
                audit_row = pinn_audit[
                    path.name
                ]

                metric = (
                    metric_from_pinn_audit(
                        audit_row
                    )
                )

                runtime_seconds = float(
                    audit_row.get(
                        "runtime_seconds",
                        np.nan,
                    )
                )

            else:
                start = time.perf_counter()

                if method == "osvd":
                    pred = (
                        reconstruct_osvd(
                            case,
                            cfg,
                        )
                    )

                elif method == "meb":
                    pred = (
                        reconstruct_meb(
                            case,
                            cfg,
                        )
                    )

                elif method == "pinn":
                    pred = (
                        reconstruct_pinn(
                            case,
                            cfg,
                            device,
                            seed=(
                                pinn_seed_base
                                + 1000 * i
                            ),
                        )
                    )

                elif method == "spinned":
                    pred = (
                        reconstruct_spinned(
                            case,
                            spinned_model,
                            device,
                        )
                    )

                else:
                    raise ValueError(
                        "Unknown benchmark "
                        f"method: {method}"
                    )

                if device.type == "cuda":
                    torch.cuda.synchronize(
                        device
                    )

                runtime_seconds = (
                    time.perf_counter()
                    - start
                )

                pred = np.asarray(
                    pred,
                    dtype=np.float32,
                )

                if pred.shape != (
                    case["h_gt"].shape
                ):
                    raise ValueError(
                        f"{method} returned "
                        f"{pred.shape}; expected "
                        f"{case['h_gt'].shape}."
                    )

                if not np.isfinite(
                    pred
                ).all():
                    raise FloatingPointError(
                        f"{method} produced "
                        "non-finite values for "
                        f"{path.name}."
                    )

                metric = evaluate_case(
                    pred,
                    case,
                    eps_relative=
                        eps_relative,
                )

            rows.append(
                {
                    "case_index": i,
                    "case_file": path.name,
                    "family": family,
                    "method": method,
                    "runtime_seconds":
                        runtime_seconds,
                    **metric,
                }
            )

            if (
                pred is not None
                and method
                in prediction_store
            ):
                prediction_store[
                    method
                ].append(pred)

            print(
                f"  {method:<10} "
                f"CBF="
                f"{metric['cbf_absrel_percent']:.3f}% "
                f"CBV="
                f"{metric['cbv_absrel_percent']:.3f}% "
                f"MTT="
                f"{metric['mtt_absrel_percent']:.3f}% "
                f"hNRMSE="
                f"{metric['h_nrmse']:.5f}"
            )

    summary = summarize(
        rows,
        methods,
    )

    family_summary = (
        summarize_by_family(
            rows,
            methods,
        )
    )

    write_csv(
        outdir / "per_case.csv",
        rows,
    )

    write_csv(
        outdir / "summary_absrel.csv",
        summary,
    )

    write_csv(
        outdir / "summary_by_family.csv",
        family_summary,
    )

    write_csv(
        outdir
        / "table_biexponential.csv",
        publication_family_rows(
            family_summary,
            "biexponential",
        ),
    )

    write_csv(
        outdir
        / "table_pharmacokinetic.csv",
        publication_family_rows(
            family_summary,
            "pharmacokinetic",
        ),
    )

    write_csv(
        outdir
        / "summary_absolute_values.csv",
        absolute_summary(
            rows,
            methods,
        ),
    )

    arrays_to_save = {}

    for method, values in (
        prediction_store.items()
    ):
        if len(values) != len(paths):
            continue

        arrays_to_save[
            method
        ] = np.stack(
            values,
            axis=0,
        ).astype(
            np.float32
        )

    np.savez_compressed(
        outdir
        / "predictions.npz",
        **arrays_to_save,
    )

    print_result_table(
        "OVERALL BALANCED TEST SET",
        summary,
    )

    for family, title in (
        (
            "biexponential",
            "BI-EXPONENTIAL TEST SET",
        ),
        (
            "pharmacokinetic",
            "PHARMACOKINETIC TEST SET",
        ),
    ):
        fam_rows = [
            r
            for r in family_summary
            if r["family"] == family
        ]

        print_result_table(
            title,
            fam_rows,
        )

    print(
        "\nSaved:",
        outdir,
    )

    print(
        "Prediction arrays saved for:",
        sorted(
            arrays_to_save
        ),
    )

    if (
        "pinn" in methods
        and "pinn"
        not in arrays_to_save
    ):
        print(
            "PINN metrics were reused from "
            "the standalone audit. "
            "06_visual_reports.py will "
            "reconstruct only the two "
            "representative PINN cases."
        )

    if args.max_cases is not None:
        print(
            "\nNOTE: smoke test only. "
            "Run without --max-cases "
            "for the final 120-case table."
        )


if __name__ == "__main__":
    main()
