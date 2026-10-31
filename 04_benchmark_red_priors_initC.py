#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from synthetic_utils import (
    batch_paths,
    evaluate_case,
    list_split,
    load_yaml,
    make_prior,
    mean_sd,
    resolve_device,
    resolve_repo_path,
    stack_cases,
    write_csv,
)


@torch.inference_mode()
def red_reconstruct_batch_init_c(
    *,
    A_np,
    C_np,
    shape_hwt,
    prior,
    strength,
    device,
    lambda_red,
    alpha_numerator,
    max_iter,
    min_iter,
    tol,
    normalize_forward=True,
):
    """
    Same RED-GD update as the project solver, except:
        H0 = normalized noisy C_obs
    instead of H0 = 0.
    """
    H, W, T = shape_hwt

    A_raw = torch.as_tensor(
        A_np,
        dtype=torch.float32,
        device=device,
    )

    C_raw = torch.as_tensor(
        C_np.reshape(len(C_np), H * W, T),
        dtype=torch.float32,
        device=device,
    )

    B = A_raw.shape[0]

    if normalize_forward:
        forward_scale = torch.amax(
            torch.abs(A_raw),
            dim=(1, 2),
        ).clamp_min(1.0e-8)

        A = A_raw / forward_scale[:, None, None]
        C = C_raw / forward_scale[:, None, None]
    else:
        forward_scale = torch.ones(
            B,
            dtype=torch.float32,
            device=device,
        )
        A = A_raw
        C = C_raw

    AtA = torch.bmm(
        A.transpose(1, 2),
        A,
    )

    L = torch.linalg.eigvalsh(
        AtA
    ).amax(dim=1)

    alpha = (
        float(alpha_numerator)
        / (L + float(lambda_red))
    )

    # ONLY solver change.
    x = C.clone()

    active = torch.ones(
        B,
        dtype=torch.bool,
        device=device,
    )
    converged = torch.zeros(
        B,
        dtype=torch.bool,
        device=device,
    )
    iterations = torch.full(
        (B,),
        int(max_iter),
        dtype=torch.int64,
        device=device,
    )
    rel_final = torch.full(
        (B,),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )

    start = time.perf_counter()

    for it in range(
        1,
        int(max_iter) + 1,
    ):
        idx = torch.nonzero(
            active,
            as_tuple=False,
        ).squeeze(1)

        if idx.numel() == 0:
            break

        old = x.index_select(
            0,
            idx,
        )

        den_hwt = prior(
            old.reshape(
                -1,
                H,
                W,
                T,
            ),
            strength,
        )

        den = den_hwt.reshape(
            -1,
            H * W,
            T,
        )

        Ai = A.index_select(
            0,
            idx,
        )

        Ci = C.index_select(
            0,
            idx,
        )

        pred_c = torch.bmm(
            old,
            Ai.transpose(1, 2),
        )

        grad = torch.bmm(
            pred_c - Ci,
            Ai,
        )

        ai = alpha.index_select(
            0,
            idx,
        )[:, None, None]

        new = old - ai * (
            grad
            + float(lambda_red)
            * (old - den)
        )

        num = torch.linalg.vector_norm(
            (new - old).flatten(1),
            dim=1,
        )

        denom = torch.maximum(
            torch.linalg.vector_norm(
                new.flatten(1),
                dim=1,
            ),
            torch.linalg.vector_norm(
                old.flatten(1),
                dim=1,
            ),
        )

        rc = num / (
            denom + 1.0e-12
        )

        x.index_copy_(
            0,
            idx,
            new,
        )
        rel_final.index_copy_(
            0,
            idx,
            rc,
        )

        if it >= int(min_iter):
            done = (
                rc < float(tol)
            )

            if torch.any(done):
                done_idx = idx[done]
                converged[done_idx] = True
                active[done_idx] = False
                iterations[done_idx] = it

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    runtime = (
        time.perf_counter()
        - start
    )

    pred = (
        x.reshape(
            B,
            H,
            W,
            T,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    infos = []

    for b in range(B):
        pred_c = (
            x[b] @ A[b].T
        )

        residual = float(
            torch.linalg.norm(
                pred_c - C[b]
            )
            / (
                torch.linalg.norm(
                    C[b]
                )
                + 1.0e-12
            )
        )

        infos.append(
            {
                "iterations":
                    int(
                        iterations[b]
                        .item()
                    ),
                "converged":
                    bool(
                        converged[b]
                        .item()
                    ),
                "rel_change":
                    float(
                        rel_final[b]
                        .item()
                    ),
                "L_f":
                    float(
                        L[b].item()
                    ),
                "alpha":
                    float(
                        alpha[b].item()
                    ),
                "forward_scale":
                    float(
                        forward_scale[b]
                        .item()
                    ),
                "data_residual":
                    residual,
                "effective_runtime_seconds":
                    float(
                        runtime / B
                    ),
            }
        )

    return pred, infos


def recover_existing_params(cfg, method):
    """
    Parameter priority:
      1. config.test_red_params.<method>
      2. previous H0=0 test per_case.csv (unique lambda/prior_strength)

    This avoids inventing a BM4D parameter if an existing tested one exists.
    """
    test_cfg = cfg.get(
        "test_red_params",
        {},
    ).get(
        method,
        None,
    )

    if test_cfg is not None:
        return (
            float(
                test_cfg["lambda_red"]
            ),
            float(
                test_cfg["strength"]
            ),
            "config.test_red_params",
        )

    output_root = resolve_repo_path(
        cfg["paths"]["output_root"]
    )

    csv_path = (
        output_root
        / "red_priors_test"
        / method
        / "per_case.csv"
    )

    if not csv_path.exists():
        return None

    pairs = set()

    with csv_path.open(
        "r",
        newline="",
    ) as f:
        for row in csv.DictReader(f):
            if (
                "lambda_red" in row
                and "prior_strength" in row
            ):
                pairs.add(
                    (
                        float(
                            row[
                                "lambda_red"
                            ]
                        ),
                        float(
                            row[
                                "prior_strength"
                            ]
                        ),
                    )
                )

    if len(pairs) != 1:
        raise RuntimeError(
            f"{csv_path}: expected one unique "
            f"(lambda_red, prior_strength), found {pairs}"
        )

    lam, strength = next(
        iter(pairs)
    )

    return (
        lam,
        strength,
        str(csv_path),
    )


def summarize(rows):
    out = {
        "n_cases": len(rows),
    }

    for key in (
        "cbf_absrel_percent",
        "cbv_absrel_percent",
        "mtt_absrel_percent",
        "h_nrmse",
        "iterations",
        "effective_runtime_seconds",
    ):
        mu, sd = mean_sd(
            [r[key] for r in rows]
        )
        out[key + "_mean"] = mu
        out[key + "_sd"] = sd

    out["convergence_fraction"] = float(
        np.mean(
            [
                float(
                    r["converged"]
                )
                for r in rows
            ]
        )
    )

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="config_v3_model_consistent.yml",
    )

    ap.add_argument(
        "--method",
        choices=[
            "ccd",
            "tv",
            "bm4d",
        ],
        required=True,
    )

    ap.add_argument(
        "--lambda-red",
        type=float,
        default=None,
    )

    ap.add_argument(
        "--strength",
        type=float,
        default=None,
    )

    ap.add_argument(
        "--max-cases",
        type=int,
        default=None,
    )

    args = ap.parse_args()

    cfg = load_yaml(
        args.config
    )

    device = resolve_device(
        cfg["device"]
    )

    red = cfg["red"]

    # ------------------------------------------------------------
    # Resolve parameters.
    # ------------------------------------------------------------
    recovered = recover_existing_params(
        cfg,
        args.method,
    )

    if (
        args.lambda_red is not None
        or args.strength is not None
    ):
        if (
            args.lambda_red is None
            or args.strength is None
        ):
            raise ValueError(
                "If overriding parameters, pass BOTH "
                "--lambda-red and --strength."
            )

        lambda_red = float(
            args.lambda_red
        )
        strength = float(
            args.strength
        )
        param_source = (
            "command-line override"
        )

    else:
        if recovered is None:
            raise RuntimeError(
                f"No existing parameters found for {args.method}. "
                "Pass --lambda-red and --strength explicitly."
            )

        (
            lambda_red,
            strength,
            param_source,
        ) = recovered

    # ------------------------------------------------------------
    # Test set.
    # ------------------------------------------------------------
    data_root = resolve_repo_path(
        cfg["paths"]["data_root"]
    )

    paths = list_split(
        data_root,
        "test",
    )

    if args.max_cases is not None:
        paths = paths[
            : int(
                args.max_cases
            )
        ]

    batch_size = int(
        cfg["runtime"]["batch_size"]
    )

    prior = make_prior(
        args.method,
        cfg=cfg,
        device=device,
    )

    print("=" * 80)
    print("SYNTHETIC V3 RED TEST — H0 = NORMALIZED C_obs")
    print("=" * 80)
    print("method        :", args.method)
    print("n_cases       :", len(paths))
    print("lambda_red    :", lambda_red)
    print("strength      :", strength)
    print("param source  :", param_source)
    print("initialization: H0 = normalized noisy C_obs")
    print("normalized    :", bool(
        red.get(
            "normalize_forward",
            True,
        )
    ))
    print("max_iter      :", red["max_iter"])
    print("tol           :", red["tol"])

    rows = []
    predictions = []
    case_index = 0

    for pbatch in batch_paths(
        paths,
        batch_size,
    ):
        batch = stack_cases(
            pbatch
        )

        B, H, W, T = (
            batch["h_gt"].shape
        )

        pred, infos = (
            red_reconstruct_batch_init_c(
                A_np=batch["A"],
                C_np=batch["C"],
                shape_hwt=(H, W, T),
                prior=prior,
                strength=strength,
                device=device,
                lambda_red=lambda_red,
                alpha_numerator=float(
                    red[
                        "alpha_numerator"
                    ]
                ),
                max_iter=int(
                    red["max_iter"]
                ),
                min_iter=int(
                    red["min_iter"]
                ),
                tol=float(
                    red["tol"]
                ),
                normalize_forward=bool(
                    red.get(
                        "normalize_forward",
                        True,
                    )
                ),
            )
        )

        predictions.append(
            pred
        )

        for j, case in enumerate(
            batch["cases"]
        ):
            metric = evaluate_case(
                pred[j],
                case,
                eps_relative=float(
                    cfg[
                        "evaluation"
                    ][
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

            rows.append(
                {
                    "case_index":
                        case_index,
                    "case_file":
                        batch[
                            "paths"
                        ][j].name,
                    "family":
                        family,
                    "method":
                        f"red_{args.method}",
                    "lambda_red":
                        lambda_red,
                    "prior_strength":
                        strength,
                    "initialization":
                        "normalized_C_obs",
                    **infos[j],
                    **metric,
                }
            )

            case_index += 1

    pred_array = np.concatenate(
        predictions,
        axis=0,
    )

    output_root = resolve_repo_path(
        cfg["paths"]["output_root"]
    )

    suffix = (
        f"{args.method}_init_c"
    )

    if args.max_cases is None:
        outdir = (
            output_root
            / "red_priors_test"
            / suffix
        )
    else:
        outdir = (
            output_root
            / "red_priors_test"
            / (
                suffix
                + f"_smoke_{len(paths)}"
            )
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        outdir / "per_case.csv",
        rows,
    )

    summary = summarize(
        rows
    )

    write_csv(
        outdir
        / "summary_absrel.csv",
        [
            {
                "method":
                    f"red_{args.method}",
                "lambda_red":
                    lambda_red,
                "strength":
                    strength,
                "initialization":
                    "normalized_C_obs",
                **summary,
            }
        ],
    )

    np.savez_compressed(
        outdir
        / "predictions.npz",
        prediction=
            pred_array.astype(
                np.float32
            ),
    )

    (
        outdir
        / "selected_params.json"
    ).write_text(
        json.dumps(
            {
                "method":
                    args.method,
                "lambda_red":
                    lambda_red,
                "strength":
                    strength,
                "parameter_source":
                    param_source,
                "initialization":
                    "H0 = normalized C_obs",
                "n_test_cases":
                    len(paths),
            },
            indent=2,
        )
    )

    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    print(
        "CBF AbsRel : "
        f"{summary['cbf_absrel_percent_mean']:.4f} "
        f"± {summary['cbf_absrel_percent_sd']:.4f}%"
    )

    print(
        "CBV AbsRel : "
        f"{summary['cbv_absrel_percent_mean']:.4f} "
        f"± {summary['cbv_absrel_percent_sd']:.4f}%"
    )

    print(
        "MTT AbsRel : "
        f"{summary['mtt_absrel_percent_mean']:.4f} "
        f"± {summary['mtt_absrel_percent_sd']:.4f}%"
    )

    print(
        "h-NRMSE    : "
        f"{summary['h_nrmse_mean']:.6f} "
        f"± {summary['h_nrmse_sd']:.6f}"
    )

    print(
        "Converged  : "
        f"{100.0 * summary['convergence_fraction']:.1f}%"
    )

    print(
        "Iterations : "
        f"{summary['iterations_mean']:.1f} "
        f"± {summary['iterations_sd']:.1f}"
    )

    print(
        "Runtime    : "
        f"{summary['effective_runtime_seconds_mean']:.4f} "
        f"± {summary['effective_runtime_seconds_sd']:.4f} s/case"
    )

    print()
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
