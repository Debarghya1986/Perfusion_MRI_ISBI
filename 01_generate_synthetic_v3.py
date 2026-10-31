#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm

from v3.utils import load_yaml, resolve_path

LABEL_BACKGROUND = 0
LABEL_GM = 1
LABEL_WM = 2
LABEL_CORE = 3


def _ellipse(H, W, cy, cx, ry, rx, angle=0.0):
    yy, xx = np.mgrid[0:H, 0:W]
    c, s = math.cos(angle), math.sin(angle)
    x = xx - cx
    y = yy - cy
    xr = c * x + s * y
    yr = -s * x + c * y
    return (xr / max(rx, 1e-6)) ** 2 + (yr / max(ry, 1e-6)) ** 2 <= 1.0


def shape_mask(H: int, W: int, rng: np.random.Generator, archetype: str):
    """Literature-inspired 3-class brain mask.

    Giacalone et al. used I-KNOW-derived label masks with healthy GM, healthy WM,
    and ischemic core, including small/large compact and medium/large fragmented
    lesions. Those original masks are not distributed with the paper; this
    function creates reproducible geometric analogues of the four archetypes.
    """
    cy = 0.50 * (H - 1) + rng.uniform(-0.03, 0.03) * H
    cx = 0.50 * (W - 1) + rng.uniform(-0.03, 0.03) * W
    brain = _ellipse(H, W, cy, cx, 0.43 * H, 0.43 * W, rng.uniform(-0.08, 0.08))
    wm = _ellipse(H, W, cy, cx, 0.31 * H, 0.30 * W, rng.uniform(-0.10, 0.10)) & brain
    gm = brain & (~wm)

    # Lesion center kept within brain; lesion overrides GM/WM labels.
    lcy = cy + rng.uniform(-0.15, 0.15) * H
    lcx = cx + rng.uniform(-0.18, 0.18) * W
    if "small" in archetype:
        scale = 0.10
    elif "medium" in archetype:
        scale = 0.15
    else:
        scale = 0.21

    if "compact" in archetype:
        lesion = _ellipse(
            H, W, lcy, lcx,
            scale * H,
            (scale * rng.uniform(0.7, 1.3)) * W,
            rng.uniform(-0.9, 0.9),
        )
    else:
        lesion = np.zeros((H, W), dtype=bool)
        nfrag = 3 if "medium" in archetype else 5
        for _ in range(nfrag):
            fcy = lcy + rng.uniform(-0.08, 0.08) * H
            fcx = lcx + rng.uniform(-0.08, 0.08) * W
            fs = scale * rng.uniform(0.35, 0.65)
            lesion |= _ellipse(
                H, W, fcy, fcx,
                fs * H,
                fs * rng.uniform(0.7, 1.3) * W,
                rng.uniform(-1.0, 1.0),
            )
    lesion &= brain

    labels = np.zeros((H, W), np.uint8)
    labels[gm] = LABEL_GM
    labels[wm] = LABEL_WM
    labels[lesion] = LABEL_CORE
    return labels


def sample_truncnorm(rng, mean, sd, lo, hi, size):
    a = (lo - mean) / sd
    b = (hi - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size, random_state=rng).astype(np.float32)


def hemodynamic_maps(labels, cfg, rng):
    tc = cfg["tissue_classes"]
    cbf = np.zeros(labels.shape, np.float32)
    mtt_target = np.zeros(labels.shape, np.float32)
    spec = {
        LABEL_GM: tc["healthy_gray"],
        LABEL_WM: tc["healthy_white"],
        LABEL_CORE: tc["ischemic_core"],
    }
    for label, s in spec.items():
        idx = labels == label
        n = int(idx.sum())
        if n == 0:
            continue
        cbf[idx] = sample_truncnorm(
            rng, float(s["cbf_mean"]), float(s["cbf_sd"]),
            float(tc["cbf_lower"]), float(tc["cbf_upper"]), n,
        )
        mtt_target[idx] = sample_truncnorm(
            rng, float(s["mtt_mean"]), float(s["mtt_sd"]),
            float(tc["mtt_lower"]), float(tc["mtt_upper"]), n,
        )
    return cbf, mtt_target


def base_residue(u: np.ndarray, family: str, cfg):
    if family == "biexponential":
        p = cfg["biexponential"]
        A = float(p["A"])
        q = A * np.exp(-float(p["tau1"]) * u) + (1 - A) * np.exp(-float(p["tau2"]) * u)
    elif family == "pharmacokinetic":
        p = cfg["pharmacokinetic"]
        l2, l3 = float(p["lambda2"]), float(p["lambda3"])
        mtt0 = float(p["reference_mtt"])
        l1 = (l3 - l2) / (mtt0 * l2 * l3)
        d21 = l2 - l1
        d31 = l3 - l1
        if abs(d21) < 1e-9 or abs(d31) < 1e-9:
            raise ValueError("Degenerate pharmacokinetic rate configuration.")
        q = (
            (np.exp(-l1 * u) - np.exp(-l2 * u)) / d21
            - (np.exp(-l1 * u) - np.exp(-l3 * u)) / d31
        )
        q = np.maximum(q, 0.0)
    else:
        raise ValueError(family)
    mx = float(np.max(q))
    return q / max(mx, 1e-12)


def residue_volume(t: np.ndarray, mtt_target: np.ndarray, family: str, cfg):
    """Generate peak-normalized q(x,y,t) and time-warp to target MTT.

    Pizzolato supplies the bi-exponential and pharmacokinetic *shape families*.
    Giacalone supplies voxelwise MTT distributions. We connect them by a pure
    time dilation: q_v(t)=q0(a_v t), with a_v chosen so the continuous area of
    the peak-normalized base curve equals the target MTT. This is an explicit
    project adaptation and is recorded in metadata.
    """
    # Numerically estimate the full base-curve area once on a long fine grid.
    u = np.linspace(0.0, 300.0, 60001, dtype=np.float64)
    qb = base_residue(u, family, cfg)
    base_area = float(np.trapezoid(qb, u))

    target = np.maximum(mtt_target.astype(np.float64), 1e-3)
    scale = base_area / target  # integral q0(scale*t) dt = target
    tt = t[None, None, :].astype(np.float64) * scale[..., None]

    if family == "biexponential":
        p = cfg["biexponential"]
        A = float(p["A"])
        q = A * np.exp(-float(p["tau1"]) * tt) + (1 - A) * np.exp(-float(p["tau2"]) * tt)
        q /= np.maximum(np.max(q, axis=-1, keepdims=True), 1e-12)
    else:
        p = cfg["pharmacokinetic"]
        l2, l3 = float(p["lambda2"]), float(p["lambda3"])
        mtt0 = float(p["reference_mtt"])
        l1 = (l3 - l2) / (mtt0 * l2 * l3)
        d21 = l2 - l1
        d31 = l3 - l1
        q = (
            (np.exp(-l1 * tt) - np.exp(-l2 * tt)) / d21
            - (np.exp(-l1 * tt) - np.exp(-l3 * tt)) / d31
        )
        q = np.maximum(q, 0.0)
        q /= np.maximum(np.max(q, axis=-1, keepdims=True), 1e-12)
    return q.astype(np.float32), base_area


def gamma_variate_aif(t: np.ndarray, cfg):
    p = cfg["aif"]
    ymax, tmax, d, a = map(float, [p["ymax"], p["tmax"], p["arrival_d"], p["alpha"]])
    x = t - d
    y = np.zeros_like(t, dtype=np.float64)
    good = x > 0
    z = x[good] / tmax
    y[good] = ymax * (z ** a) * np.exp(a * (1.0 - z))
    return y.astype(np.float32)


def conv_matrix(aif: np.ndarray, dt: float):
    T = len(aif)
    A = np.zeros((T, T), np.float32)
    for i in range(T):
        A[i, : i + 1] = aif[i::-1]
    return float(dt) * A


def signal_domain_noise(C_phys: np.ndarray, brain_mask: np.ndarray, cfg, rng):
    p = cfg["noise"]
    S0 = float(p["S0"])
    TE = float(p["TE_s"])
    snr_db = float(p["snr_db"])
    target_drop = float(p["target_peak_signal_drop"])

    mean_curve = C_phys[brain_mask > 0].mean(axis=0)
    c_peak = max(float(np.max(mean_curve)), 1e-8)
    # exp(-k*TE*c_peak)=1-target_drop
    kappa = -math.log(max(1.0 - target_drop, 1e-6)) / (TE * c_peak)

    S = S0 * np.exp(-kappa * TE * C_phys)
    sigma_N = S0 / (10.0 ** (snr_db / 20.0))
    S_noisy = S + rng.normal(0.0, sigma_N, size=S.shape)
    S_noisy = np.maximum(S_noisy, 1e-4)
    C_obs = -np.log(S_noisy / S0) / (kappa * TE)
    C_obs[brain_mask == 0] = 0.0
    return C_obs.astype(np.float32), float(kappa), float(sigma_N)


def family_for_index(i: int, mode: str):
    mode = str(mode)
    if mode == "balanced_cases":
        return "biexponential" if i % 2 == 0 else "pharmacokinetic"
    if mode in ("biexponential", "pharmacokinetic"):
        return mode
    raise ValueError(mode)


def make_case(cfg, seed: int, case_index: int):
    rng = np.random.default_rng(seed)
    H, W = int(cfg["H"]), int(cfg["W"])
    dt, duration = float(cfg["dt"]), float(cfg["duration_s"])
    t = np.arange(0.0, duration + 0.5 * dt, dt, dtype=np.float32)

    archetypes = list(cfg["lesion_shapes"])
    # Pair consecutive bi-exp/PK cases with the same shape archetype so
    # temporal family and lesion geometry are not confounded.
    shape_model = archetypes[(case_index // 2) % len(archetypes)]
    labels = shape_mask(H, W, rng, shape_model)
    mask = (labels > 0).astype(np.float32)
    cbf, mtt_target = hemodynamic_maps(labels, cfg, rng)

    family = family_for_index(case_index, cfg["family_mode"])
    q, base_area = residue_volume(t, mtt_target, family, cfg)
    q *= mask[..., None]

    h_phys = (cbf[..., None] / 60.0) * q
    cbv = np.trapezoid(np.maximum(h_phys, 0.0), dx=dt, axis=-1).astype(np.float32)
    cbf_gt = (60.0 * np.max(np.maximum(h_phys, 0.0), axis=-1)).astype(np.float32)
    mtt_gt = (60.0 * cbv / np.maximum(cbf_gt, 1e-12)).astype(np.float32)
    cbf_gt *= mask; cbv *= mask; mtt_gt *= mask

    h_scale = float(cfg["cbf_reference"]) / 60.0
    h_net = (h_phys / h_scale).astype(np.float32)

    aif = gamma_variate_aif(t, cfg)
    A = conv_matrix(aif, dt)
    C_clean_phys = (h_phys.reshape(-1, len(t)) @ A.T).reshape(H, W, len(t)).astype(np.float32)
    C_obs_phys, kappa, sigma_N = signal_domain_noise(C_clean_phys, mask, cfg, rng)
    C_clean = (C_clean_phys / h_scale).astype(np.float32)
    C_obs = (C_obs_phys / h_scale).astype(np.float32)

    return {
        "clean": np.transpose(h_net, (2, 0, 1)),
        "A": A,
        "aif": aif,
        "t": t,
        "C_clean": C_clean,
        "C_obs": C_obs,
        "brain_mask": mask,
        "tissue_label": labels,
        "cbf_gt": cbf_gt,
        "cbv_gt": cbv,
        "mtt_gt": mtt_gt,
        "mtt_target": mtt_target.astype(np.float32),
        "h_scale": np.asarray(h_scale, np.float32),
        "residue_family": np.asarray(family),
        "shape_model": np.asarray(shape_model),
        "base_residue_area": np.asarray(base_area, np.float32),
        "signal_kappa": np.asarray(kappa, np.float32),
        "signal_noise_sd": np.asarray(sigma_N, np.float32),
    }


def write_split(root: Path, split: str, n: int, seed0: int, cfg):
    d = root / split
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        p = d / f"sample_{i:06d}.npz"
        if p.exists() and not bool(cfg.get("overwrite", False)):
            continue
        case = make_case(cfg, seed0 + i, i)
        np.savez_compressed(p, **case)
        if (i + 1) % max(1, n // 10) == 0:
            print(f"{split}: {i+1}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v3.yml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    all_cfg = load_yaml(args.config)
    cfg = dict(all_cfg["dataset"])
    if args.force:
        cfg["overwrite"] = True
    root = resolve_path(all_cfg["paths"]["data_root"])
    if root.exists() and bool(cfg.get("overwrite", False)):
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    seed = int(all_cfg.get("seed", 0))
    write_split(root, "train", int(cfg["train_size"]), seed, cfg)
    write_split(root, "val", int(cfg["val_size"]), seed + 1_000_000, cfg)
    write_split(root, "test", int(cfg["test_size"]), seed + 2_000_000, cfg)

    meta = {
        "generator": "Synthetic-V3",
        "literature_basis": {
            "spatial_hemodynamics": "Giacalone et al., Magn Reson Med 2017, DOI 10.1002/mrm.26573",
            "temporal_families": "Pizzolato et al., ISBI 2015, DOI 10.1109/ISBI.2015.7164057",
        },
        "important_adaptations": [
            "Exact I-KNOW label masks are unavailable; lesion geometry is generated from the four published compact/fragmented archetypes.",
            "Pizzolato residue families are time-dilated per voxel so their area/peak matches the Giacalone tissue-class MTT draw.",
            "The discrete forward model is evaluated directly at dt=1 s so the saved A matrix is exactly the operator used by all reconstruction methods.",
        ],
        "train_val_test": [int(cfg["train_size"]), int(cfg["val_size"]), int(cfg["test_size"])],
        "shape": [int(cfg["H"]), int(cfg["W"]), int(round(float(cfg["duration_s"])/float(cfg["dt"]))) + 1],
        "families": ["biexponential", "pharmacokinetic"],
        "tissue_labels": {"0": "background", "1": "healthy_gray", "2": "healthy_white", "3": "ischemic_core"},
        "measurement": "C_obs contains Giacalone-style MR signal-domain SNR noise; C_clean is retained for audit only.",
    }
    (root / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Synthetic V3 ready:", root)


if __name__ == "__main__":
    main()
