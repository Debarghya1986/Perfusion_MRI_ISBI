from __future__ import annotations

import numpy as np


def perfusion_maps_from_h(h_net, h_scale: float, dt: float, mask=None):
    hp = np.maximum(np.asarray(h_net, np.float64) * float(h_scale), 0.0)
    cbf = 60.0 * np.max(hp, axis=-1)
    cbv = np.trapezoid(hp, dx=float(dt), axis=-1)
    mtt = 60.0 * cbv / np.maximum(cbf, 1e-12)
    if mask is not None:
        m = np.asarray(mask) > 0
        cbf = cbf * m; cbv = cbv * m; mtt = mtt * m
    return cbf.astype(np.float32), cbv.astype(np.float32), mtt.astype(np.float32)


def _nrmse(pred, gt, mask):
    m = np.asarray(mask) > 0
    p, g = np.asarray(pred, np.float64)[m], np.asarray(gt, np.float64)[m]
    return float(np.linalg.norm(p-g) / (np.linalg.norm(g) + 1e-12))


def _mae(pred, gt, mask):
    m = np.asarray(mask) > 0
    return float(np.mean(np.abs(np.asarray(pred)[m] - np.asarray(gt)[m])))


def _absrel(pred, gt, mask, eps=1e-8):
    m = np.asarray(mask) > 0
    p, g = np.asarray(pred, np.float64)[m], np.asarray(gt, np.float64)[m]
    return float(np.mean(np.abs(p-g) / np.maximum(np.abs(g), eps)))


def evaluate_case(h_pred, case, eps_relative=1e-8):
    t = np.asarray(case["t"], np.float64)
    dt = float(t[1]-t[0]) if len(t) > 1 else 1.0
    cbf, cbv, mtt = perfusion_maps_from_h(h_pred, case["h_scale"], dt, case["mask"])
    out = {}
    for name, pred, gt in (
        ("cbf", cbf, case["cbf_gt"]),
        ("cbv", cbv, case["cbv_gt"]),
        ("mtt", mtt, case["mtt_gt"]),
    ):
        out[f"{name}_nrmse"] = _nrmse(pred, gt, case["mask"])
        out[f"{name}_mae"] = _mae(pred, gt, case["mask"])
        out[f"{name}_absrel"] = _absrel(pred, gt, case["mask"], eps_relative)
        out[f"{name}_absrel_percent"] = 100.0 * out[f"{name}_absrel"]
    out["mean_map_nrmse"] = (out["cbf_nrmse"] + out["cbv_nrmse"] + out["mtt_nrmse"]) / 3.0
    m = np.asarray(case["mask"]) > 0
    out["h_nrmse"] = float(
        np.linalg.norm((np.asarray(h_pred)-case["h_gt"])[m]) /
        (np.linalg.norm(np.asarray(case["h_gt"])[m]) + 1e-12)
    )
    out.update({"cbf_pred": cbf, "cbv_pred": cbv, "mtt_pred": mtt})
    return out


def tissue_metrics(h_pred, case, label_map=None):
    if label_map is None:
        label_map = {1: "healthy_gray", 2: "healthy_white", 3: "ischemic_core"}
    t = np.asarray(case["t"], np.float64)
    dt = float(t[1]-t[0]) if len(t) > 1 else 1.0
    cbf, cbv, mtt = perfusion_maps_from_h(h_pred, case["h_scale"], dt, case["mask"])
    rows = []
    labels = np.asarray(case["tissue_label"])
    for label, tissue in label_map.items():
        mask = labels == int(label)
        if not mask.any():
            continue
        row = {"tissue": tissue, "n_voxels": int(mask.sum())}
        for name, pred, gt in (("cbf",cbf,case["cbf_gt"]),("cbv",cbv,case["cbv_gt"]),("mtt",mtt,case["mtt_gt"])):
            row[f"{name}_mae"] = float(np.mean(np.abs(pred[mask]-gt[mask])))
            row[f"{name}_nrmse"] = float(np.linalg.norm(pred[mask]-gt[mask])/(np.linalg.norm(gt[mask])+1e-12))
        rows.append(row)
    return rows
