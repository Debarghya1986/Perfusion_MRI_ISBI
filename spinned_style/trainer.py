#!/usr/bin/env python3
"""Train the SPINNED-style benchmark on Synthetic V2.

Important fairness rules
------------------------
- train/validation/test case files remain disjoint;
- training uses only the training split;
- model selection uses only the validation split;
- the checkpoint-selection metric is the same perfusion-map objective used for
  RED prior tuning in Synthetic V2:

      mean(CBF AbsRel, CBV AbsRel, MTT AbsRel)

- h-NRMSE is recorded only as a secondary validation metric;
- the 120-case test split is NOT evaluated in this trainer. Final testing is
  performed later by 05_benchmark_all_methods.py together with every baseline;
- AIF, A, and C are divided by the same per-case max|AIF| scale, leaving h
  unchanged because C/s = h (A/s)^T.

This is the project's direct-fit SPINNED-style adaptation. It does not claim to
reproduce unreleased author code.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from model import SPINNEDStyleMLP, reconvolve_rowwise


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class CaseDataset(Dataset):
    def __init__(self, root: str, split: str) -> None:
        self.files = sorted((Path(root) / split).glob("sample_*.npz"))
        if not self.files:
            raise RuntimeError(f"No samples found in {Path(root) / split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        with np.load(self.files[index], allow_pickle=False) as d:
            h = np.asarray(d["clean"], dtype=np.float32)          # [T,H,W]
            C = np.asarray(d["C_obs"], dtype=np.float32)        # [H,W,T]
            A = np.asarray(d["A"], dtype=np.float32)              # [T,T]
            aif = np.asarray(d["aif"], dtype=np.float32)          # [T]

            cbf_gt = np.asarray(d["cbf_gt"], dtype=np.float32)    # [H,W]
            cbv_gt = np.asarray(d["cbv_gt"], dtype=np.float32)    # [H,W]
            mtt_gt = np.asarray(d["mtt_gt"], dtype=np.float32)    # [H,W]

            if "brain_mask" in d.files:
                mask = np.asarray(d["brain_mask"], dtype=np.float32)
            else:
                mask = np.ones_like(cbf_gt, dtype=np.float32)

            if "h_scale" in d.files:
                h_scale = float(np.asarray(d["h_scale"]).reshape(()))
            else:
                h_scale = 1.0

            t = np.asarray(d["t"], dtype=np.float32)

        T, H, W = h.shape
        h = np.transpose(h, (1, 2, 0)).reshape(H * W, T)          # [V,T]
        C = C.reshape(H * W, T)                                   # [V,T]

        return (
            torch.from_numpy(h),
            torch.from_numpy(C),
            torch.from_numpy(A),
            torch.from_numpy(aif),
            torch.from_numpy(cbf_gt),
            torch.from_numpy(cbv_gt),
            torch.from_numpy(mtt_gt),
            torch.from_numpy(mask),
            torch.tensor(h_scale, dtype=torch.float32),
            torch.from_numpy(t),
        )


# -----------------------------------------------------------------------------
# Forward-model helpers
# -----------------------------------------------------------------------------


def parse_cnr(value):
    if value is None or str(value).lower() in {"none", "null", "clean"}:
        return None
    return float(value)


def add_cnr_noise(C: Tensor, cnr: float | None, generator: torch.Generator) -> Tensor:
    """Voxel-wise CNR noise used during training.

    C: [B,V,T]
    """
    if cnr is None:
        return C

    delta = C.amax(dim=-1) - C.amin(dim=-1)                       # [B,V]
    sigma = delta / float(cnr)
    noise = torch.randn(
        C.shape,
        generator=generator,
        device=C.device,
        dtype=C.dtype,
    )
    return C + sigma.unsqueeze(-1) * noise


def add_cnr_noise_deterministic(
    C: Tensor,
    cnr: float | None,
    *,
    noise_seed: int,
    global_case: int,
    device: torch.device,
) -> Tensor:
    """Deterministic validation noise using the same per-case NumPy rule."""
    if cnr is None:
        return C

    noisy_cases = []
    for j in range(C.shape[0]):
        c_np = C[j].detach().cpu().numpy()
        rng = np.random.default_rng(int(noise_seed) + global_case + j)

        flat = c_np.reshape(-1, c_np.shape[-1])
        delta = flat.max(axis=1) - flat.min(axis=1)
        sigma = delta / float(cnr)

        cn = (
            flat + rng.normal(size=flat.shape) * sigma[:, None]
        ).reshape(c_np.shape).astype(np.float32)

        noisy_cases.append(torch.from_numpy(cn))

    return torch.stack(noisy_cases, dim=0).to(
        device=device,
        dtype=torch.float32,
    )


def normalize_forward_model(C: Tensor, A: Tensor, aif: Tensor):
    """Scale AIF, A, and C by the same per-case scalar; h is unchanged."""
    scale = aif.abs().amax(dim=-1).clamp_min(1e-8)                 # [B]
    aif_n = aif / scale[:, None]
    A_n = A / scale[:, None, None]
    C_n = C / scale[:, None, None]
    return C_n, A_n, aif_n


# -----------------------------------------------------------------------------
# Synthetic-V2 evaluation metrics
# -----------------------------------------------------------------------------


def perfusion_maps_from_h_torch(
    h_network: Tensor,
    *,
    h_scale: Tensor,
    dt: Tensor,
):
    """Convert network-scale h to Synthetic-V2 CBF/CBV/MTT maps.

    This mirrors synthetic_utils.perfusion_maps_from_h:
      h_phys = h_network * h_scale
      CBF = 60 * max_t(max(h_phys, 0))
      CBV = trapezoidal integral of max(h_phys, 0)
      MTT = 60 * CBV / CBF

    Parameters
    ----------
    h_network : [B,V,T]
    h_scale   : [B]
    dt        : [B]
    """
    h_phys = h_network * h_scale[:, None, None]
    h_maps = torch.clamp(h_phys, min=0.0)

    cbf = 60.0 * torch.amax(h_maps, dim=-1)                       # [B,V]

    if h_maps.shape[-1] > 1:
        cbv = (
            0.5
            * (h_maps[..., :-1] + h_maps[..., 1:]).sum(dim=-1)
            * dt[:, None]
        )
    else:
        cbv = torch.zeros_like(cbf)

    mtt = 60.0 * cbv / cbf.clamp_min(1e-12)
    return cbf, cbv, mtt


def absrel_per_case(
    pred: Tensor,
    gt: Tensor,
    mask: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Mean absolute relative error for each case.

    pred, gt, mask: [B,H,W]
    returns: [B]
    """
    valid = mask > 0
    err = torch.abs(pred - gt) / torch.abs(gt).clamp_min(float(eps))
    err = torch.where(valid, err, torch.zeros_like(err))

    count = valid.flatten(1).sum(dim=1).clamp_min(1)
    return err.flatten(1).sum(dim=1) / count


def h_nrmse_per_case(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Masked h-NRMSE matching the project definition.

    pred,target: [B,V,T]
    mask:        [B,H,W]
    """
    B, V, _ = pred.shape
    mask_v = (mask.reshape(B, V) > 0).to(pred.dtype)[..., None]

    num = ((pred - target) ** 2 * mask_v).sum(dim=(1, 2))
    den = (target**2 * mask_v).sum(dim=(1, 2)).clamp_min(1e-24)
    return torch.sqrt(num / den)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    *,
    cnr,
    noise_seed,
    eps_relative=1e-8,
):
    """Validation evaluation using the project perfusion-map objective."""
    model.eval()
    rows = []
    global_case = 0

    for batch in loader:
        (
            h,
            C,
            A,
            aif,
            cbf_gt,
            cbv_gt,
            mtt_gt,
            mask,
            h_scale,
            t,
        ) = batch

        h = h.to(device=device, dtype=torch.float32)
        C = C.to(device=device, dtype=torch.float32)
        A = A.to(device=device, dtype=torch.float32)
        aif = aif.to(device=device, dtype=torch.float32)
        cbf_gt = cbf_gt.to(device=device, dtype=torch.float32)
        cbv_gt = cbv_gt.to(device=device, dtype=torch.float32)
        mtt_gt = mtt_gt.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.float32)
        h_scale = h_scale.to(device=device, dtype=torch.float32)
        t = t.to(device=device, dtype=torch.float32)

        Cn = add_cnr_noise_deterministic(
            C,
            cnr,
            noise_seed=int(noise_seed),
            global_case=global_case,
            device=device,
        )

        Cn, An, aifn = normalize_forward_model(Cn, A, aif)

        B, V, T = h.shape
        H, W = cbf_gt.shape[-2:]

        aif_vox = aifn[:, None, :].expand(B, V, T).reshape(B * V, T)
        C_vox = Cn.reshape(B * V, T)

        pred = model(aif_vox, C_vox).reshape(B, V, T)

        dt = (
            t[:, 1] - t[:, 0]
            if t.shape[-1] > 1
            else torch.ones(B, device=device, dtype=torch.float32)
        )

        cbf_pred, cbv_pred, mtt_pred = perfusion_maps_from_h_torch(
            pred,
            h_scale=h_scale,
            dt=dt,
        )

        cbf_pred = cbf_pred.reshape(B, H, W)
        cbv_pred = cbv_pred.reshape(B, H, W)
        mtt_pred = mtt_pred.reshape(B, H, W)

        e_cbf = absrel_per_case(
            cbf_pred, cbf_gt, mask, eps=float(eps_relative)
        )
        e_cbv = absrel_per_case(
            cbv_pred, cbv_gt, mask, eps=float(eps_relative)
        )
        e_mtt = absrel_per_case(
            mtt_pred, mtt_gt, mask, eps=float(eps_relative)
        )

        h_nrmse = h_nrmse_per_case(pred, h, mask)
        objective = (e_cbf + e_cbv + e_mtt) / 3.0

        # Optional forward residual, retained only as an audit quantity.
        Cpred = torch.bmm(pred, An.transpose(1, 2))
        data_residual = (
            (Cpred - Cn).flatten(1).norm(dim=1)
            / Cn.flatten(1).norm(dim=1).clamp_min(1e-12)
        )

        for j in range(B):
            rows.append(
                {
                    "case": global_case + j,
                    "cbf_absrel": float(e_cbf[j]),
                    "cbv_absrel": float(e_cbv[j]),
                    "mtt_absrel": float(e_mtt[j]),
                    "cbf_absrel_percent": 100.0 * float(e_cbf[j]),
                    "cbv_absrel_percent": 100.0 * float(e_cbv[j]),
                    "mtt_absrel_percent": 100.0 * float(e_mtt[j]),
                    "objective_mean_absrel": float(objective[j]),
                    "h_nrmse": float(h_nrmse[j]),
                    "data_residual": float(data_residual[j]),
                }
            )

        global_case += B

    return rows


def summarize_validation(rows):
    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    def sample_sd(key):
        x = np.asarray([r[key] for r in rows], dtype=np.float64)
        return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    return {
        "n_cases": len(rows),
        "objective_mean_absrel": mean("objective_mean_absrel"),
        "cbf_absrel_percent_mean": mean("cbf_absrel_percent"),
        "cbf_absrel_percent_sd": sample_sd("cbf_absrel_percent"),
        "cbv_absrel_percent_mean": mean("cbv_absrel_percent"),
        "cbv_absrel_percent_sd": sample_sd("cbv_absrel_percent"),
        "mtt_absrel_percent_mean": mean("mtt_absrel_percent"),
        "mtt_absrel_percent_sd": sample_sd("mtt_absrel_percent"),
        "h_nrmse_mean": mean("h_nrmse"),
        "h_nrmse_sd": sample_sd("h_nrmse"),
        "data_residual_mean": mean("data_residual"),
    }


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, device, cfg, epoch):
    model.train()
    total = 0.0
    count = 0

    rng = random.Random(int(cfg["seed"]) + epoch)
    gen = torch.Generator(device=device).manual_seed(
        int(cfg["seed"]) + 10000 + epoch
    )
    cnrs = [parse_cnr(v) for v in cfg["train_cnr_values"]]

    for batch in loader:
        # Only h, C, A, and AIF are needed for optimization. The remaining
        # quantities are carried by the dataset for validation.
        h, C, A, aif = batch[:4]

        h = h.to(device=device, dtype=torch.float32)
        C = C.to(device=device, dtype=torch.float32)
        A = A.to(device=device, dtype=torch.float32)
        aif = aif.to(device=device, dtype=torch.float32)

        cnr = rng.choice(cnrs)
        Cn = add_cnr_noise(C, cnr, gen)
        Cn, An, aifn = normalize_forward_model(Cn, A, aif)

        B, V, T = h.shape
        k = min(int(cfg["voxels_per_case"]), V)

        indices = []
        for b in range(B):
            idx = torch.randperm(V, generator=gen, device=device)[:k] + b * V
            indices.append(idx)
        indices = torch.cat(indices)

        h_all = h.reshape(B * V, T)
        C_all = Cn.reshape(B * V, T)
        aif_all = aifn[:, None, :].expand(B, V, T).reshape(B * V, T)
        A_all = An[:, None, :, :].expand(B, V, T, T).reshape(B * V, T, T)

        h_t = h_all[indices]
        C_t = C_all[indices]
        aif_t = aif_all[indices]
        A_t = A_all[indices]

        pred = model(aif_t, C_t)
        C_pred = reconvolve_rowwise(pred, A_t)

        loss_h = torch.mean((pred - h_t) ** 2)
        loss_c = torch.mean((C_pred - C_t) ** 2)
        loss = (
            float(cfg["trf_loss_weight"]) * loss_h
            + float(cfg["physics_loss_weight"]) * loss_c
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.get("grad_clip_norm") is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg["grad_clip_norm"]),
            )

        optimizer.step()

        total += float(loss.detach()) * len(indices)
        count += len(indices)

    return total / max(count, 1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    train_ds = CaseDataset(cfg["data_root"], "train")
    val_ds = CaseDataset(cfg["data_root"], "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["case_batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["eval_case_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        drop_last=False,
    )

    model = SPINNEDStyleMLP(
        timepoints=int(cfg["timepoints"]),
        hidden_dims=cfg["hidden_dims"],
        activation=cfg["activation"],
        dropout=float(cfg["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["epochs"]),
        eta_min=float(cfg["min_lr"]),
    )

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    best = math.inf
    best_epoch = None
    history = []

    val_cnr = parse_cnr(cfg.get("validation_cnr", None))
    eps_relative = float(cfg.get("eps_relative", 1.0e-8))

    print("=" * 80)
    print("SPINNED-STYLE TRAINING: SYNTHETIC V2")
    print("=" * 80)
    print("train cases          :", len(train_ds))
    print("validation cases     :", len(val_ds))
    print("device               :", device)
    print("validation CNR       :", "clean" if val_cnr is None else val_cnr)
    print("checkpoint criterion : mean AbsRel(CBF, CBV, MTT)")
    print("test split evaluated : NO")
    print("=" * 80)

    for epoch in range(int(cfg["epochs"])):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg,
            epoch,
        )
        scheduler.step()

        if (epoch + 1) % int(cfg["validate_every"]) != 0:
            continue

        val_rows = evaluate(
            model,
            val_loader,
            device,
            cnr=val_cnr,
            noise_seed=int(cfg["noise_seed"]) + 200000,
            eps_relative=eps_relative,
        )

        s = summarize_validation(val_rows)

        record = {
            "epoch": epoch + 1,
            "train_loss": float(loss),
            **s,
        }
        history.append(record)

        print(
            f"epoch={epoch + 1:04d} "
            f"loss={loss:.5e} | "
            f"objective={s['objective_mean_absrel']:.8f} | "
            f"CBF={s['cbf_absrel_percent_mean']:.4f}% "
            f"CBV={s['cbv_absrel_percent_mean']:.4f}% "
            f"MTT={s['mtt_absrel_percent_mean']:.4f}% | "
            f"hNRMSE={s['h_nrmse_mean']:.6f}"
        )

        if s["objective_mean_absrel"] < best:
            best = float(s["objective_mean_absrel"])
            best_epoch = epoch + 1

            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": best_epoch,
                    "best_val_objective": best,
                    "validation_summary": s,
                },
                out / "best.pt",
            )

            (out / "best_validation.json").write_text(
                json.dumps(
                    {
                        "epoch": best_epoch,
                        **s,
                    },
                    indent=2,
                )
            )

            print(
                f"  -> saved new best.pt "
                f"(epoch={best_epoch}, objective={best:.8f})"
            )

        (out / "history.json").write_text(json.dumps(history, indent=2))

    if best_epoch is None:
        raise RuntimeError("No validation checkpoint was produced.")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print("best epoch           :", best_epoch)
    print("best val objective   :", f"{best:.8f}")
    print("checkpoint            :", out / "best.pt")
    print("validation summary    :", out / "best_validation.json")
    print("training history      :", out / "history.json")
    print("120-case test         : NOT TOUCHED")
    print(
        "Next: enable the checkpoint in config_synthetic_v2.yml and run "
        "05_benchmark_all_methods.py for the common held-out test evaluation."
    )


if __name__ == "__main__":
    main()
