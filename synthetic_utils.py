from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml


# synthetic_utils.py lives directly in ~/ISBI_2027.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from checkpoint_utils import load_frozen_ccd3d  # noqa: E402


# ============================================================================
# General utilities
# ============================================================================

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return obj


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_device(value: str) -> torch.device:
    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(value)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return dev


def trapz(y: np.ndarray, dx: float, axis: int = -1) -> np.ndarray:
    fn = getattr(np, "trapezoid", None)
    if fn is not None:
        return fn(y, dx=dx, axis=axis)
    return np.trapz(y, dx=dx, axis=axis)


def import_module_from_path(name: str, path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Module file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Synthetic V2 data loading
# ============================================================================

def load_case(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        out = {k: np.asarray(d[k]) for k in d.files}

    clean_thw = np.asarray(out["clean"], dtype=np.float32)
    if clean_thw.ndim != 3:
        raise ValueError(f"{path}: clean must have shape [T,H,W].")

    # Project convention everywhere else is [H,W,T].
    out["h_gt"] = np.transpose(clean_thw, (1, 2, 0)).astype(np.float32)

    out["A"] = np.asarray(out["A"], dtype=np.float32)
    out["C_clean"] = np.asarray(out["C_clean"], dtype=np.float32)
    # V3 dataset-specific measurement used by all reconstruction methods.
    out["C_obs"] = np.asarray(out["C_obs"], dtype=np.float32)
    out["aif"] = np.asarray(out["aif"], dtype=np.float32)
    out["t"] = np.asarray(out["t"], dtype=np.float32)

    out["mask"] = np.asarray(
        out.get(
            "brain_mask",
            np.ones(out["h_gt"].shape[:2], dtype=np.float32),
        ),
        dtype=np.float32,
    )

    out["h_scale"] = float(
        np.asarray(out.get("h_scale", 1.0)).reshape(())
    )

    return out


def list_split(data_root: str | Path, split: str) -> List[Path]:
    paths = sorted(
        (Path(data_root) / split).glob("sample_*.npz")
    )
    if not paths:
        raise RuntimeError(
            f"No samples found in {Path(data_root) / split}"
        )
    return paths


def batch_paths(paths: Sequence[Path], batch_size: int):
    batch_size = int(batch_size)
    for i in range(0, len(paths), batch_size):
        yield list(paths[i : i + batch_size])


def stack_cases(paths: Sequence[Path]) -> Dict[str, Any]:
    cases = [load_case(p) for p in paths]

    HWT = cases[0]["h_gt"].shape
    T = HWT[-1]

    for c in cases:
        if c["h_gt"].shape != HWT:
            raise ValueError(
                "All cases in one batch must have identical H,W,T."
            )
        if c["A"].shape != (T, T):
            raise ValueError(
                f"A must have shape [T,T]; got {c['A'].shape}."
            )

    return {
        "paths": [Path(p) for p in paths],
        "cases": cases,
        "h_gt": np.stack(
            [c["h_gt"] for c in cases],
            axis=0,
        ),
        "A": np.stack(
            [c["A"] for c in cases],
            axis=0,
        ),
        "C": np.stack(
            [c["C_obs"] for c in cases],
            axis=0,
        ),
        "mask": np.stack(
            [c["mask"] for c in cases],
            axis=0,
        ),
        "h_scale": np.asarray(
            [c["h_scale"] for c in cases],
            dtype=np.float32,
        ),
    }


# ============================================================================
# Common Synthetic V2 perfusion metrics
# ============================================================================

def perfusion_maps_from_h(
    h_network: np.ndarray,
    *,
    h_scale: float,
    dt: float,
    mask: Optional[np.ndarray] = None,
    nonnegative_for_maps: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert network-scale h(t) to the project's CBF/CBV/MTT maps.

        h_phys = h_network * h_scale
        CBF    = 60 * max_t h_phys(t)
        CBV    = integral h_phys(t) dt
        MTT    = 60 * CBV / CBF

    Negative values are clipped only for map extraction, matching the
    Synthetic V2 evaluation used for RED tuning/testing.
    """
    h = (
        np.asarray(h_network, dtype=np.float64)
        * float(h_scale)
    )

    if nonnegative_for_maps:
        h = np.maximum(h, 0.0)

    cbf = 60.0 * np.max(h, axis=-1)
    cbv = trapz(
        h,
        dx=float(dt),
        axis=-1,
    )
    mtt = (
        60.0
        * cbv
        / np.maximum(cbf, 1.0e-12)
    )

    if mask is not None:
        m = np.asarray(mask) > 0
        cbf = np.where(m, cbf, 0.0)
        cbv = np.where(m, cbv, 0.0)
        mtt = np.where(m, mtt, 0.0)

    return (
        cbf.astype(np.float32),
        cbv.astype(np.float32),
        mtt.astype(np.float32),
    )


def _valid(pred, gt, mask):
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)

    if mask is None:
        m = np.ones_like(g, dtype=bool)
    else:
        m = np.asarray(mask) > 0
        # Needed for h(t), where mask is [H,W,1] and data are [H,W,T].
        if m.shape != g.shape:
            try:
                m = np.broadcast_to(m, g.shape)
            except ValueError as exc:
                raise ValueError(
                    f"Mask shape {m.shape} cannot broadcast to data shape {g.shape}."
                ) from exc

    return p[m], g[m]


def absrel(pred, gt, mask=None, eps=1.0e-8) -> float:
    p, g = _valid(pred, gt, mask)
    return float(
        np.mean(
            np.abs(p - g)
            / np.maximum(np.abs(g), float(eps))
        )
    )


def nrmse(pred, gt, mask=None) -> float:
    p, g = _valid(pred, gt, mask)
    return float(
        np.linalg.norm(p - g)
        / (np.linalg.norm(g) + 1.0e-12)
    )


def mae(pred, gt, mask=None) -> float:
    p, g = _valid(pred, gt, mask)
    return float(np.mean(np.abs(p - g)))


def pearson(pred, gt, mask=None) -> float:
    p, g = _valid(pred, gt, mask)

    if (
        p.size < 2
        or np.std(p) < 1.0e-12
        or np.std(g) < 1.0e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(p, g)[0, 1]
    )


def evaluate_case(
    pred_h: np.ndarray,
    case: Mapping[str, Any],
    eps_relative: float = 1.0e-8,
) -> Dict[str, float]:
    """
    Common evaluator for EVERY method in Synthetic V2.

    Primary objective:
        mean(CBF AbsRel, CBV AbsRel, MTT AbsRel)

    h-NRMSE remains a secondary response-function metric.
    """
    pred_h = np.asarray(
        pred_h,
        dtype=np.float32,
    )

    if pred_h.shape != np.asarray(case["h_gt"]).shape:
        raise ValueError(
            f"pred_h has shape {pred_h.shape}, "
            f"expected {np.asarray(case['h_gt']).shape}."
        )

    t = np.asarray(case["t"])
    dt = (
        float(t[1] - t[0])
        if len(t) > 1
        else 1.0
    )

    cbf, cbv, mtt = perfusion_maps_from_h(
        pred_h,
        h_scale=float(case["h_scale"]),
        dt=dt,
        mask=case["mask"],
        nonnegative_for_maps=True,
    )

    out: Dict[str, float] = {}

    for name, pred, gt in (
        ("cbf", cbf, case["cbf_gt"]),
        ("cbv", cbv, case["cbv_gt"]),
        ("mtt", mtt, case["mtt_gt"]),
    ):
        out[f"{name}_absrel"] = absrel(
            pred,
            gt,
            case["mask"],
            eps=eps_relative,
        )

        out[f"{name}_absrel_percent"] = (
            100.0
            * out[f"{name}_absrel"]
        )

        out[f"{name}_nrmse"] = nrmse(
            pred,
            gt,
            case["mask"],
        )

        out[f"{name}_mae"] = mae(
            pred,
            gt,
            case["mask"],
        )

        out[f"{name}_pearson"] = pearson(
            pred,
            gt,
            case["mask"],
        )

    mask3 = np.asarray(
        case["mask"]
    )[..., None]

    out["h_nrmse"] = nrmse(
        pred_h,
        case["h_gt"],
        mask3,
    )

    out["tuning_objective"] = float(
        (
            out["cbf_absrel"]
            + out["cbv_absrel"]
            + out["mtt_absrel"]
        )
        / 3.0
    )

    # Case-level absolute-value summaries.
    m = np.asarray(case["mask"]) > 0

    out.update(
        {
            "gt_cbf_mean": float(
                np.mean(
                    np.asarray(case["cbf_gt"])[m]
                )
            ),
            "gt_cbv_mean": float(
                np.mean(
                    np.asarray(case["cbv_gt"])[m]
                )
            ),
            "gt_mtt_mean": float(
                np.mean(
                    np.asarray(case["mtt_gt"])[m]
                )
            ),
            "est_cbf_mean": float(
                np.mean(cbf[m])
            ),
            "est_cbv_mean": float(
                np.mean(cbv[m])
            ),
            "est_mtt_mean": float(
                np.mean(mtt[m])
            ),
        }
    )

    return out


def mean_sd(
    values: Sequence[float],
) -> Tuple[float, float]:
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return (
        float(np.mean(x)),
        (
            float(np.std(x, ddof=1))
            if x.size > 1
            else 0.0
        ),
    )


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    rows = list(rows)

    if not rows:
        raise ValueError(
            "Cannot write empty CSV."
        )

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    keys: List[str] = []
    seen = set()

    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=keys,
        )
        w.writeheader()

        for r in rows:
            w.writerow(r)


def save_json(
    path: str | Path,
    obj: Any,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            indent=2,
        )
    )


# ============================================================================
# CCD prior
# ============================================================================

def validate_ccd_checkpoint(
    ckpt: Mapping[str, Any],
    expected: Mapping[str, Any],
    model: torch.nn.Module,
) -> Dict[str, Any]:
    mc = ckpt.get("model_config")

    if not isinstance(mc, Mapping):
        raise KeyError(
            "CCD checkpoint has no model_config mapping."
        )

    errors: List[str] = []

    pmc = mc.get(
        "param_multi_conv",
        {},
    )

    if list(
        pmc.get(
            "num_channels",
            [],
        )
    ) != list(
        expected["num_channels"]
    ):
        errors.append(
            f"num_channels={pmc.get('num_channels')} "
            f"!= {expected['num_channels']}"
        )

    if list(
        pmc.get(
            "size_kernels",
            [],
        )
    ) != list(
        expected["kernel_sizes"]
    ):
        errors.append(
            f"kernel_sizes={pmc.get('size_kernels')} "
            f"!= {expected['kernel_sizes']}"
        )

    pa = mc.get(
        "param_spline_activation",
        {},
    )

    ea = expected["activation"]

    for k in (
        "num_knots",
        "x_min",
        "x_max",
        "init",
        "slope_min",
        "slope_max",
        "clamp",
    ):
        if pa.get(k) != ea.get(k):
            errors.append(
                f"activation.{k}="
                f"{pa.get(k)!r} "
                f"!= {ea.get(k)!r}"
            )

    ps = mc.get(
        "param_spline_scaling",
        {},
    )

    es = expected["scaling"]

    for k in (
        "num_knots",
        "x_min",
        "x_max",
        "init",
        "clamp",
    ):
        if ps.get(k) != es.get(k):
            errors.append(
                f"scaling.{k}="
                f"{ps.get(k)!r} "
                f"!= {es.get(k)!r}"
            )

    kw = mc.get(
        "kwargs",
        {},
    )

    expected_kw = {
        "blind":
            expected["blind"],
        "scaling":
            expected["scaling_enabled"],
        "step_size_type":
            expected["step_size_type"],
    }

    for k, v in expected_kw.items():
        if kw.get(k) != v:
            errors.append(
                f"kwargs.{k}="
                f"{kw.get(k)!r} "
                f"!= {v!r}"
            )

    if errors:
        raise RuntimeError(
            "CCD checkpoint architecture differs "
            "from config audit:\n  - "
            + "\n  - ".join(errors)
        )

    tau = float(
        model.effective_tau
        .detach()
        .cpu()
    )

    gamma_d = float(
        model.step_size()
        .detach()
        .cpu()
    )

    lo, hi = [
        float(v)
        for v
        in expected[
            "tau_admissible_range"
        ]
    ]

    if not lo <= tau <= hi:
        raise RuntimeError(
            f"Checkpoint tau={tau} "
            f"outside [{lo},{hi}]."
        )

    return {
        "tau_checkpoint": tau,
        "gamma_d": gamma_d,
    }


class CCDPrior:
    name = "ccd"

    def __init__(
        self,
        checkpoint: str | Path,
        device: torch.device,
        expected: Mapping[str, Any],
    ):
        self.device = device

        self.model, self.ckpt = (
            load_frozen_ccd3d(
                checkpoint,
                device,
            )
        )

        self.audit = (
            validate_ccd_checkpoint(
                self.ckpt,
                expected,
                self.model,
            )
        )

        self.train_sigma_lo = (
            self.ckpt.get(
                "training_sigma_min_255",
                0.0,
            )
        )

        self.train_sigma_hi = (
            self.ckpt.get(
                "training_sigma_max_255",
                25.0,
            )
        )

    @torch.inference_mode()
    def __call__(
        self,
        x_hwt: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        sigma_255 = float(strength)

        if not (
            float(self.train_sigma_lo)
            <= sigma_255
            <= float(self.train_sigma_hi)
        ):
            raise ValueError(
                f"CCD sigma={sigma_255}/255 "
                "outside checkpoint training range "
                f"[{self.train_sigma_lo},"
                f"{self.train_sigma_hi}]/255."
            )

        # [B,H,W,T] -> [B,1,T,H,W]
        x = (
            x_hwt
            .permute(0, 3, 1, 2)
            .unsqueeze(1)
            .contiguous()
        )

        sigma = torch.full(
            (x.shape[0],),
            sigma_255 / 255.0,
            dtype=x.dtype,
            device=x.device,
        )

        y = self.model(
            x,
            sigma,
        )

        return (
            y[:, 0]
            .permute(0, 2, 3, 1)
            .contiguous()
        )


# ============================================================================
# Batched 3-D TV prior
# ============================================================================

def _grad3(
    u: torch.Tensor,
) -> torch.Tensor:
    # u [B,T,H,W]
    # output [B,3,T,H,W]

    gt = torch.zeros_like(u)
    gy = torch.zeros_like(u)
    gx = torch.zeros_like(u)

    gt[:, :-1] = (
        u[:, 1:]
        - u[:, :-1]
    )

    gy[:, :, :-1] = (
        u[:, :, 1:]
        - u[:, :, :-1]
    )

    gx[:, :, :, :-1] = (
        u[:, :, :, 1:]
        - u[:, :, :, :-1]
    )

    return torch.stack(
        (gt, gy, gx),
        dim=1,
    )


def _grad3_adjoint(
    p: torch.Tensor,
) -> torch.Tensor:
    # Exact adjoint of _grad3.
    # p [B,3,T,H,W]

    pt, py, px = (
        p[:, 0],
        p[:, 1],
        p[:, 2],
    )

    out = torch.zeros_like(pt)

    # time
    out[:, 0] -= pt[:, 0]
    out[:, 1:-1] += (
        pt[:, :-2]
        - pt[:, 1:-1]
    )
    out[:, -1] += pt[:, -2]

    # y
    out[:, :, 0] -= py[:, :, 0]
    out[:, :, 1:-1] += (
        py[:, :, :-2]
        - py[:, :, 1:-1]
    )
    out[:, :, -1] += py[:, :, -2]

    # x
    out[:, :, :, 0] -= px[:, :, :, 0]
    out[:, :, :, 1:-1] += (
        px[:, :, :, :-2]
        - px[:, :, :, 1:-1]
    )
    out[:, :, :, -1] += px[:, :, :, -2]

    return out


class BatchedTVPrior:
    """
    Batched isotropic 3-D TV proximal denoiser.

    Each case is a [T,H,W] volume. The leading batch dimension
    is never differentiated across.
    """

    name = "tv"

    def __init__(
        self,
        inner_iterations: int = 40,
        primal_step: float = 0.25,
        dual_step: float = 0.25,
        theta: float = 1.0,
    ):
        self.inner_iterations = int(
            inner_iterations
        )

        self.primal_step = float(
            primal_step
        )

        self.dual_step = float(
            dual_step
        )

        self.theta = float(theta)

        if (
            self.primal_step
            * self.dual_step
            * 12.0
            >= 1.0
        ):
            raise ValueError(
                "TV primal_step*dual_step*12 "
                "must be < 1."
            )

    @torch.inference_mode()
    def __call__(
        self,
        x_hwt: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        weight = float(strength)

        if weight <= 0:
            return x_hwt

        # [B,H,W,T] -> [B,T,H,W]
        f = (
            x_hwt
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        u = f.clone()
        ubar = u.clone()

        p = torch.zeros(
            (
                f.shape[0],
                3,
                *f.shape[1:],
            ),
            dtype=f.dtype,
            device=f.device,
        )

        tau = self.primal_step
        sigma = self.dual_step

        for _ in range(
            self.inner_iterations
        ):
            p = (
                p
                + sigma
                * _grad3(ubar)
            )

            norm = (
                torch.linalg
                .vector_norm(
                    p,
                    dim=1,
                    keepdim=True,
                )
            )

            p = p / torch.maximum(
                torch.ones(
                    (),
                    device=p.device,
                    dtype=p.dtype,
                ),
                norm / weight,
            )

            u_old = u

            u = (
                u
                - tau
                * _grad3_adjoint(p)
                + tau * f
            ) / (1.0 + tau)

            ubar = (
                u
                + self.theta
                * (u - u_old)
            )

        return (
            u
            .permute(0, 2, 3, 1)
            .contiguous()
        )


# ============================================================================
# BM4D prior
# ============================================================================

class BM4DPrior:
    """
    BM4D wrapper.

    backend='cuda':
        CuPy implementation in third_party/bm4d_cuda.py.

    backend='reference':
        official CPU bm4d package.

    Every experiment case is denoised independently.
    """

    name = "bm4d"

    def __init__(
        self,
        backend: str,
        third_party_file: str | Path,
        device: torch.device,
        keep_on_gpu: bool = True,
        verbose: bool = False,
    ):
        self.backend = str(
            backend
        ).lower()

        self.device = device

        self.keep_on_gpu = bool(
            keep_on_gpu
        )

        self.verbose = bool(
            verbose
        )

        if self.backend == "cuda":
            if device.type != "cuda":
                raise RuntimeError(
                    "BM4D CUDA backend requires "
                    "a CUDA torch device."
                )

            third_party_file = Path(
                third_party_file
            )

            if not third_party_file.exists():
                raise FileNotFoundError(
                    "Fast BM4D file not found: "
                    f"{third_party_file}"
                )

            self.module = (
                import_module_from_path(
                    "bm4d_cuda_external",
                    third_party_file,
                )
            )

            try:
                import cupy as cp
            except ImportError as exc:
                raise ImportError(
                    "CuPy is required for "
                    "BM4D CUDA."
                ) from exc

            self.cp = cp

        elif self.backend == "reference":
            try:
                import bm4d as bm4d_pkg
            except ImportError as exc:
                raise ImportError(
                    "Install reference BM4D "
                    "with: pip install bm4d"
                ) from exc

            self.module = bm4d_pkg
            self.cp = None

        else:
            raise ValueError(
                "bm4d.backend must be "
                "'cuda' or 'reference'."
            )

    def _torch_to_cupy(
        self,
        x: torch.Tensor,
    ):
        cp = self.cp

        try:
            return cp.from_dlpack(x)
        except Exception:
            return cp.fromDlpack(
                torch.utils.dlpack
                .to_dlpack(x)
            )

    def _cupy_to_torch(
        self,
        x,
    ):
        return (
            torch.utils.dlpack
            .from_dlpack(x)
        )

    @torch.inference_mode()
    def __call__(
        self,
        x_hwt: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        sigma = float(strength)

        # BM4D volume convention here: [T,H,W].
        vol = (
            x_hwt
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        outputs: List[
            torch.Tensor
        ] = []

        if self.backend == "cuda":
            for b in range(
                vol.shape[0]
            ):
                cp_in = (
                    self._torch_to_cupy(
                        vol[b]
                    )
                )

                cp_out = (
                    self.module.bm4d(
                        cp_in,
                        sigma=sigma,
                        keep_on_gpu=True,
                        verbose=self.verbose,
                    )
                )

                out = (
                    self._cupy_to_torch(
                        cp_out
                    )
                )

                outputs.append(
                    out.to(
                        device=x_hwt.device,
                        dtype=x_hwt.dtype,
                    )
                )

            den = torch.stack(
                outputs,
                dim=0,
            )

        else:
            for b in range(
                vol.shape[0]
            ):
                inp = (
                    vol[b]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                out = self.module.bm4d(
                    inp,
                    sigma,
                )

                outputs.append(
                    torch.as_tensor(
                        np.asarray(
                            out,
                            dtype=np.float32,
                        ),
                        device=x_hwt.device,
                        dtype=x_hwt.dtype,
                    )
                )

            den = torch.stack(
                outputs,
                dim=0,
            )

        return (
            den
            .permute(0, 2, 3, 1)
            .contiguous()
        )


# ============================================================================
# RED reconstruction
# ============================================================================

@torch.inference_mode()
def red_reconstruct_batch(
    *,
    A_np: np.ndarray,
    C_np: np.ndarray,
    shape_hwt: Tuple[int, int, int],
    prior,
    strength: float,
    device: torch.device,
    lambda_red: float,
    alpha_numerator: float,
    max_iter: int,
    min_iter: int,
    tol: float,
    normalize_forward: bool = True,
) -> Tuple[
    np.ndarray,
    List[Dict[str, Any]],
]:
    """
    Batched RED-GD.

    If normalize_forward=True, each case uses

        s_b = max |A_b|
        A_b <- A_b / s_b
        C_b <- C_b / s_b

    which preserves the unknown h(t), but prevents the scale of the
    case-specific AIF from dominating the RED/data-term balance.

    Iteration:
        H_{k+1} = H_k - alpha_b [
            (H_k A_b^T - C_b) A_b
            + lambda_red (H_k - D(H_k))
        ]

    alpha_b =
        alpha_numerator /
        ( ||A_b||_2^2 + lambda_red ).
    """
    H, W, T = shape_hwt

    A_raw = torch.as_tensor(
        A_np,
        dtype=torch.float32,
        device=device,
    )

    C_raw = torch.as_tensor(
        C_np.reshape(
            len(C_np),
            H * W,
            T,
        ),
        dtype=torch.float32,
        device=device,
    )

    B = A_raw.shape[0]

    if normalize_forward:
        forward_scale = (
            torch.amax(
                torch.abs(A_raw),
                dim=(1, 2),
            )
            .clamp_min(1.0e-8)
        )

        A = (
            A_raw
            / forward_scale[
                :,
                None,
                None,
            ]
        )

        C = (
            C_raw
            / forward_scale[
                :,
                None,
                None,
            ]
        )

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

    L = (
        torch.linalg
        .eigvalsh(AtA)
        .amax(dim=1)
    )

    alpha = (
        float(alpha_numerator)
        / (
            L
            + float(lambda_red)
        )
    )

    x = torch.zeros(
        (
            B,
            H * W,
            T,
        ),
        dtype=torch.float32,
        device=device,
    )

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

        old_hwt = old.reshape(
            -1,
            H,
            W,
            T,
        )

        den_hwt = prior(
            old_hwt,
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

        ai = (
            alpha
            .index_select(
                0,
                idx,
            )[
                :,
                None,
                None,
            ]
        )

        new = (
            old
            - ai
            * (
                grad
                + float(lambda_red)
                * (old - den)
            )
        )

        num = (
            torch.linalg
            .vector_norm(
                (new - old)
                .flatten(1),
                dim=1,
            )
        )

        denom = torch.maximum(
            torch.linalg
            .vector_norm(
                new.flatten(1),
                dim=1,
            ),
            torch.linalg
            .vector_norm(
                old.flatten(1),
                dim=1,
            ),
        )

        rc = (
            num
            / (
                denom
                + 1.0e-12
            )
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
            done_local = (
                rc < float(tol)
            )

            if torch.any(
                done_local
            ):
                done_idx = (
                    idx[
                        done_local
                    ]
                )

                converged[
                    done_idx
                ] = True

                active[
                    done_idx
                ] = False

                iterations[
                    done_idx
                ] = it

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
    )

    infos: List[
        Dict[str, Any]
    ] = []

    for b in range(B):
        xb = x[b]

        # Relative residual is invariant to the shared A/C normalization.
        pred_c = (
            xb
            @ A[b].T
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
                        L[b]
                        .item()
                    ),
                "alpha":
                    float(
                        alpha[b]
                        .item()
                    ),
                "forward_scale":
                    float(
                        forward_scale[b]
                        .item()
                    ),
                "data_residual":
                    residual,
                "batch_runtime_seconds":
                    float(runtime),
                "effective_runtime_seconds":
                    float(
                        runtime / B
                    ),
            }
        )

    return pred, infos


def make_prior(
    name: str,
    *,
    cfg: Mapping[str, Any],
    device: torch.device,
):
    name = name.lower()

    if name == "ccd":
        return CCDPrior(
            resolve_repo_path(
                cfg["paths"][
                    "checkpoint"
                ]
            ),
            device,
            cfg["ccd_expected"],
        )

    if name == "tv":
        tv = cfg["tv"]

        return BatchedTVPrior(
            inner_iterations=int(
                tv[
                    "inner_iterations"
                ]
            ),
            primal_step=float(
                tv[
                    "primal_step"
                ]
            ),
            dual_step=float(
                tv[
                    "dual_step"
                ]
            ),
            theta=float(
                tv["theta"]
            ),
        )

    if name == "bm4d":
        bcfg = cfg["bm4d"]

        return BM4DPrior(
            backend=bcfg[
                "backend"
            ],
            third_party_file=(
                REPO_ROOT
                / bcfg[
                    "third_party_file"
                ]
            ),
            device=device,
            keep_on_gpu=bool(
                bcfg.get(
                    "keep_on_gpu",
                    True,
                )
            ),
            verbose=bool(
                bcfg.get(
                    "verbose",
                    False,
                )
            ),
        )

    raise ValueError(
        f"Unknown prior: {name}"
    )


# ============================================================================
# Classical / learned benchmark wrappers
# ============================================================================

def _load_csvd_threshold(
    cfg: Mapping[str, Any],
) -> float:
    """
    Read the cSVD threshold chosen ONLY on the 128-case Synthetic V2
    validation set using the common CBF/CBV/MTT objective.

    There is deliberately no fallback to the old H-PSNR-selected threshold.
    """
    method_cfg = (
        cfg.get(
            "methods",
            {},
        )
        .get(
            "csvd",
            {},
        )
    )

    threshold_file = (
        method_cfg.get(
            "threshold_file",
            (
                "outputs/synthetic_v2/"
                "csvd_tuning/"
                "best_threshold.json"
            ),
        )
    )

    path = resolve_repo_path(
        threshold_file
    )

    if not path.exists():
        raise FileNotFoundError(
            "Final Synthetic V2 cSVD threshold file "
            f"not found: {path}\n"
            "Run csvd/trainer_synthetic_v2.py on the "
            "128-case validation split before the final benchmark."
        )

    obj = json.loads(
        path.read_text()
    )

    if "best_threshold" not in obj:
        raise KeyError(
            f"{path} does not contain 'best_threshold'."
        )

    return float(
        obj["best_threshold"]
    )


def reconstruct_csvd(
    case,
    cfg,
):
    module = (
        import_module_from_path(
            "csvd_model_synv2",
            REPO_ROOT
            / "csvd"
            / "model.py",
        )
    )

    dt = (
        float(
            case["t"][1]
            - case["t"][0]
        )
        if len(case["t"]) > 1
        else 1.0
    )

    threshold = (
        _load_csvd_threshold(
            cfg
        )
    )

    method_cfg = (
        cfg.get(
            "methods",
            {},
        )
        .get(
            "csvd",
            {},
        )
    )

    return module.csvd_deconvolve(
        case["C_obs"],
        case["aif"],
        threshold=threshold,
        dt=dt,
        nonnegative=bool(
            method_cfg.get(
                "nonnegative",
                True,
            )
        ),
        padding_factor=int(
            method_cfg.get(
                "padding_factor",
                2,
            )
        ),
    )


def reconstruct_osvd(
    case,
    cfg,
):
    """Pizzolato-style block-circulant oSVD comparator with fixed 20% cutoff."""
    module = import_module_from_path(
        "osvd_model_synv2",
        REPO_ROOT / "osvd" / "model.py",
    )

    ocfg = load_yaml(
        REPO_ROOT / "osvd" / "config.yml"
    )

    dt = (
        float(case["t"][1] - case["t"][0])
        if len(case["t"]) > 1
        else 1.0
    )

    # Final literature comparator: fixed 20% cutoff.
    threshold = float(
        ocfg.get("threshold", 0.20)
    )

    pred, _ = module.osvd_deconvolve(
        case["C_obs"],
        case["aif"],
        dt=dt,
        default_threshold=threshold,
        nonnegative=bool(
            ocfg.get("nonnegative", False)
        ),
        padding_factor=int(
            ocfg.get("padding_factor", 2)
        ),
    )

    return np.asarray(
        pred,
        dtype=np.float32,
    )


def reconstruct_tisvd(
    case,
    cfg,
):
    module = (
        import_module_from_path(
            "tisvd_model_synv2",
            REPO_ROOT
            / "tisvd"
            / "model.py",
        )
    )

    tcfg = load_yaml(
        REPO_ROOT
        / "tisvd"
        / "config.yml"
    )

    lambdas = np.logspace(
        float(
            tcfg.get(
                "lambda_log10_min",
                -5,
            )
        ),
        float(
            tcfg.get(
                "lambda_log10_max",
                1,
            )
        ),
        int(
            tcfg.get(
                "lambda_count",
                61,
            )
        ),
    )

    scale = max(
        float(
            np.max(
                np.abs(
                    case["aif"]
                )
            )
        ),
        1.0e-8,
    )

    _, pred, _ = (
        module.lcurve_select(
            case["C_obs"]
            / scale,
            case["A"]
            / scale,
            lambdas,
            bool(
                tcfg.get(
                    "nonnegative",
                    False,
                )
            ),
        )
    )

    return pred


def reconstruct_meb(
    case,
    cfg,
):
    """
    Pizzolato et al. Modified Exponential Bases (MEB).

    The local meb/model.py is the project reimplementation. For the clean
    Synthetic V2 experiment the MEB config should use tau_min=tau_max=0,
    because no separate AIF/tissue delay is injected into these data.
    """
    module = (
        import_module_from_path(
            "meb_model_synv2",
            REPO_ROOT
            / "meb"
            / "model.py",
        )
    )

    mcfg = load_yaml(
        REPO_ROOT
        / "meb"
        / "config.yml"
    )

    t = np.asarray(
        case["t"]
    )

    dt = (
        float(t[1] - t[0])
        if len(t) > 1
        else 1.0
    )

    # Paper-style MTTmax rule:
    # MTTmax = factor * MTT_oSVD.
    osvd_pred = reconstruct_osvd(
        case,
        cfg,
    )

    _, _, mtt_osvd = (
        perfusion_maps_from_h(
            osvd_pred,
            h_scale=float(
                case["h_scale"]
            ),
            dt=dt,
            mask=case["mask"],
            nonnegative_for_maps=True,
        )
    )

    factor = float(
        mcfg.get(
            "mttmax_factor",
            4.0,
        )
    )

    mtt_max_map = np.maximum(
        factor
        * np.asarray(
            mtt_osvd,
            dtype=np.float64,
        ),
        dt,
    )

    pred, _, _ = (
        module.meb_deconvolve(
            case["C_obs"],
            case["aif"],
            dt=dt,
            mtt_max_map=
                mtt_max_map,
            n_basis=int(
                mcfg.get(
                    "basis_order",
                    30,
                )
            ),
            tau_min=float(
                mcfg.get(
                    "tau_min",
                    0.0,
                )
            ),
            tau_max=float(
                mcfg.get(
                    "tau_max",
                    0.0,
                )
            ),
            tau_step=float(
                mcfg.get(
                    "tau_step",
                    0.25,
                )
            ),
            padding_factor=int(
                mcfg.get(
                    "padding_factor",
                    2,
                )
            ),
            mask=case["mask"],
            nnls_maxiter=
                mcfg.get(
                    "nnls_maxiter",
                    None,
                ),
        )
    )

    return np.asarray(
        pred,
        dtype=np.float32,
    )


def reconstruct_pinn(
    case,
    cfg,
    device: torch.device,
    seed: int = 0,
):
    """Rotkopf-style voxel-wise PINN for the clean Synthetic V2 experiment.

    The final synthetic benchmark uses the raw A/C forward scale
    (normalize_forward: false in rotkopf_pinn/config.yml), because the fixed
    lambda_EN=lambda_TV=0.5 weights were selected for the paper's synthetic
    experiment rather than the in-vivo max-AIF-rescaled pipeline.

    No GT h/CBF/CBV/MTT enters optimization.
    """
    module = import_module_from_path(
        "pinn_model_synv2",
        REPO_ROOT
        / "rotkopf_pinn"
        / "model.py",
    )

    pcfg = load_yaml(
        REPO_ROOT
        / "rotkopf_pinn"
        / "config.yml"
    )

    C_np = np.asarray(
        case["C_obs"],
        dtype=np.float32,
    )

    if C_np.ndim != 3:
        raise ValueError(
            "C_clean must have shape [H,W,T]."
        )

    H, W, T = C_np.shape
    V = H * W

    normalize_forward = bool(
        pcfg.get(
            "normalize_forward",
            False,
        )
    )

    if normalize_forward:
        scale = max(
            float(
                np.max(
                    np.abs(case["aif"])
                )
            ),
            1.0e-8,
        )
    else:
        scale = 1.0

    A_t = torch.as_tensor(
        np.asarray(case["A"], np.float32)
        / scale,
        dtype=torch.float32,
        device=device,
    )

    C_t = torch.as_tensor(
        C_np.reshape(V, T)
        / scale,
        dtype=torch.float32,
        device=device,
    )

    t_t = torch.as_tensor(
        case["t"],
        dtype=torch.float32,
        device=device,
    )

    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            int(seed)
        )

    model = module.VoxelMLPBank(
        n_voxels=V,
        hidden_width=int(
            pcfg.get(
                "hidden_width",
                32,
            )
        ),
        hidden_layers=int(
            pcfg.get(
                "hidden_layers",
                4,
            )
        ),
        use_output_scale=bool(
            pcfg.get(
                "use_output_scale",
                True,
            )
        ),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            pcfg.get(
                "peak_lr",
                0.005,
            )
        ),
        weight_decay=float(
            pcfg.get(
                "weight_decay",
                0.0,
            )
        ),
    )

    override = cfg["methods"].get(
        "pinn_epochs_override",
        None,
    )

    epochs = (
        int(override)
        if override is not None
        else int(
            pcfg.get(
                "epochs",
                2000,
            )
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(
                pcfg.get(
                    "end_lr",
                    5.0e-5,
                )
            ),
        )
    )

    for _ in range(epochs):
        h = model(t_t)

        loss, *_ = module.rotkopf_loss(
            h,
            C_t,
            A_t,
            lambda_en=float(
                pcfg.get(
                    "lambda_en",
                    0.5,
                )
            ),
            lambda_tv=float(
                pcfg.get(
                    "lambda_tv",
                    0.5,
                )
            ),
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        pred = (
            model(t_t)
            .reshape(H, W, T)
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        pred,
        dtype=np.float32,
    )


def load_spinned_model(
    cfg,
    device: torch.device,
):
    """
    Load the validation-selected Synthetic V2 SPINNED-style checkpoint.

    The architecture saved inside best.pt is preferred, preventing later
    edits to spinned_style/config.yml from silently changing inference.
    """
    scfg = (
        cfg["methods"][
            "spinned"
        ]
    )

    if not bool(
        scfg.get(
            "enabled",
            False,
        )
    ):
        return None

    checkpoint = resolve_repo_path(
        scfg["checkpoint"]
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            "SPINNED V2 checkpoint not found: "
            f"{checkpoint}\n"
            "Train spinned_style/trainer.py with "
            "CBF/CBV/MTT validation checkpoint selection first."
        )

    ckpt = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    saved_cfg = ckpt.get(
        "config",
        {},
    )

    if not isinstance(
        saved_cfg,
        Mapping,
    ):
        saved_cfg = {}

    fallback_cfg = load_yaml(
        REPO_ROOT
        / "spinned_style"
        / "config.yml"
    )

    timepoints = int(
        saved_cfg.get(
            "timepoints",
            fallback_cfg[
                "timepoints"
            ],
        )
    )

    hidden_dims = saved_cfg.get(
        "hidden_dims",
        fallback_cfg[
            "hidden_dims"
        ],
    )

    activation = saved_cfg.get(
        "activation",
        fallback_cfg[
            "activation"
        ],
    )

    dropout = float(
        saved_cfg.get(
            "dropout",
            fallback_cfg.get(
                "dropout",
                0.0,
            ),
        )
    )

    model_module = (
        import_module_from_path(
            "spinned_model_synv2",
            REPO_ROOT
            / "spinned_style"
            / "model.py",
        )
    )

    model = (
        model_module
        .SPINNEDStyleMLP(
            timepoints=
                timepoints,
            hidden_dims=
                hidden_dims,
            activation=
                activation,
            dropout=
                dropout,
        )
        .to(device)
    )

    if "model" not in ckpt:
        raise KeyError(
            f"{checkpoint} has no 'model' state_dict."
        )

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    return model


@torch.inference_mode()
def reconstruct_spinned(
    case,
    model,
    device: torch.device,
):
    """
    Direct-fit SPINNED-style inference:
        [AIF(t), C_t(t)] -> h(t)

    AIF and C are divided by the same case-specific scale, so h remains
    unchanged.
    """
    if model is None:
        raise RuntimeError(
            "SPINNED model is None."
        )

    C_np = np.asarray(
        case["C_obs"],
        dtype=np.float32,
    )

    if C_np.ndim != 3:
        raise ValueError(
            "C_clean must have shape [H,W,T]."
        )

    H, W, T = C_np.shape
    V = H * W

    scale = max(
        float(
            np.max(
                np.abs(
                    case["aif"]
                )
            )
        ),
        1.0e-8,
    )

    C_t = torch.as_tensor(
        C_np.reshape(
            V,
            T,
        )
        / scale,
        dtype=torch.float32,
        device=device,
    )

    a_t = (
        torch.as_tensor(
            case["aif"]
            / scale,
            dtype=torch.float32,
            device=device,
        )[
            None
        ]
        .expand(
            V,
            T,
        )
    )

    pred = model(
        a_t,
        C_t,
    )

    return (
        pred.reshape(
            H,
            W,
            T,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )
