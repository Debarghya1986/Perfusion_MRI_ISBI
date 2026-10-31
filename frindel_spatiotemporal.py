from __future__ import annotations

"""
Frindel-style handcrafted spatio-temporal deconvolution benchmark.

Reference:
    C. Frindel, M. C. Robini, D. Rousseau,
    "A 3-D spatio-temporal deconvolution approach for MR perfusion in the brain,"
    Medical Image Analysis 18(1), 144-160, 2014.

V3 adaptation:
    The paper uses a 3-D spatial voxel lattice plus time and a 26-neighbour
    spatial system. Synthetic V3 is a single 2-D spatial slice plus time, so
    this implementation uses the direct 2-D analogue: an 8-neighbour system.
    No tissue labels are used by the reconstruction.

Implemented objective (convex w1 potential from the paper):
    ||A h - C||_2^2
    + lambda_t ||D_t h||_2^2
    + lambda_s sum_edges,sum_t phi((h_i(t)-h_j(t))/d_ij)

with
    phi(u) = sqrt(u^2 + delta^2) - delta.

Optimization follows the paper's multiplicative half-quadratic / IRLS fixed
point. Each quadratic subproblem is solved matrix-free with conjugate
gradients instead of explicitly forming the huge sparse matrix.
"""

from dataclasses import dataclass, asdict
from math import sqrt
from typing import Dict

import numpy as np
import torch


@dataclass
class FrindelInfo:
    hq_iterations: int
    final_relative_change: float
    final_cg_iterations: int
    final_cg_relative_residual: float
    objective: float


def _time_laplacian(x: torch.Tensor, dt: float) -> torch.Tensor:
    if x.shape[-1] < 2:
        return torch.zeros_like(x)

    inv_dt2 = 1.0 / (float(dt) ** 2)
    out = torch.zeros_like(x)
    out[..., 0] = (x[..., 0] - x[..., 1]) * inv_dt2
    out[..., -1] = (x[..., -1] - x[..., -2]) * inv_dt2

    if x.shape[-1] > 2:
        out[..., 1:-1] = (
            2.0 * x[..., 1:-1] - x[..., :-2] - x[..., 2:]
        ) * inv_dt2

    return out


def _edge_specs():
    # One representative of every undirected edge in an 8-neighbour 2-D grid.
    return (
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 1, sqrt(2.0)),
        (1, -1, sqrt(2.0)),
    )


def _edge_slices(H: int, W: int, dy: int, dx: int):
    if dy >= 0:
        ys = slice(0, H - dy)
        yd = slice(dy, H)
    else:
        ys = slice(-dy, H)
        yd = slice(0, H + dy)

    if dx >= 0:
        xs = slice(0, W - dx)
        xd = slice(dx, W)
    else:
        xs = slice(-dx, W)
        xd = slice(0, W + dx)

    return ys, xs, yd, xd


def _spatial_weights(x: torch.Tensor, mask: torch.Tensor, delta: float):
    H, W, _ = x.shape
    weights = []

    for dy, dx, dist in _edge_specs():
        ys, xs, yd, xd = _edge_slices(H, W, dy, dx)
        valid = (mask[ys, xs] & mask[yd, xd]).unsqueeze(-1)

        u = (x[ys, xs, :] - x[yd, xd, :]) / float(dist)
        w = torch.rsqrt(u.square() + float(delta) ** 2)
        w = torch.where(valid, w, torch.zeros_like(w))

        weights.append((ys, xs, yd, xd, float(dist), w))

    return weights


def _spatial_quadratic_operator(z: torch.Tensor, weights) -> torch.Tensor:
    out = torch.zeros_like(z)

    for ys, xs, yd, xd, dist, w in weights:
        q = w * (z[ys, xs, :] - z[yd, xd, :]) / (dist * dist)
        out[ys, xs, :] += q
        out[yd, xd, :] -= q

    return out


def _cg(matvec, b, x0, max_iter: int, tol: float):
    x = torch.zeros_like(b) if x0 is None else x0.clone()

    r = b - matvec(x)
    p = r.clone()
    rr = torch.sum(r * r)
    bnorm = torch.linalg.vector_norm(b).clamp_min(1.0e-12)

    rel = torch.sqrt(rr) / bnorm
    if float(rel.detach().cpu()) <= float(tol):
        return x, 0, float(rel.detach().cpu())

    final_rel = float(rel.detach().cpu())
    n_done = 0

    for k in range(int(max_iter)):
        Ap = matvec(p)
        denom = torch.sum(p * Ap).clamp_min(1.0e-20)
        alpha = rr / denom

        x = x + alpha * p
        r = r - alpha * Ap

        rr_new = torch.sum(r * r)
        rel = torch.sqrt(rr_new) / bnorm

        final_rel = float(rel.detach().cpu())
        n_done = k + 1

        if final_rel <= float(tol):
            break

        beta = rr_new / rr.clamp_min(1.0e-20)
        p = r + beta * p
        rr = rr_new

    return x, n_done, final_rel


def _objective(h, A, C, mask, dt, lambda_t, lambda_s, delta):
    mask3 = mask.unsqueeze(-1)
    pred = torch.matmul(h, A.transpose(0, 1))
    data = torch.sum(((pred - C) * mask3) ** 2)

    d = (h[..., 1:] - h[..., :-1]) / float(dt)
    temporal = float(lambda_t) * torch.sum(
        (d * mask3.expand_as(h)[..., 1:]) ** 2
    )

    spatial = h.new_tensor(0.0)
    H, W, _ = h.shape

    for dy, dx, dist in _edge_specs():
        ys, xs, yd, xd = _edge_slices(H, W, dy, dx)
        valid = (mask[ys, xs] & mask[yd, xd]).unsqueeze(-1)

        u = (h[ys, xs, :] - h[yd, xd, :]) / float(dist)
        phi = torch.sqrt(u.square() + float(delta) ** 2) - float(delta)

        spatial = spatial + float(lambda_s) * torch.sum(
            torch.where(valid, phi, torch.zeros_like(phi))
        )

    return data + temporal + spatial


@torch.inference_mode()
def reconstruct_frindel(
    case: Dict,
    cfg: Dict,
    device: torch.device,
    *,
    lambda_t: float | None = None,
    lambda_s: float | None = None,
    delta: float | None = None,
):
    fcfg = cfg["frindel"]

    lambda_t = float(fcfg["lambda_t"] if lambda_t is None else lambda_t)
    lambda_s = float(fcfg["lambda_s"] if lambda_s is None else lambda_s)
    delta = float(fcfg.get("delta", 0.02) if delta is None else delta)

    if lambda_t <= 0:
        raise ValueError("lambda_t must be > 0.")
    if lambda_s < 0:
        raise ValueError("lambda_s must be >= 0.")
    if delta <= 0:
        raise ValueError("delta must be > 0.")

    A = torch.as_tensor(
        np.asarray(case["A"], np.float32),
        device=device,
        dtype=torch.float32,
    )
    C = torch.as_tensor(
        np.asarray(case["C_obs"], np.float32),
        device=device,
        dtype=torch.float32,
    )
    mask = torch.as_tensor(
        np.asarray(case["mask"]) > 0,
        device=device,
        dtype=torch.bool,
    )

    H, W, T = C.shape
    if A.shape != (T, T):
        raise ValueError(f"A has shape {tuple(A.shape)}, expected {(T, T)}.")

    t = np.asarray(case["t"], np.float64)
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    mask3 = mask.unsqueeze(-1)

    if bool(fcfg.get("normalize_forward", False)):
        scale = torch.max(torch.abs(A)).clamp_min(1.0e-8)
        A = A / scale
        C = C / scale

    At = A.transpose(0, 1)
    AtA = At @ A

    # Generator convention is C = h @ A.T, hence RHS = 2 C A.
    rhs = 2.0 * torch.matmul(C, A)
    rhs = torch.where(mask3, rhs, torch.zeros_like(rhs))

    ridge = float(fcfg.get("numerical_ridge", 1.0e-7))

    def base_op(z):
        z = torch.where(mask3, z, torch.zeros_like(z))
        data = 2.0 * torch.matmul(z, AtA)
        temporal = 2.0 * lambda_t * _time_laplacian(z, dt)
        y = data + temporal + ridge * z

        # Outside-mask identity forces the solution there to zero.
        return torch.where(mask3, y, z)

    # Temporal-only initial solution.
    h, cg_n, cg_rel = _cg(
        base_op,
        rhs,
        x0=None,
        max_iter=int(
            fcfg.get("initial_cg_max_iter", fcfg.get("cg_max_iter", 80))
        ),
        tol=float(fcfg.get("cg_tol", 1.0e-5)),
    )
    h = torch.where(mask3, h, torch.zeros_like(h))

    hq_max = int(fcfg.get("hq_max_iter", 20))
    hq_tol = float(fcfg.get("hq_tol", 1.0e-4))

    final_rel = float("inf")
    final_cg_n = cg_n
    final_cg_rel = cg_rel
    outer_done = 0

    for outer in range(hq_max):
        weights = _spatial_weights(h, mask, delta)

        def full_op(z):
            z = torch.where(mask3, z, torch.zeros_like(z))

            data = 2.0 * torch.matmul(z, AtA)
            temporal = 2.0 * lambda_t * _time_laplacian(z, dt)
            spatial = lambda_s * _spatial_quadratic_operator(z, weights)

            y = data + temporal + spatial + ridge * z
            return torch.where(mask3, y, z)

        new_h, final_cg_n, final_cg_rel = _cg(
            full_op,
            rhs,
            x0=h,
            max_iter=int(fcfg.get("cg_max_iter", 80)),
            tol=float(fcfg.get("cg_tol", 1.0e-5)),
        )
        new_h = torch.where(mask3, new_h, torch.zeros_like(new_h))

        denom = torch.linalg.vector_norm(h[mask]).clamp_min(1.0e-12)
        final_rel = float(
            (
                torch.linalg.vector_norm((new_h - h)[mask]) / denom
            ).detach().cpu()
        )

        h = new_h
        outer_done = outer + 1

        if final_rel <= hq_tol:
            break

    obj = float(
        _objective(
            h,
            A,
            C,
            mask,
            dt,
            lambda_t,
            lambda_s,
            delta,
        ).detach().cpu()
    )

    info = FrindelInfo(
        hq_iterations=outer_done,
        final_relative_change=final_rel,
        final_cg_iterations=int(final_cg_n),
        final_cg_relative_residual=float(final_cg_rel),
        objective=obj,
    )

    return h.detach().cpu().numpy().astype(np.float32), asdict(info)
