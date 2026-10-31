"""Pizzolato-style block-circulant oSVD baseline for DSC-MRI.

For the ISBI 2027 Synthetic V2 experiment we use the comparator exactly in the
form reported by Pizzolato et al. (ISBI 2015): block-circulant truncated SVD
with a fixed 20% singular-value cutoff ("oSVD (20%)").

Important naming note
---------------------
Wu et al. originally used "oSVD" for a block-circulant method with an
oscillation-controlled threshold. Pizzolato et al. use "oSVD" for the standard
block-circulant tSVD comparator and explicitly report a 20% truncation in their
experiments. We use the latter because MEB is being benchmarked against that
paper.

The function keeps the old project call signature for drop-in compatibility.
The adaptive-OI arguments are accepted but intentionally ignored; the fixed
threshold is taken from ``default_threshold``.
"""

from __future__ import annotations

import numpy as np


def build_block_circulant_aif_matrix(
    aif: np.ndarray,
    dt: float = 1.0,
    padding_factor: int = 2,
) -> np.ndarray:
    """Construct the zero-padded block-circulant AIF convolution matrix."""
    aif = np.asarray(aif, dtype=np.float64).reshape(-1)
    T = aif.size
    L = int(padding_factor) * T

    if L < 2 * T:
        raise ValueError(
            "padding_factor must give a circular length >= 2*T."
        )

    a = np.zeros(L, dtype=np.float64)
    a[:T] = aif

    i = np.arange(L)[:, None]
    j = np.arange(L)[None, :]

    # B[i,j] = dt * a[(i-j) mod L] implements circular convolution.
    return float(dt) * a[(i - j) % L]


def osvd_deconvolve(
    concentration,
    aif,
    dt=1.0,
    target_oi=None,
    default_threshold=0.20,
    threshold_min=None,
    threshold_max=None,
    threshold_count=None,
    nonnegative=False,
    padding_factor=2,
):
    """Recover h(t) using fixed-threshold block-circulant truncated SVD.

    Parameters
    ----------
    concentration : array_like, shape (..., T)
        Tissue concentration curves.
    aif : array_like, shape (T,)
        Arterial input function.
    dt : float
        Sampling interval.
    default_threshold : float
        Relative singular-value cutoff. For the final benchmark this is 0.20.
    nonnegative : bool
        Defaults to False for the paper-faithful oSVD comparator. The common
        Synthetic V2 evaluator may still clip negative h values only when
        deriving CBF/CBV/MTT maps.
    padding_factor : int
        Circular length is padding_factor*T; use 2.

    Returns
    -------
    h : np.ndarray
        First T samples of the recovered response, same shape as concentration.
    threshold_used : np.ndarray
        One constant threshold value per voxel, retained for compatibility with
        the previous adaptive-oSVD project API.
    """
    del target_oi, threshold_min, threshold_max, threshold_count

    C = np.asarray(concentration, dtype=np.float64)
    if C.ndim < 1:
        raise ValueError("concentration must have at least one dimension.")

    T = C.shape[-1]
    L = int(padding_factor) * T

    if np.asarray(aif).size != T:
        raise ValueError(
            f"AIF length {np.asarray(aif).size} does not match T={T}."
        )

    threshold = float(default_threshold)
    if not (0.0 <= threshold < 1.0):
        raise ValueError("default_threshold must satisfy 0 <= threshold < 1.")

    B = build_block_circulant_aif_matrix(
        aif,
        dt=float(dt),
        padding_factor=int(padding_factor),
    )

    U, S, Vh = np.linalg.svd(B, full_matrices=False)

    cutoff = threshold * float(S[0])
    Sinv = np.zeros_like(S)
    keep = S > cutoff
    Sinv[keep] = 1.0 / S[keep]

    flat = C.reshape(-1, T)

    Cpad = np.zeros((flat.shape[0], L), dtype=np.float64)
    Cpad[:, :T] = flat

    # Row-vector form of V diag(S^-1) U^T C.
    hpad = ((Cpad @ U) * Sinv[None, :]) @ Vh
    h = hpad[:, :T]

    if bool(nonnegative):
        h = np.maximum(h, 0.0)

    threshold_used = np.full(
        flat.shape[0],
        threshold,
        dtype=np.float32,
    )

    return (
        h.reshape(C.shape).astype(np.float32),
        threshold_used,
    )


__all__ = [
    "build_block_circulant_aif_matrix",
    "osvd_deconvolve",
]
