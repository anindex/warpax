"""50-digit mpmath verification of the Hawking-Ellis classifier.

The float64 classifier at :mod:`warpax.energy_conditions.classification`
uses a two-tier imaginary-part tolerance (``tol * scale`` absolute OR
``imag_rtol * unclamped_scale`` relative). This reactive tolerance may
misclassify a genuine Type-IV spectrum whose ``|Im λ|`` sits below the
relative threshold (e.g. ``|λ| ~ 1``, ``|Im λ| ~ 2e-5`` flips to Type I
at ``imag_rtol = 3e-3``).

This module offers a post-hoc 50-digit recomputation path. It is **not**
wired into the JAX grid pipeline (mpmath is pure-Python and thousands of
times slower than ``jnp.linalg.eig``). Intended use: audit a small
subset of Type-IV grid points reported by the paper to confirm they are
genuinely Type IV at 50-digit precision, and flag any that flip.

Functions
---------
- :func:`eigenvalues_mpmath` - 4-eigenvalue tuple at arbitrary precision.
- :func:`classify_hawking_ellis_mpmath` - single-point 50-digit verdict
  mirroring the float64 branch logic (same tolerances, more accurate
  imaginary parts).
- :func:`verify_classification_at_points` - batched audit returning a
  flip-rate report against an existing float64 ``he_types`` array.

Rationale: see classify_hawking_ellis docstring for context.
"""
from __future__ import annotations

from typing import Any

import mpmath
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def eigenvalues_mpmath(
    T_mixed: np.ndarray,
    precision: int = 50,
) -> tuple[mpmath.mpc, mpmath.mpc, mpmath.mpc, mpmath.mpc]:
    """Return the four eigenvalues of ``T_mixed`` at ``precision`` digits.

    Parameters
    ----------
    T_mixed : np.ndarray, shape (4, 4)
        Mixed stress-energy tensor ``T^a_{\\;b}`` in float64. NaN entries
        are replaced with zero before eigen-decomposition (consistent with
        the float64 classifier's cuSolver-safety guard).
    precision : int
        Decimal digits for the mpmath context (default 50).

    Returns
    -------
    tuple of mpmath.mpc
        Four complex eigenvalues in mpmath's arbitrary-precision format.
        Order matches ``mpmath.eig``'s return (no re-sorting).
    """
    if T_mixed.shape != (4, 4):
        raise ValueError(f"Expected (4, 4) matrix, got {T_mixed.shape}")

    T_safe = np.where(np.isnan(T_mixed), 0.0, T_mixed)

    with mpmath.workdps(precision):
        M = mpmath.matrix(T_safe.tolist())
        evals, _ = mpmath.eig(M)
        return tuple(evals)  # type: ignore[return-value]


def classify_hawking_ellis_mpmath(
    T_mixed: np.ndarray,
    g_ab: np.ndarray,
    precision: int = 50,
    tol: float = 1e-10,
    imag_rtol: float = 0.0,
) -> dict[str, Any]:
    """Classify a single point at ``precision`` digits.

    Mirrors the branchless logic of :func:`classify_hawking_ellis` but
    evaluates eigenvalues (and, for real spectra, eigenvectors) at
    arbitrary precision before applying tolerances.

    The default ``imag_rtol = 0.0`` is deliberately tighter than the
    float64 classifier's ``3e-3``. The relative tolerance exists in the
    float64 path to mask split-degenerate-pair noise from
    ``jnp.linalg.eig`` at large ``|λ|``; that noise does not exist at
    50-digit precision, so any non-negligible imaginary part is
    physical. Pass ``imag_rtol=3e-3`` explicitly to reproduce the
    float64 verdict exactly (useful when checking tolerance sensitivity
    rather than physical classification).

    Parameters
    ----------
    T_mixed : np.ndarray, shape (4, 4)
        Mixed stress-energy tensor.
    g_ab : np.ndarray, shape (4, 4)
        Covariant metric used for the causal-character check on
        eigenvectors (only consulted when the spectrum is real).
    precision : int
        mpmath decimal precision (default 50).
    tol : float
        Absolute imaginary-part threshold: ``|Im λ| < tol * max(|Re λ|, 1)``
        marks the spectrum as real. Default matches the float64 classifier.
    imag_rtol : float
        Relative imaginary-part threshold (default 0.0 - absolute-only
        audit; see note above).

    Returns
    -------
    dict with keys
        - ``he_type`` : int in {1, 2, 3, 4}
        - ``all_real`` : bool (passes two-tier imaginary check)
        - ``near_vacuum`` : bool
        - ``n_timelike``, ``n_null`` : int (only meaningful for real spectra)
        - ``n_unique`` : int (distinct real eigenvalues under tol*scale)
        - ``max_imag_abs`` : float (50-digit)
        - ``max_real_abs`` : float
        - ``eigenvalues_real``, ``eigenvalues_imag`` : list[float] length 4
        - ``precision`` : int
        - ``cond_V`` : float (Bauer-Fike eigenvector-matrix
          condition number ``sigma_max(V) / sigma_min(V)`` via
          ``mpmath.svd``; possibly ``float('inf')`` for exact-defective
          inputs or SVD failure)
        - ``uncertain`` : bool (``cond_V > 10 ** (precision / 2)``
          per Demmel 1997 Thm 4.4; True flags precision loss > half-digit,
          NOT a wrong classification - this is a known limitation of float64)
    """
    if T_mixed.shape != (4, 4) or g_ab.shape != (4, 4):
        raise ValueError(
            f"Expected (4, 4) matrices, got T={T_mixed.shape}, g={g_ab.shape}"
        )

    T_safe = np.where(np.isnan(T_mixed), 0.0, T_mixed)

    with mpmath.workdps(precision):
        M = mpmath.matrix(T_safe.tolist())
        evals, evecs = mpmath.eig(M)

        evals_real = [float(mpmath.re(e)) for e in evals]
        evals_imag = [float(mpmath.im(e)) for e in evals]

        max_real_abs = max(abs(r) for r in evals_real)
        max_imag_abs = max(abs(i) for i in evals_imag)

        scale = max(max_real_abs, 1.0)
        unclamped_scale = max(max_real_abs, tol**0.5)

        all_abs = all(abs(i) < tol * scale for i in evals_imag)
        all_rel = all(abs(i) < imag_rtol * unclamped_scale for i in evals_imag)
        all_real = all_abs or all_rel

        near_vacuum = max_real_abs < tol

        # Degeneracy count on real parts (matches float64 logic).
        sorted_re = sorted(evals_real)
        gaps = [sorted_re[i + 1] - sorted_re[i] for i in range(3)]
        n_unique = 1 + sum(1 for gap in gaps if gap > tol * scale)

        # Causal character: only meaningful when spectrum is real. For
        # complex spectra we skip causal analysis and defer to the
        # all_real / near_vacuum verdict.
        if all_real and not near_vacuum:
            n_timelike, n_null = _causal_counts(evecs, g_ab, tol)
        else:
            n_timelike, n_null = 0, 0

        is_type_iv = (not all_real) and (not near_vacuum)
        is_type_i = (all_real and n_timelike >= 1 and n_null == 0) or near_vacuum
        is_type_iii = (
            all_real and not near_vacuum and n_null >= 1 and n_unique == 1
        )

        if is_type_iv:
            he_type = 4
        elif is_type_i:
            he_type = 1
        elif is_type_iii:
            he_type = 3
        else:
            he_type = 2

        # Bauer-Fike eigenvector-matrix sensitivity diagnostic.
        # Threshold: cond_V > 10**(precision/2) per Demmel 1997 Thm 4.4.
        # Mpmath-only path - float64 classifier does NOT carry this
        # diagnostic (Bauer-Fike analysis is meaningful only at >= 50-digit
        # precision). 
        cond_V, uncertain = _cond_V_mpmath(evecs, precision)

    return {
        "he_type": he_type,
        "all_real": all_real,
        "near_vacuum": near_vacuum,
        "n_timelike": int(n_timelike),
        "n_null": int(n_null),
        "n_unique": int(n_unique),
        "max_imag_abs": max_imag_abs,
        "max_real_abs": max_real_abs,
        "eigenvalues_real": evals_real,
        "eigenvalues_imag": evals_imag,
        "precision": precision,
        "cond_V": cond_V,        # Bauer-Fike condition number
        "uncertain": uncertain,  # cond_V > 10^(precision/2)
    }


def verify_classification_at_points(
    T_mixed_batch: np.ndarray,
    g_ab_batch: np.ndarray,
    float64_he_types: np.ndarray,
    precision: int = 50,
    tol: float = 1e-10,
    imag_rtol: float = 0.0,
) -> dict[str, Any]:
    """Recompute Hawking-Ellis types at 50 digits and report flip rate.

    A "flip" is a point where the 50-digit verdict disagrees with the
    float64 verdict supplied in ``float64_he_types``.

    Parameters
    ----------
    T_mixed_batch : np.ndarray, shape (N, 4, 4)
        Batch of float64 ``T^a_{\\;b}`` matrices.
    g_ab_batch : np.ndarray, shape (N, 4, 4)
        Batch of covariant metrics (one per point).
    float64_he_types : np.ndarray, shape (N,)
        Existing float64 classifier output integers (1..4).
    precision, tol, imag_rtol : see :func:`classify_hawking_ellis_mpmath`.

    Returns
    -------
    dict with keys
        - ``n_points`` : int
        - ``n_flips`` : int
        - ``flip_rate`` : float (``n_flips / n_points``, zero when empty)
        - ``flip_indices`` : np.ndarray of flipped row indices
        - ``mpmath_he_types`` : np.ndarray, shape (N,), verdict per point
        - ``max_imag_abs`` : np.ndarray, shape (N,)
        - ``precision`` : int
        - ``tol``, ``imag_rtol`` : float
        - ``cond_V_per_point`` : np.ndarray[float64], shape (N,) - Bauer-Fike eigenvector-matrix condition number per point (possibly
          ``inf``).
        - ``uncertain_mask`` : np.ndarray[bool], shape (N,) - per-point
          ``cond_V > 10 ** (precision / 2)`` flag.
    """
    n = int(T_mixed_batch.shape[0])
    if g_ab_batch.shape[0] != n or float64_he_types.shape[0] != n:
        raise ValueError(
            "Batch sizes mismatch: "
            f"T={T_mixed_batch.shape[0]}, "
            f"g={g_ab_batch.shape[0]}, "
            f"he_types={float64_he_types.shape[0]}"
        )

    mp_types = np.zeros((n,), dtype=np.int32)
    max_imag = np.zeros((n,), dtype=np.float64)
    # per-point Bauer-Fike diagnostic arrays
    cond_V_per_point = np.zeros((n,), dtype=np.float64)
    uncertain_mask = np.zeros((n,), dtype=np.bool_)

    for i in range(n):
        report = classify_hawking_ellis_mpmath(
            T_mixed_batch[i],
            g_ab_batch[i],
            precision=precision,
            tol=tol,
            imag_rtol=imag_rtol,
        )
        mp_types[i] = report["he_type"]
        max_imag[i] = report["max_imag_abs"]
        cond_V_per_point[i] = report["cond_V"]
        uncertain_mask[i] = report["uncertain"]

    flip_mask = mp_types != np.asarray(float64_he_types, dtype=np.int32)
    flip_indices = np.nonzero(flip_mask)[0]
    n_flips = int(flip_indices.size)
    flip_rate = (n_flips / n) if n > 0 else 0.0

    return {
        "n_points": n,
        "n_flips": n_flips,
        "flip_rate": flip_rate,
        "flip_indices": flip_indices,
        "mpmath_he_types": mp_types,
        "max_imag_abs": max_imag,
        "precision": precision,
        "tol": tol,
        "imag_rtol": imag_rtol,
        "cond_V_per_point": cond_V_per_point,  # 
        "uncertain_mask": uncertain_mask,      # 
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cond_V_mpmath(
    evecs: mpmath.matrix, precision: int
) -> tuple[float, bool]:
    """Bauer-Fike eigenvector-matrix condition number for the mpmath classifier.

    Computes ``cond(V) = sigma_max(V) / sigma_min(V)`` via ``mpmath.svd``
    and returns ``(cond_V_float, uncertain_bool)``. The threshold for
    ``uncertain=True`` is ``cond_V > 10 ** (precision / 2)`` per Demmel 1997
    Thm 4.4 Bauer-Fike (half-digit precision-loss rule).

    For exact-Jordan-defective inputs (``sigma_min = 0``) or any SVD failure,
    returns ``(float('inf'), True)`` - fail-safe uncertain.

    Notes
    -----
    ``uncertain=True`` means precision loss exceeds half the working dps;
    it does NOT mean the Hawking-Ellis verdict is wrong. cond(V) qualifies
    V-conditioning, not |lambda|. See

    Parameters
    ----------
    evecs : mpmath.matrix, shape (4, 4)
        Eigenvector columns as returned by ``mpmath.eig``.
    precision : int
        Working dps used by the classifier; determines the threshold.

    Returns
    -------
    cond_V : float
        Eigenvector-matrix condition number (possibly ``float('inf')``).
    uncertain : bool
        ``True`` iff ``cond_V > 10 ** (precision / 2)``.

    References
    ----------
    - Demmel J. 1997, Applied Numerical Linear Algebra, Theorem 4.4.
    -
    """
    try:
        _U, S, _Vt = mpmath.svd(evecs)
        S_list = [S[i] for i in range(len(S))]
        sigma_max = S_list[0]
        sigma_min = S_list[-1]
        # Treat numerically-zero sigma_min as exact-defective
        if sigma_min == 0 or abs(sigma_min) < mpmath.mpf(10) ** (-precision - 5):
            cond_V = float("inf")
        else:
            cond_V = float(sigma_max / sigma_min)
    except (ZeroDivisionError, Exception):  # noqa: BLE001 - fail-safe
        cond_V = float("inf")

    uncertain = cond_V > 10 ** (precision / 2)
    return cond_V, uncertain


def _causal_counts(
    evecs: mpmath.matrix, g_ab: np.ndarray, tol: float
) -> tuple[int, int]:
    """Count timelike / null eigenvectors via g_{ab} v^a v^b.

    Mirrors the float64 classifier's causal-character accounting. For
    the real-spectrum branch only; complex spectra short-circuit to
    Type IV before this helper is consulted.

    relative-sign threshold normalized against max |v^T g v|.
    Mirrors the float64 path's relative-threshold fix; restores
    scale-aware sign discrimination at metrics where
    |g_{ij}|/|g_{00}| >> 1 (e.g., WarpShell at v_s=0.5). Behaviour at
    Minkowski / unit-scale (max|quad| <= 1.0) is bit-preserved because
    the scale floors at 1.0. See:

    Parameters
    ----------
    evecs : mpmath.matrix, shape (4, 4)
        Eigenvector columns as returned by ``mpmath.eig``.
    g_ab : np.ndarray, shape (4, 4)
        Covariant metric.
    tol : float
        Timelike / null threshold (matches float64 classifier).

    Returns
    -------
    (n_timelike, n_null)
    """
    # First pass: compute all 4 quadratic forms ``v^T g v`` per eigenvector.
    quads: list[float] = []
    for k in range(4):
        v = [float(mpmath.re(evecs[a, k])) for a in range(4)]
        quad = 0.0
        for a in range(4):
            for b in range(4):
                quad += g_ab[a, b] * v[a] * v[b]
        quads.append(quad)

    # Second pass: relative-sign test .
    max_abs_quad = max(abs(q) for q in quads) if quads else 1.0
    scale = max(max_abs_quad, 1.0)
    n_timelike = sum(1 for q in quads if q / scale < -tol)
    n_null = sum(1 for q in quads if abs(q) / scale <= tol)
    return n_timelike, n_null
