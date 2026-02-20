"""Optimization-based energy condition verification for general stress-energy tensors.

When the stress-energy is not Type I (or as a validation check), we minimize
the energy condition functional over the observer parameter space:

- WEC: min_{ζ,θ,φ} T_{ab} u^a u^b  where u is timelike (rapidity-parameterized)
- NEC: min_{θ,φ} T_{ab} k^a k^b  where k is null
- DEC: check that T_{ab} u^b is causal (timelike or null) for all u
- SEC: min_{ζ,θ,φ} (R_{ab} u^a u^b) ≡ min (T_{ab} - ½T g_{ab}) u^a u^b

Uses multi-start L-BFGS-B + differential_evolution as fallback for robustness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, minimize

from .observer import (
    compute_orthonormal_tetrad,
    null_from_angles,
    timelike_from_rapidity,
)


@dataclass
class OptimizationResult:
    """Result of optimization-based energy condition check."""

    satisfied: bool
    margin: float  # Minimum value found (negative = violated)
    worst_params: NDArray  # Parameters at the minimum
    worst_vector: NDArray  # The 4-vector achieving the minimum


def check_wec_optimization(
    T_ab: NDArray, g_ab: NDArray, n_starts: int = 8, zeta_max: float = 5.0
) -> OptimizationResult:
    """Minimize T_{ab} u^a u^b over timelike observers.

    A negative minimum indicates WEC violation.
    """
    tetrad = compute_orthonormal_tetrad(g_ab)

    def objective(params):
        zeta, theta, phi = params
        u = timelike_from_rapidity(zeta, theta, phi, tetrad)
        return u @ T_ab @ u

    bounds = [(0, zeta_max), (0, np.pi), (0, 2 * np.pi)]

    best_result = _multistart_minimize(objective, bounds, n_starts)

    # Fallback: differential evolution for global search
    de_result = differential_evolution(objective, bounds, seed=42, maxiter=100, tol=1e-10)
    if de_result.fun < best_result.fun:
        best_result = de_result

    params = best_result.x
    u = timelike_from_rapidity(params[0], params[1], params[2], tetrad)

    return OptimizationResult(
        satisfied=best_result.fun >= -1e-12,
        margin=float(best_result.fun),
        worst_params=params,
        worst_vector=u,
    )


def check_nec_optimization(
    T_ab: NDArray, g_ab: NDArray, n_starts: int = 8
) -> OptimizationResult:
    """Minimize T_{ab} k^a k^b over null directions.

    NEC is a 2D optimization over S² (θ, φ).
    """
    tetrad = compute_orthonormal_tetrad(g_ab)

    def objective(params):
        theta, phi = params
        k = null_from_angles(theta, phi, tetrad)
        return k @ T_ab @ k

    bounds = [(0, np.pi), (0, 2 * np.pi)]

    best_result = _multistart_minimize(objective, bounds, n_starts)

    de_result = differential_evolution(objective, bounds, seed=42, maxiter=100, tol=1e-10)
    if de_result.fun < best_result.fun:
        best_result = de_result

    params = best_result.x
    k = null_from_angles(params[0], params[1], tetrad)

    return OptimizationResult(
        satisfied=best_result.fun >= -1e-12,
        margin=float(best_result.fun),
        worst_params=params,
        worst_vector=k,
    )


def check_dec_optimization(
    T_ab: NDArray, g_ab: NDArray, n_starts: int = 8, zeta_max: float = 5.0
) -> OptimizationResult:
    """Check Dominant Energy Condition via optimization.

    DEC requires: (1) WEC satisfied, and (2) T^a_b u^b is causal for all timelike u.
    We check (2) by minimizing -g_{ab} (T^a_c u^c)(T^b_d u^d) over timelike u.
    If this quantity is positive, the energy flux is spacelike (violation).
    """
    tetrad = compute_orthonormal_tetrad(g_ab)
    g_inv = np.linalg.inv(g_ab)
    T_mixed = g_inv @ T_ab  # T^a_b

    def objective(params):
        zeta, theta, phi = params
        u = timelike_from_rapidity(zeta, theta, phi, tetrad)
        # Energy flux: j^a = -T^a_b u^b
        j = -T_mixed @ u
        # Causal character: g_{ab} j^a j^b (negative = timelike/null = OK)
        return j @ g_ab @ j  # We want this ≤ 0 for DEC

    bounds = [(0, zeta_max), (0, np.pi), (0, 2 * np.pi)]

    best_result = _multistart_minimize(objective, bounds, n_starts)

    de_result = differential_evolution(objective, bounds, seed=42, maxiter=100, tol=1e-10)
    if de_result.fun < best_result.fun:
        best_result = de_result

    params = best_result.x
    u = timelike_from_rapidity(params[0], params[1], params[2], tetrad)

    # DEC satisfied if max(causal character) ≤ 0, i.e., margin = -max ≥ 0
    return OptimizationResult(
        satisfied=best_result.fun <= 1e-12,
        margin=float(-best_result.fun),
        worst_params=params,
        worst_vector=u,
    )


def check_sec_optimization(
    T_ab: NDArray, g_ab: NDArray, n_starts: int = 8, zeta_max: float = 5.0
) -> OptimizationResult:
    """Check Strong Energy Condition via optimization.

    SEC: (T_{ab} - ½ T g_{ab}) u^a u^b ≥ 0 for all timelike u.
    Equivalently: R_{ab} u^a u^b ≥ 0 (Raychaudhuri equation).
    """
    tetrad = compute_orthonormal_tetrad(g_ab)
    g_inv = np.linalg.inv(g_ab)
    T_trace = np.einsum("ab,ab", g_inv, T_ab)
    sec_tensor = T_ab - 0.5 * T_trace * g_ab

    def objective(params):
        zeta, theta, phi = params
        u = timelike_from_rapidity(zeta, theta, phi, tetrad)
        return u @ sec_tensor @ u

    bounds = [(0, zeta_max), (0, np.pi), (0, 2 * np.pi)]

    best_result = _multistart_minimize(objective, bounds, n_starts)

    de_result = differential_evolution(objective, bounds, seed=42, maxiter=100, tol=1e-10)
    if de_result.fun < best_result.fun:
        best_result = de_result

    params = best_result.x
    u = timelike_from_rapidity(params[0], params[1], params[2], tetrad)

    return OptimizationResult(
        satisfied=best_result.fun >= -1e-12,
        margin=float(best_result.fun),
        worst_params=params,
        worst_vector=u,
    )


def _multistart_minimize(objective, bounds, n_starts, rng_seed=42):
    """Multi-start L-BFGS-B minimization."""
    rng = np.random.default_rng(rng_seed)
    best_result = None

    for _ in range(n_starts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        try:
            result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
            if best_result is None or result.fun < best_result.fun:
                best_result = result
        except Exception:
            continue

    if best_result is None:
        # All starts failed; return something
        x0 = np.array([(lo + hi) / 2 for lo, hi in bounds])
        best_result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    return best_result
