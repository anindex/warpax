"""ADM grid validation and 3+1 decomposition of a covariant 4-metric grid."""

from __future__ import annotations

import numpy as np
from jax.typing import ArrayLike

__all__ = ["adm_from_covariant_grid", "validate_adm_grids"]


def _first_bad(mask: np.ndarray) -> tuple[int, tuple[int, ...]]:
    count = int(np.count_nonzero(mask))
    first = tuple(int(i) for i in np.argwhere(mask)[0])
    return count, first


def _check_gamma(gamma_dd: np.ndarray) -> None:
    """Raise ValueError unless gamma_ij is finite, symmetric and positive definite everywhere."""
    bad_finite = ~np.isfinite(gamma_dd).all(axis=(-2, -1))
    if bad_finite.any():
        count, first = _first_bad(bad_finite)
        raise ValueError(
            f"spatial metric gamma_ij is non-finite at {count} grid point(s); "
            f"first at index {first}."
        )
    asym = np.abs(gamma_dd - np.swapaxes(gamma_dd, -1, -2)).max(axis=(-2, -1))
    bad_sym = ~(asym <= 1e-12 * np.maximum(1.0, np.abs(gamma_dd).max(axis=(-2, -1))))
    if bad_sym.any():
        count, first = _first_bad(bad_sym)
        raise ValueError(
            f"spatial metric gamma_ij is not symmetric at {count} grid point(s); "
            f"first at index {first}."
        )
    min_eig = np.linalg.eigvalsh(gamma_dd)[..., 0]
    bad_spd = ~(min_eig > 0.0)
    if bad_spd.any():
        count, first = _first_bad(bad_spd)
        raise ValueError(
            f"spatial metric gamma_ij is not positive definite at {count} grid point(s); "
            f"first at index {first} (min eigenvalue {min_eig[first]})."
        )


def validate_adm_grids(alpha: ArrayLike, beta: ArrayLike, gamma: ArrayLike) -> None:
    """Raise ValueError unless the ADM grids describe a valid 3+1 slicing at every point.

    Requires ``alpha`` finite and positive, ``beta`` finite, and ``gamma``
    (shape ``(..., 3, 3)``) finite, symmetric and positive definite.
    The message carries the offending count and the first offending grid index.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    bad_alpha = ~(np.isfinite(alpha) & (alpha > 0.0))
    if bad_alpha.any():
        count, first = _first_bad(bad_alpha)
        raise ValueError(
            f"lapse alpha is non-finite or <= 0 at {count} grid point(s); "
            f"first at index {first} (alpha = {alpha[first]}). "
            "The grid has no valid lapse there."
        )
    bad_beta = ~np.isfinite(beta).all(axis=-1)
    if bad_beta.any():
        count, first = _first_bad(bad_beta)
        raise ValueError(
            f"shift beta^i is non-finite at {count} grid point(s); first at index {first}."
        )
    _check_gamma(gamma)


def adm_from_covariant_grid(
    g_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a grid of covariant g_{ab} (shape (..., 4, 4)) into ADM quantities.

    Returns ``(alpha, beta_up, gamma_dd)`` with ``gamma_ij = g_ij``,
    ``beta^i = gamma^{ij} g_{0j}`` and ``alpha = sqrt(beta^i g_{0i} - g_00)``.
    Raises :class:`ValueError` via :func:`validate_adm_grids` if ``gamma_ij``
    is not symmetric positive definite anywhere, or if ``alpha^2`` is
    non-finite or non-positive anywhere.
    """
    g_grid = np.asarray(g_grid, dtype=np.float64)
    if g_grid.ndim < 2 or g_grid.shape[-2:] != (4, 4):
        raise ValueError(f"g_grid must have shape (..., 4, 4); got {g_grid.shape}.")
    gamma_dd = g_grid[..., 1:, 1:]
    beta_lower = g_grid[..., 0, 1:]
    g_00 = g_grid[..., 0, 0]

    _check_gamma(gamma_dd)
    beta_upper = np.einsum("...ij,...j->...i", np.linalg.inv(gamma_dd), beta_lower)
    alpha_sq = np.einsum("...i,...i->...", beta_lower, beta_upper) - g_00
    with np.errstate(invalid="ignore"):
        alpha = np.sqrt(alpha_sq)
    validate_adm_grids(alpha, beta_upper, gamma_dd)
    return alpha, beta_upper, gamma_dd
