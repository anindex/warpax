"""Fast eigenvalue-based energy condition verification for Type-I stress-energy.

For Type I tensors with eigenvalues {-ρ, p₁, p₂, p₃}:

- WEC: ρ ≥ 0  and  ρ + pᵢ ≥ 0  for all i
- NEC: ρ + pᵢ ≥ 0  for all i
- DEC: ρ ≥ |pᵢ|  for all i
- SEC: ρ + pᵢ ≥ 0  for all i  and  ρ + Σpᵢ ≥ 0

These are O(1) per point and fully vectorizable across the grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class EigenvalueResult:
    """Result of eigenvalue-based energy condition check at a point or on a grid."""

    satisfied: bool
    margin: float  # Minimum margin (negative = violated)


def check_wec_eigenvalue(rho: float | NDArray, pressures: NDArray) -> EigenvalueResult:
    """Weak Energy Condition: ρ ≥ 0 and ρ + pᵢ ≥ 0."""
    pressures = np.asarray(pressures)
    rho = np.asarray(rho)

    if pressures.ndim == 1:
        margins = np.concatenate([[rho], rho + pressures])
    else:
        margins = np.concatenate([rho[..., None], rho[..., None] + pressures], axis=-1)

    min_margin = float(np.min(margins))
    return EigenvalueResult(satisfied=min_margin >= 0, margin=min_margin)


def check_nec_eigenvalue(rho: float | NDArray, pressures: NDArray) -> EigenvalueResult:
    """Null Energy Condition: ρ + pᵢ ≥ 0."""
    pressures = np.asarray(pressures)
    rho = np.asarray(rho)

    if pressures.ndim == 1:
        margins = rho + pressures
    else:
        margins = rho[..., None] + pressures

    min_margin = float(np.min(margins))
    return EigenvalueResult(satisfied=min_margin >= 0, margin=min_margin)


def check_dec_eigenvalue(rho: float | NDArray, pressures: NDArray) -> EigenvalueResult:
    """Dominant Energy Condition: ρ ≥ |pᵢ|."""
    pressures = np.asarray(pressures)
    rho = np.asarray(rho)

    if pressures.ndim == 1:
        margins = rho - np.abs(pressures)
    else:
        margins = rho[..., None] - np.abs(pressures)

    min_margin = float(np.min(margins))
    return EigenvalueResult(satisfied=min_margin >= 0, margin=min_margin)


def check_sec_eigenvalue(rho: float | NDArray, pressures: NDArray) -> EigenvalueResult:
    """Strong Energy Condition: ρ + pᵢ ≥ 0 and ρ + Σpᵢ ≥ 0."""
    pressures = np.asarray(pressures)
    rho = np.asarray(rho)

    if pressures.ndim == 1:
        trace_margin = rho + np.sum(pressures)
        margins = np.concatenate([rho + pressures, [trace_margin]])
    else:
        trace_margin = rho + np.sum(pressures, axis=-1)
        margins = np.concatenate(
            [rho[..., None] + pressures, trace_margin[..., None]], axis=-1
        )

    min_margin = float(np.min(margins))
    return EigenvalueResult(satisfied=min_margin >= 0, margin=min_margin)


def check_all_eigenvalue(
    rho: float | NDArray, pressures: NDArray
) -> dict[str, EigenvalueResult]:
    """Check all four energy conditions using eigenvalue method."""
    return {
        "WEC": check_wec_eigenvalue(rho, pressures),
        "NEC": check_nec_eigenvalue(rho, pressures),
        "DEC": check_dec_eigenvalue(rho, pressures),
        "SEC": check_sec_eigenvalue(rho, pressures),
    }


def check_eigenvalue_grid(
    rho_field: NDArray, pressure_field: NDArray
) -> dict[str, NDArray]:
    """Vectorized energy condition margins across an entire grid.

    Parameters
    ----------
    rho_field : NDArray
        Energy density, shape (*grid_shape,).
    pressure_field : NDArray
        Principal pressures, shape (*grid_shape, 3).

    Returns
    -------
    dict
        Maps condition name to margin field, shape (*grid_shape,).
        Negative values indicate violation.
    """
    p = pressure_field

    # WEC margin: min(ρ, ρ+p1, ρ+p2, ρ+p3)
    wec_candidates = np.stack(
        [rho_field, rho_field + p[..., 0], rho_field + p[..., 1], rho_field + p[..., 2]],
        axis=-1,
    )
    wec_margin = np.min(wec_candidates, axis=-1)

    # NEC margin: min(ρ+p1, ρ+p2, ρ+p3)
    nec_candidates = rho_field[..., None] + p
    nec_margin = np.min(nec_candidates, axis=-1)

    # DEC margin: min(ρ-|p1|, ρ-|p2|, ρ-|p3|)
    dec_candidates = rho_field[..., None] - np.abs(p)
    dec_margin = np.min(dec_candidates, axis=-1)

    # SEC margin: min(ρ+p1, ρ+p2, ρ+p3, ρ+p1+p2+p3)
    p_sum = np.sum(p, axis=-1)
    sec_candidates = np.stack(
        [rho_field + p[..., 0], rho_field + p[..., 1], rho_field + p[..., 2],
         rho_field + p_sum],
        axis=-1,
    )
    sec_margin = np.min(sec_candidates, axis=-1)

    return {
        "WEC": wec_margin,
        "NEC": nec_margin,
        "DEC": dec_margin,
        "SEC": sec_margin,
    }
