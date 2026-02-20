"""Result types for the energy conditions pipeline.

All types are ``typing.NamedTuple`` subclasses for automatic JAX pytree
registration, consistent with the ``CurvatureResult`` / ``GridCurvatureResult``
pattern.

Conventions
-----------
- Margins are signed scalars: positive = satisfied, negative = violated.
- For non-Type-I points, ``rho`` and ``pressures`` are NaN (these fields
  are only meaningful after Type I eigenvalue extraction).
- ``he_type`` encodes the Hawking-Ellis integer label: 1=I, 2=II, 3=III, 4=IV.
"""
from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Array, Float


class ClassificationResult(NamedTuple):
    """Result of Hawking-Ellis type classification at a single point."""

    he_type: Float[Array, ""]  # int-valued: 1=I, 2=II, 3=III, 4=IV
    eigenvalues: Float[Array, "4"]  # Real part of eigenvalues of T^a_b
    eigenvectors: Float[Array, "4 4"]  # Real part of eigenvectors (columns)
    rho: Float[Array, ""]  # Energy density (Type I) or NaN
    pressures: Float[Array, "3"]  # Principal pressures (Type I) or NaN
    eigenvalues_imag: Float[Array, "4"]  # Imaginary parts of eigenvalues


class ECPointResult(NamedTuple):
    """Full energy condition result at a single spacetime point.

    Combines classification, eigenvalue-based margins (Type I), and
    worst-observer information from optimization (all types).
    """

    he_type: Float[Array, ""]  # Hawking-Ellis type (1-4)
    eigenvalues: Float[Array, "4"]  # Real eigenvalues of T^a_b
    rho: Float[Array, ""]  # Energy density (Type I) or NaN
    pressures: Float[Array, "3"]  # Principal pressures (Type I) or NaN
    nec_margin: Float[Array, ""]  # Min NEC margin
    wec_margin: Float[Array, ""]  # Min WEC margin
    sec_margin: Float[Array, ""]  # Min SEC margin
    dec_margin: Float[Array, ""]  # Min DEC margin
    worst_observer: Float[Array, "4"]  # u^a of worst-case observer
    worst_params: Float[Array, "3"]  # (zeta, theta, phi) of worst observer


class ECSummary(NamedTuple):
    """Per-condition summary statistics across a grid."""

    fraction_violated: Float[Array, ""]  # Fraction of points violating
    max_violation: Float[Array, ""]  # Largest (most negative) margin
    min_margin: Float[Array, ""]  # Minimum margin across grid


class ECGridResult(NamedTuple):
    """Energy condition results across an entire evaluation grid.

    Each field has leading grid dimensions ``(*grid_shape,)``.
    """

    he_types: Float[Array, "..."]  # int-valued, shape (*grid_shape,)
    eigenvalues: Float[Array, "... 4"]  # shape (*grid_shape, 4)
    rho: Float[Array, "..."]  # shape (*grid_shape,)
    pressures: Float[Array, "... 3"]  # shape (*grid_shape, 3)
    nec_margins: Float[Array, "..."]  # shape (*grid_shape,)
    wec_margins: Float[Array, "..."]  # shape (*grid_shape,)
    sec_margins: Float[Array, "..."]  # shape (*grid_shape,)
    dec_margins: Float[Array, "..."]  # shape (*grid_shape,)
    worst_observers: Float[Array, "... 4"]  # shape (*grid_shape, 4)
    worst_params: Float[Array, "... 3"]  # shape (*grid_shape, 3)
    nec_summary: ECSummary
    wec_summary: ECSummary
    sec_summary: ECSummary
    dec_summary: ECSummary
    # Raw optimizer margins (before merge with algebraic)
    nec_opt_margins: Float[Array, "..."] | None  # shape (*grid_shape,)
    wec_opt_margins: Float[Array, "..."] | None
    sec_opt_margins: Float[Array, "..."] | None
    dec_opt_margins: Float[Array, "..."] | None
    # Classification statistics
    n_type_i: int | None
    n_type_ii: int | None
    n_type_iii: int | None
    n_type_iv: int | None
    max_imag_eigenvalue: float | None
    # Per-condition optimizer convergence diagnostics (optional)
    nec_opt_converged: Float[Array, "..."] | None  # 1.0=converged, 0.0=hit max_steps
    wec_opt_converged: Float[Array, "..."] | None
    sec_opt_converged: Float[Array, "..."] | None
    dec_opt_converged: Float[Array, "..."] | None
    nec_opt_n_steps: Float[Array, "..."] | None
    wec_opt_n_steps: Float[Array, "..."] | None
    sec_opt_n_steps: Float[Array, "..."] | None
    dec_opt_n_steps: Float[Array, "..."] | None
