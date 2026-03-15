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


class WallRestrictedStats(NamedTuple):
    """Wall-restricted Hawking-Ellis Type breakdown and EC statistics.

    All counts and fractions are conditional on the provided wall mask.

    Attributes
    ----------
    n_type_i : int
        Number of Type I points within the wall.
    n_type_ii : int
        Number of Type II points within the wall.
    n_type_iii : int
        Number of Type III points within the wall.
    n_type_iv : int
        Number of Type IV points within the wall.
    n_total : int
        Total number of points within the wall.
    frac_type_i : float
        Fraction of wall points that are Type I.
    frac_type_ii : float
        Fraction of wall points that are Type II.
    frac_type_iii : float
        Fraction of wall points that are Type III.
    frac_type_iv : float
        Fraction of wall points that are Type IV.
    nec_violated : int
        Number of NEC-violated points within the wall.
    wec_violated : int
        Number of WEC-violated points within the wall.
    sec_violated : int
        Number of SEC-violated points within the wall.
    dec_violated : int
        Number of DEC-violated points within the wall.
    nec_frac_violated : float
        Fraction of wall points with NEC violations.
    wec_frac_violated : float
        Fraction of wall points with WEC violations.
    sec_frac_violated : float
        Fraction of wall points with SEC violations.
    dec_frac_violated : float
        Fraction of wall points with DEC violations.
    nec_miss_rate : float | None
        Conditional NEC miss rate, or None if no violations exist.
    wec_miss_rate : float | None
        Conditional WEC miss rate, or None if no violations exist.
    sec_miss_rate : float | None
        Conditional SEC miss rate, or None if no violations exist.
    dec_miss_rate : float | None
        Conditional DEC miss rate, or None if no violations exist.
    """

    # Type counts within wall
    n_type_i: int
    n_type_ii: int
    n_type_iii: int
    n_type_iv: int
    n_total: int  # Total points in wall

    # Type fractions
    frac_type_i: float
    frac_type_ii: float
    frac_type_iii: float
    frac_type_iv: float

    # Per-condition violation counts within wall
    nec_violated: int
    wec_violated: int
    sec_violated: int
    dec_violated: int

    # Per-condition violation fractions within wall
    nec_frac_violated: float
    wec_frac_violated: float
    sec_frac_violated: float
    dec_frac_violated: float

    # Per-condition miss rates within wall (None if no violations exist)
    nec_miss_rate: float | None
    wec_miss_rate: float | None
    sec_miss_rate: float | None
    dec_miss_rate: float | None
