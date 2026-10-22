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

from typing import NamedTuple

from jaxtyping import Array, Float


class ClassificationResult(NamedTuple):
    """Result of Hawking-Ellis type classification at a single point.

    Attributes
    ----------
    he_type
        Int-valued label: 1=I, 2=II, 3=III, 4=IV (NaN for non-finite input).
    eigenvalues, eigenvalues_imag
        Real and imaginary parts of the eigenvalues of ``T^a_b``.
    eigenvectors
        Real part of the eigenvectors (columns).
    rho, pressures
        Energy density and principal pressures for Type I, else NaN.
    is_vacuum
        1.0 when the point is vacuum, else 0.0. Vacuum requires BOTH
        ``max|lambda| < tol`` AND an exactly zero tensor ``T^a_b``; a small but
        nonzero tensor is never vacuum (negative null dust has a zero spectrum
        yet is Type II). Vacuum points are classified ``he_type = 1``; this flag
        lets callers exclude them.
    """

    he_type: Float[Array, ""]
    eigenvalues: Float[Array, "4"]
    eigenvectors: Float[Array, "4 4"]
    rho: Float[Array, ""]
    pressures: Float[Array, "3"]
    eigenvalues_imag: Float[Array, "4"]
    is_vacuum: Float[Array, ""]


class ECPointResult(NamedTuple):
    """Full energy condition result at a single spacetime point.

    Combines classification, eigenvalue-based margins (Type I), and
    worst-observer information from optimization (all types).

    ``he_type`` is the Hawking-Ellis type (1-4); ``eigenvalues`` the real
    eigenvalues of ``T^a_b``; ``rho``/``pressures`` the energy density and
    principal pressures (Type I) or NaN; ``*_margin`` the minimum margin per
    condition; ``worst_observer`` the ``u^a`` and ``worst_params`` the
    ``(zeta, theta, phi)`` of the worst-case observer. ``*_opt_margin`` are the
    raw optimizer margins before the merge with the algebraic Type-I proxies.
    """

    he_type: Float[Array, ""]
    eigenvalues: Float[Array, "4"]
    rho: Float[Array, ""]
    pressures: Float[Array, "3"]
    nec_margin: Float[Array, ""]
    wec_margin: Float[Array, ""]
    sec_margin: Float[Array, ""]
    dec_margin: Float[Array, ""]
    worst_observer: Float[Array, "4"]
    worst_params: Float[Array, "3"]
    nec_opt_margin: Float[Array, ""] | None = None
    wec_opt_margin: Float[Array, ""] | None = None
    sec_opt_margin: Float[Array, ""] | None = None
    dec_opt_margin: Float[Array, ""] | None = None
    eig_ill_conditioned: Float[Array, ""] | None = None


class ECSummary(NamedTuple):
    """Per-condition summary statistics across a grid.

    ``fraction_violated`` is the fraction of points violating, ``max_violation``
    the largest (most negative) margin and ``min_margin`` the minimum margin.
    """

    fraction_violated: Float[Array, ""]
    max_violation: Float[Array, ""]
    min_margin: Float[Array, ""]


class ECGridResult(NamedTuple):
    """Energy condition results across an entire evaluation grid.

    Each field has leading grid dimensions ``(*grid_shape,)``. ``nec_margins``
    at non-Type-I points is the smaller of the sampled optimizer margin and the
    exact Eulerian null deficit, so the multistart search never reports a
    weaker NEC than the closed-form minimum; ``nec_opt_margins`` keeps the raw
    optimizer value (NaN where the optimizer was skipped). ``eig_ill_conditioned``
    is 1.0 wherever the eigenvector matrix fails the conditioning guard; it
    changes the route only at Type I, where such points carry optimizer margins
    instead of eigenvalue slacks. ``n_vacuum`` counts near-vacuum points
    (``||T|| ~ 0``), a subset of ``n_type_i``. ``*_opt_converged`` is 1.0 where
    the optimizer converged and 0.0 where it hit ``max_steps``; ``*_opt_n_steps``
    is the number of steps taken.
    """

    he_types: Float[Array, "..."]
    eigenvalues: Float[Array, "... 4"]
    rho: Float[Array, "..."]
    pressures: Float[Array, "... 3"]
    nec_margins: Float[Array, "..."]
    wec_margins: Float[Array, "..."]
    sec_margins: Float[Array, "..."]
    dec_margins: Float[Array, "..."]
    worst_observers: Float[Array, "... 4"]
    worst_params: Float[Array, "... 3"]
    nec_summary: ECSummary
    wec_summary: ECSummary
    sec_summary: ECSummary
    dec_summary: ECSummary
    nec_opt_margins: Float[Array, "..."] | None
    wec_opt_margins: Float[Array, "..."] | None
    sec_opt_margins: Float[Array, "..."] | None
    dec_opt_margins: Float[Array, "..."] | None
    n_type_i: int | None
    n_type_ii: int | None
    n_type_iii: int | None
    n_type_iv: int | None
    n_vacuum: int | None
    max_imag_eigenvalue: float | None
    nec_opt_converged: Float[Array, "..."] | None
    wec_opt_converged: Float[Array, "..."] | None
    sec_opt_converged: Float[Array, "..."] | None
    dec_opt_converged: Float[Array, "..."] | None
    nec_opt_n_steps: Float[Array, "..."] | None
    wec_opt_n_steps: Float[Array, "..."] | None
    sec_opt_n_steps: Float[Array, "..."] | None
    dec_opt_n_steps: Float[Array, "..."] | None
    eig_ill_conditioned: Float[Array, "..."] | None = None


class FrameFreeGridResult(NamedTuple):
    """Frame-independent (eigenvalue-only) certification across a grid.

    Unlike :class:`ECGridResult`, this carries NO optimizer or Eulerian-frame
    quantities: every field derives solely from the Hawking-Ellis classification
    of ``T^a_b`` and the Type-I eigenvalue inequalities. Because neither the
    Eulerian normal nor a timelike tetrad is ever constructed, this result is
    well-defined at ALL warp velocities, including v_s >= 1, where the
    coordinate-stationary observer d_t ceases to be timelike while the Eulerian
    normal stays unit timelike.

    For non-Type-I points rho and pressures are NaN (no invariant rest frame /
    proper energy density exists), but the NEC/WEC/SEC/DEC margins carry the
    certified cap-free witness (< 0 for Type III/IV), not NaN. Use the type census
    to quantify the non-Type-I (intrinsically observer-dependent) fraction.

    Each margin/scalar field has leading grid dimensions ``(*grid_shape,)``.
    ``eigenvalues``/``eigenvalues_imag`` are the real and imaginary parts of the
    eigenvalues of ``T^a_b``; ``nec_noise_floor`` is the magnitude below which a
    margin is inconclusive; ``is_vacuum`` is 1.0 at near-vacuum points.
    ``lmi_substituted`` is 1.0 where the margin came from the LMI rather than the
    eigenvalue inequalities; the two are different quantities and only their sign
    is comparable, so an invariant eigenvalue minimum must skip these points.
    """

    he_types: Float[Array, "..."]
    eigenvalues: Float[Array, "... 4"]
    eigenvalues_imag: Float[Array, "... 4"]
    rho: Float[Array, "..."]
    pressures: Float[Array, "... 3"]
    nec_noise_floor: Float[Array, "..."]
    nec_margins: Float[Array, "..."]
    wec_margins: Float[Array, "..."]
    sec_margins: Float[Array, "..."]
    dec_margins: Float[Array, "..."]
    is_vacuum: Float[Array, "..."]
    n_type_i: int
    n_type_ii: int
    n_type_iii: int
    n_type_iv: int
    n_vacuum: int
    n_total: int
    max_imag_eigenvalue: float
    lmi_substituted: Float[Array, "..."] | None = None


class WallRestrictedStats(NamedTuple):
    """Wall-restricted HE-type breakdown and EC violation statistics.

    All counts and fractions are conditional on the wall mask. ``*_miss_rate``
    fields are ``None`` when the corresponding condition has zero violations
    inside the wall (the conditional rate is undefined). ``n_total`` is the
    number of points in the wall.
    """

    n_type_i: int
    n_type_ii: int
    n_type_iii: int
    n_type_iv: int
    n_total: int

    frac_type_i: float
    frac_type_ii: float
    frac_type_iii: float
    frac_type_iv: float

    nec_violated: int
    wec_violated: int
    sec_violated: int
    dec_violated: int

    nec_frac_violated: float
    wec_frac_violated: float
    sec_frac_violated: float
    dec_frac_violated: float

    nec_miss_rate: float | None
    wec_miss_rate: float | None
    sec_miss_rate: float | None
    dec_miss_rate: float | None
