"""Richardson extrapolation for grid convergence validation.

Given a quantity Q computed at multiple grid resolutions, estimates the
true (continuum) value and the observed convergence order.

Only smooth functionals are suitable for Richardson extrapolation (e.g. an
L2 norm of the violation field or an integrated violation). Discontinuous
quantities like "percentage of points violated" are NOT suitable, and neither
is a grid-sampled EXTREMUM (a min/max over samples): its residual gap
oscillates with grid alignment (aliasing) and admits no clean order; polish
such extrema to the continuum with ``analysis.extrema.refine_extremum``
instead of extrapolating them.

Uses plain Python/numpy math (not JAX) since these are post-processing
summary statistics.
"""
from __future__ import annotations

import math

import numpy as np


def richardson_extrapolation(
    values: list[float],
    grid_sizes: list[int],
    expected_order: int = 2,
) -> dict:
    """Richardson extrapolation from 3+ grid resolutions.

    Assumes ``Q(h) = Q_exact + C * h^p + O(h^{p+1})`` where
    ``h = 1 / (N - 1)``: N points over a fixed domain span N-1 cells.

    Parameters
    ----------
    values : list[float]
        Computed quantity at each resolution ``[Q(h1), Q(h2), Q(h3)]``,
        ordered from coarsest to finest.
    grid_sizes : list[int]
        Grid sizes ``[N1, N2, N3]`` (e.g. ``[25, 49, 97]``, whose spacing
        ratio is exactly 2), ordered coarsest to finest.
    expected_order : int
        Expected convergence order for validation.

    Returns
    -------
    dict
        Keys:
        - ``extrapolated_value``: Richardson estimate of Q_exact when the order
          is estimable and consistent with ``expected_order``; otherwise the
          finest computed value, unextrapolated.
        - ``observed_order``: estimated convergence order p, or ``None`` when
          the triplet is non-monotonic and p is not estimable.
        - ``error_estimate``: ``|Q_fine - Q_extrapolated|`` when
          ``error_basis == "richardson"``; otherwise the ladder spread
          ``max_i |Q_i - Q_fine|``, which assumes no order at all.
        - ``converged``: bool, True only if p was estimated AND
          ``|p - expected_order| < 1.0``.
        - ``fallback``: bool, True when the triplet was non-monotonic.
        - ``error_basis``: ``"richardson"``, ``"spread"``, or ``"exact"``,
          which of the two bounds above the error estimate is.

    Raises
    ------
    ValueError
        If fewer than 3 values/grid_sizes are provided, or if the refinement
        ladder is not geometric (``h1/h2 != h2/h3``), which the three-point
        order estimate cannot handle.
    """
    if len(values) < 3 or len(grid_sizes) < 3:
        raise ValueError(
            "Richardson extrapolation requires at least 3 resolutions, "
            f"got {len(values)}"
        )

    # Use the last 3 resolutions (coarsest -> finest)
    Q1, Q2, Q3 = values[-3], values[-2], values[-1]
    N1, N2, N3 = grid_sizes[-3], grid_sizes[-2], grid_sizes[-1]
    # Endpoint-inclusive grids: N points over a fixed domain span N-1 cells, so
    # the spacing that the error expansion is in goes as 1/(N-1). Using 1/N put
    # exactly second-order data at order 2.069 with a nonzero extrapolant, and
    # rejected [25, 49, 97], whose spacing ratio is exactly 2.
    h1, h2, h3 = 1.0 / (N1 - 1), 1.0 / (N2 - 1), 1.0 / (N3 - 1)

    # The three-point order estimate is valid only on a geometric ladder.
    r, r2 = h1 / h2, h2 / h3
    if abs(r - r2) > 1e-9 * max(r, r2):
        raise ValueError(
            "richardson_extrapolation needs a geometric refinement ladder: "
            f"h1/h2 = {r:.6f} but h2/h3 = {r2:.6f} for N = "
            f"{[N1, N2, N3]}. Three points cannot separate the order from a "
            "varying ratio."
        )

    # Assumption-free bound, used whenever Richardson is not earned.
    spread = max(abs(Q1 - Q3), abs(Q2 - Q3))

    # Estimate convergence order
    dQ12 = Q1 - Q2
    dQ23 = Q2 - Q3

    # Relative: an absolute 1e-30 called [4e-100, 2e-100, 1e-100] exact while
    # it was still halving at every level.
    q_scale = max(abs(Q1), abs(Q2), abs(Q3))
    if abs(dQ23) <= 1e-15 * q_scale:
        # Already converged to machine precision
        return {
            "extrapolated_value": float(Q3),
            "observed_order": float(expected_order),
            "error_estimate": 0.0,
            "converged": True,
            "fallback": False,
            "error_basis": "exact",
        }

    ratio = dQ12 / dQ23
    if ratio <= 0:
        # Non-monotonic: no order is estimable. This used to substitute
        # p = expected_order and then test |p - expected_order| < 1, which is
        # always true, so a failed convergence reported itself as converged.
        return {
            "extrapolated_value": float(Q3),
            "observed_order": None,
            "error_estimate": float(spread),
            "converged": False,
            "fallback": True,
            "error_basis": "spread",
        }

    p = math.log(abs(ratio)) / math.log(r)

    # Richardson extrapolation: Q_ext = (r^p * Q_fine - Q_coarse) / (r^p - 1)
    rp = r**p
    if abs(rp - 1.0) < 1e-30:
        Q_ext = Q3
    else:
        Q_ext = (rp * Q3 - Q2) / (rp - 1.0)

    converged = abs(p - expected_order) < 1.0
    return {
        "extrapolated_value": float(Q_ext),
        "observed_order": float(p),
        # Richardson is an error estimate only when the order supports it.
        "error_estimate": float(abs(Q3 - Q_ext) if converged else spread),
        "converged": bool(converged),
        "fallback": False,
        "error_basis": "richardson" if converged else "spread",
    }


def f_miss_stability(
    values: list[float],
    abs_tol_pp: float = 0.5,
    rel_tol: float = 0.05,
) -> dict:
    """Resolution-stability test for a (discontinuous) miss-fraction series.

    The missed-violation fraction ``f_miss`` is a discontinuous
    point-count statistic and is *not* a valid input for Richardson
    extrapolation. Instead we ask whether it is *stable* under grid
    refinement. A pure relative criterion (``max|f - mean| / mean``) is
    misleading when the mean is tiny: a fraction of ~0.8% wandering by a
    physically negligible 0.15 percentage points registers as a spurious
    ~10% "instability" purely because the denominator is small.

    This helper therefore declares stability when *either* the absolute
    spread is within ``abs_tol_pp`` percentage points *or* the relative
    deviation is within ``rel_tol``. The absolute floor prevents
    false-instability on small-magnitude fractions while the relative
    bound still catches genuine drift in large fractions.

    Parameters
    ----------
    values : list[float]
        ``f_miss`` in *percent* at each resolution (coarsest -> finest).
    abs_tol_pp : float
        Absolute tolerance in percentage points (default 0.5 pp).
    rel_tol : float
        Relative tolerance on max deviation from the mean (default 0.05).

    Returns
    -------
    dict
        Keys ``values``, ``mean``, ``max_dev_pp`` (max |value - mean| in
        pp), ``max_dev_rel`` (max_dev_pp / mean), ``stable`` (bool), and
        ``criterion`` (human-readable description).
    """
    if len(values) < 2:
        raise ValueError(f"stability test needs >= 2 resolutions, got {len(values)}")

    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    max_dev_pp = float(np.max(np.abs(arr - mean)))
    max_dev_rel = max_dev_pp / mean if mean > 1e-12 else math.inf

    stable = (max_dev_pp <= abs_tol_pp) or (max_dev_rel <= rel_tol)

    return {
        "values": [float(v) for v in arr],
        "mean": mean,
        "max_dev_pp": max_dev_pp,
        "max_dev_rel": max_dev_rel,
        "stable": bool(stable),
        "criterion": (
            f"stable if max|f - mean| <= {abs_tol_pp} pp (absolute floor) "
            f"OR <= {rel_tol:.0%} relative"
        ),
    }


def compute_convergence_quantity(
    margins: np.ndarray,
    quantity: str,
    cell_volume: float = 1.0,
) -> float:
    """Extract a scalar convergence quantity from margin data.

    Parameters
    ----------
    margins : np.ndarray
        Margin array from grid evaluation (any shape).
    quantity : str
        One of:
        - ``"min_margin"``: ``nanmin`` of margins (most violated point)
        - ``"l2_violation"``: discrete L2 norm of the negative margins,
          ``sqrt(sum f^2 * cell_volume)``
        - ``"integrated_violation"``: sum of |margin| where violated,
          times cell volume (volume-integrated violation)
    cell_volume : float
        Volume of a single grid cell. Required for both
        ``"integrated_violation"`` and ``"l2_violation"``: without it
        neither is an integral and both grow with the point count.

    Returns
    -------
    float
        Scalar quantity suitable for Richardson extrapolation.

    Raises
    ------
    ValueError
        If ``quantity`` is not recognized.
    """
    flat = np.asarray(margins).ravel()

    if quantity == "min_margin":
        return float(np.nanmin(flat))

    # Relative roundoff gate: an absolute -1e-10 reports zero violation for a
    # grid whose margins are all of order 1e-12.
    finite = flat[np.isfinite(flat)]
    scale = float(np.max(np.abs(finite))) if finite.size else 0.0
    cut = -1e-10 * scale

    if quantity == "l2_violation":
        violated = flat[flat < cut]
        if violated.size == 0:
            return 0.0
        # sqrt(sum f^2 dV), not sqrt(sum f^2): without the volume this counts
        # points, and its "order" -1.4 was just -log_2(r^{3/2}).
        return float(np.sqrt(np.sum(violated**2) * cell_volume))

    elif quantity == "integrated_violation":
        violated = flat[flat < cut]
        if violated.size == 0:
            return 0.0
        return float(np.sum(np.abs(violated)) * cell_volume)

    else:
        raise ValueError(
            f"Unknown convergence quantity '{quantity}'. "
            "Expected one of: 'min_margin', 'l2_violation', 'integrated_violation'"
        )
