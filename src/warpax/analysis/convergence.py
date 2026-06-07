"""Richardson extrapolation for grid convergence validation.

Given a quantity Q computed at multiple grid resolutions, estimates the
true (continuum) value and the observed convergence order. This is
essential for validating that energy condition results are numerically
converged and not grid artifacts.

Only smooth quantities are suitable for Richardson extrapolation (e.g.
minimum margin, L2 norm of violation field, integrated violation).
Discontinuous quantities like "percentage of points violated" are NOT
suitable (discontinuous quantities break Richardson extrapolation assumptions).

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
    ``h = 1 / N`` (grid spacing proportional to ``1/N``).

    Parameters
    ----------
    values : list[float]
        Computed quantity at each resolution ``[Q(h1), Q(h2), Q(h3)]``,
        ordered from coarsest to finest.
    grid_sizes : list[int]
        Grid sizes ``[N1, N2, N3]`` (e.g. ``[25, 50, 100]``), ordered
        coarsest to finest.
    expected_order : int
        Expected convergence order for validation.

    Returns
    -------
    dict
        Keys:
        - ``extrapolated_value``: Richardson-extrapolated estimate of Q_exact
        - ``observed_order``: estimated convergence order p
        - ``error_estimate``: |Q_fine - Q_extrapolated|
        - ``converged``: bool, True if |p - expected_order| < 1.0

    Raises
    ------
    ValueError
        If fewer than 3 values/grid_sizes are provided.
    """
    if len(values) < 3 or len(grid_sizes) < 3:
        raise ValueError(
            "Richardson extrapolation requires at least 3 resolutions, "
            f"got {len(values)}"
        )

    # Use the last 3 resolutions (coarsest -> finest)
    Q1, Q2, Q3 = values[-3], values[-2], values[-1]
    N1, N2, N3 = grid_sizes[-3], grid_sizes[-2], grid_sizes[-1]
    h1, h2, _h3 = 1.0 / N1, 1.0 / N2, 1.0 / N3

    # Refinement ratio
    r = h1 / h2

    # Estimate convergence order
    dQ12 = Q1 - Q2
    dQ23 = Q2 - Q3

    # Guard against zero denominator
    if abs(dQ23) < 1e-30:
        # Already converged to machine precision
        return {
            "extrapolated_value": float(Q3),
            "observed_order": float(expected_order),
            "error_estimate": 0.0,
            "converged": True,
            "fallback": False,
        }

    ratio = dQ12 / dQ23
    fallback = False
    if ratio <= 0:
        # Non-monotonic convergence: order estimation not meaningful.
        # Fall back to expected order for extrapolation; flag it.
        p = float(expected_order)
        fallback = True
    else:
        p = math.log(abs(ratio)) / math.log(r)

    # Richardson extrapolation: Q_ext = (r^p * Q_fine - Q_coarse) / (r^p - 1)
    rp = r**p
    if abs(rp - 1.0) < 1e-30:
        Q_ext = Q3
    else:
        Q_ext = (rp * Q3 - Q2) / (rp - 1.0)

    error_estimate = abs(Q3 - Q_ext)
    converged = abs(p - expected_order) < 1.0

    return {
        "extrapolated_value": float(Q_ext),
        "observed_order": float(p),
        "error_estimate": float(error_estimate),
        "converged": bool(converged),
        "fallback": fallback,
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
        - ``"l2_violation"``: L2 norm of negative margins
        - ``"integrated_violation"``: sum of |margin| where violated,
          times cell volume (volume-integrated violation)
    cell_volume : float
        Volume of a single grid cell (for ``"integrated_violation"``).

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

    elif quantity == "l2_violation":
        violated = flat[flat < -1e-10]
        if violated.size == 0:
            return 0.0
        return float(np.sqrt(np.sum(violated**2)))

    elif quantity == "integrated_violation":
        violated = flat[flat < -1e-10]
        if violated.size == 0:
            return 0.0
        return float(np.sum(np.abs(violated)) * cell_volume)

    else:
        raise ValueError(
            f"Unknown convergence quantity '{quantity}'. "
            "Expected one of: 'min_margin', 'l2_violation', 'integrated_violation'"
        )
