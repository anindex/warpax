"""Wall-restricted filtering and statistics for energy condition results.

Provides composable boolean mask builders (shape function wall, Frobenius
norm, determinant guard) and post-hoc statistics computation for
wall-restricted Hawking-Ellis Type breakdown and conditional miss rates.

All mask functions return boolean arrays with shape ``(*grid_shape,)`` that
compose via logical ``&`` (AND) and ``|`` (OR) operators.

Mask builders
-------------
- :func:`shape_function_mask` -- selects the warp-bubble wall region where
  the shape function is in ``[f_low, f_high]``.
- :func:`frobenius_norm_mask` -- selects points with non-trivial
  stress-energy (``||T_ab||_F > threshold``).
- :func:`determinant_guard_mask` -- excludes near-singular metric points
  (``|det(g)| < threshold``), emitting a warning when degenerate points
  are found.

Statistics
----------
- :func:`compute_wall_restricted_stats` -- computes Hawking-Ellis Type
  breakdown, per-condition violation counts/fractions, and conditional
  miss rates within a masked region.
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry.metric import MetricSpecification
from .types import ECGridResult, WallRestrictedStats


# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------


def shape_function_mask(
    metric: MetricSpecification,
    coords_batch: Float[Array, "N 4"],
    grid_shape: tuple[int, ...],
    f_low: float = 0.1,
    f_high: float = 0.9,
) -> Float[Array, "..."]:
    """Boolean mask: True where shape function is in [f_low, f_high].

    Parameters
    ----------
    metric : MetricSpecification
        Any metric with ``shape_function_value`` implemented.
    coords_batch : Float[Array, "N 4"]
        Flattened grid coordinates, shape ``(N, 4)`` where
        ``N = prod(grid_shape)``.
    grid_shape : tuple[int, ...]
        Original spatial grid dimensions for reshaping.
    f_low : float
        Lower bound for wall region (default 0.1).
    f_high : float
        Upper bound for wall region (default 0.9).

    Returns
    -------
    Float[Array, "..."]
        Boolean mask with shape ``(*grid_shape,)``.
    """
    f_vals = jax.vmap(metric.shape_function_value)(coords_batch)
    mask = (f_vals >= f_low) & (f_vals <= f_high)
    return mask.reshape(grid_shape)


def frobenius_norm_mask(
    stress_energy: Float[Array, "... 4 4"],
    threshold: float = 1e-12,
) -> Float[Array, "..."]:
    """Boolean mask: True where ``||T_ab||_F > threshold``.

    Parameters
    ----------
    stress_energy : Float[Array, "... 4 4"]
        Stress-energy tensor field with shape ``(*grid_shape, 4, 4)``.
    threshold : float
        Frobenius norm threshold (default 1e-12).

    Returns
    -------
    Float[Array, "..."]
        Boolean mask with shape ``(*grid_shape,)``.
    """
    grid_shape = stress_energy.shape[:-2]
    flat = stress_energy.reshape(-1, 4, 4)
    norms = jnp.linalg.norm(flat, axis=(-2, -1))
    return (norms > threshold).reshape(grid_shape)


def determinant_guard_mask(
    g_field: Float[Array, "... 4 4"],
    threshold: float = 1e-10,
) -> Float[Array, "..."]:
    """Boolean mask: True where ``|det(g)| >= threshold`` (non-degenerate).

    Points where this is False have near-singular metrics; EC margins
    at those points are unreliable and should be excluded from
    aggregate statistics.

    Emits a ``UserWarning`` when degenerate points are detected.

    Parameters
    ----------
    g_field : Float[Array, "... 4 4"]
        Metric tensor field with shape ``(*grid_shape, 4, 4)``.
    threshold : float
        Determinant magnitude threshold (default 1e-10).

    Returns
    -------
    Float[Array, "..."]
        Boolean mask with shape ``(*grid_shape,)``.
    """
    grid_shape = g_field.shape[:-2]
    flat_g = g_field.reshape(-1, 4, 4)
    dets = jax.vmap(jnp.linalg.det)(flat_g)
    mask = (jnp.abs(dets) >= threshold).reshape(grid_shape)

    n_degenerate = int(jnp.sum(~mask))
    if n_degenerate > 0:
        n_points = int(jnp.prod(jnp.array(grid_shape)))
        warnings.warn(
            f"{n_degenerate} of {n_points} grid points have |det(g)| < {threshold}. "
            f"EC margins at these points are unreliable.",
            stacklevel=2,
        )
    return mask


# ---------------------------------------------------------------------------
# Wall-restricted statistics
# ---------------------------------------------------------------------------


def compute_wall_restricted_stats(
    ec_result: ECGridResult,
    mask: Float[Array, "..."],
    atol: float = 1e-10,
    eulerian_margins: dict[str, Float[Array, "..."]] | None = None,
) -> WallRestrictedStats:
    """Compute Type breakdown and EC statistics within a masked region.

    Parameters
    ----------
    ec_result : ECGridResult
        Full grid energy condition results from ``verify_grid``.
    mask : Float[Array, "..."]
        Boolean mask with shape ``(*grid_shape,)``. True selects points
        to include in statistics.
    atol : float
        Absolute tolerance for violation detection (margin < -atol).
    eulerian_margins : dict[str, Float[Array, "..."]] | None
        If provided, dict with keys ``"nec"``, ``"wec"``, ``"sec"``,
        ``"dec"`` mapping to Eulerian margin arrays. Used to compute
        conditional miss rates:
        ``(eulerian >= 0) & (robust < -atol) & mask``.

    Returns
    -------
    WallRestrictedStats
        Wall-restricted Hawking-Ellis Type breakdown and EC statistics.
    """
    he_flat = ec_result.he_types.ravel()
    mask_flat = mask.ravel().astype(bool)
    n_total = int(jnp.sum(mask_flat))

    # Type counts
    n_i = int(jnp.sum((he_flat == 1.0) & mask_flat))
    n_ii = int(jnp.sum((he_flat == 2.0) & mask_flat))
    n_iii = int(jnp.sum((he_flat == 3.0) & mask_flat))
    n_iv = int(jnp.sum((he_flat == 4.0) & mask_flat))

    # Safe denominator
    n_safe = max(n_total, 1)

    # Type fractions
    frac_i = n_i / n_safe
    frac_ii = n_ii / n_safe
    frac_iii = n_iii / n_safe
    frac_iv = n_iv / n_safe

    # Per-condition violations
    conditions = {
        "nec": ec_result.nec_margins,
        "wec": ec_result.wec_margins,
        "sec": ec_result.sec_margins,
        "dec": ec_result.dec_margins,
    }

    violated_counts: dict[str, int] = {}
    violated_fracs: dict[str, float] = {}
    for cond, margins in conditions.items():
        v = int(jnp.sum((margins.ravel() < -atol) & mask_flat))
        violated_counts[cond] = v
        violated_fracs[cond] = v / n_safe

    # Miss rates (if eulerian_margins provided)
    miss_rates: dict[str, float | None] = {
        "nec": None,
        "wec": None,
        "sec": None,
        "dec": None,
    }
    if eulerian_margins is not None:
        for cond, robust_margins in conditions.items():
            eul = eulerian_margins[cond].ravel()
            rob = robust_margins.ravel()
            robust_violated = (rob < -atol) & mask_flat
            n_rob_violated = int(jnp.sum(robust_violated))
            if n_rob_violated > 0:
                missed = (eul >= 0.0) & robust_violated
                miss_rates[cond] = int(jnp.sum(missed)) / n_rob_violated
            else:
                miss_rates[cond] = None

    return WallRestrictedStats(
        n_type_i=n_i,
        n_type_ii=n_ii,
        n_type_iii=n_iii,
        n_type_iv=n_iv,
        n_total=n_total,
        frac_type_i=frac_i,
        frac_type_ii=frac_ii,
        frac_type_iii=frac_iii,
        frac_type_iv=frac_iv,
        nec_violated=violated_counts["nec"],
        wec_violated=violated_counts["wec"],
        sec_violated=violated_counts["sec"],
        dec_violated=violated_counts["dec"],
        nec_frac_violated=violated_fracs["nec"],
        wec_frac_violated=violated_fracs["wec"],
        sec_frac_violated=violated_fracs["sec"],
        dec_frac_violated=violated_fracs["dec"],
        nec_miss_rate=miss_rates["nec"],
        wec_miss_rate=miss_rates["wec"],
        sec_miss_rate=miss_rates["sec"],
        dec_miss_rate=miss_rates["dec"],
    )
