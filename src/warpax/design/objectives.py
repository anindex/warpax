"""objectives - EC-margin integrals + averaged + quantum.

Three objective flavours for the optimizer :

- :func:`ec_margin_objective(metric, objective='nec'|'wec'|'sec'|'dec')`:
  pointwise worst-observer EC margin on a probe grid; returns the
  *minimum* per-point margin (worst-point margin).
- :func:`averaged_objective(metric, geodesic, kind='anec'|'awec')`:
  composes with 02 line integrals.
- :func:`quantum_objective(metric, worldline, tau0)`: composes with
  Ford-Roman quantum inequality.

:data:`OBJECTIVE_REGISTRY` dict provides string-dispatch for the
``design_metric(objective='nec')`` API.

Reference:
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..energy_conditions.optimization import (
    optimize_dec,
    optimize_nec,
    optimize_sec,
    optimize_wec,
)
from ..geometry.geometry import compute_curvature_chain
from ..geometry.metric import MetricSpecification


# ---------------------------------------------------------------------------
# Pointwise EC margin dispatch
# ---------------------------------------------------------------------------


_OPTIMIZER_MAP: dict[str, Callable] = {
    "nec": optimize_nec,
    "wec": optimize_wec,
    "sec": optimize_sec,
    "dec": optimize_dec,
}


def ec_margin_objective(
    metric: MetricSpecification,
    objective: str = "nec",
    grid_shape: tuple[int, int, int] = (16, 16, 16),
    bounds: tuple[tuple[float, float], ...] = ((-2.0, 2.0),) * 3,
    t: float = 0.0,
    n_starts: int = 4,
) -> Float[Array, ""]:
    """Evaluate worst-point EC margin on a probe grid.

    For each grid point, computes the full curvature chain (metric ->
    ... -> stress-energy) and runs the ``objective``-appropriate
    single-point optimizer (``optimize_nec`` / ``wec`` / ``sec`` /
    ``dec``). Returns ``jnp.min`` over the probe grid.

    Parameters
    ----------
    metric
        Any :class:`MetricSpecification` (including
        :class:`ShapeFunctionMetric`).
    objective
        One of ``'nec'`` / ``'wec'`` / ``'sec'`` / ``'dec'``.
    grid_shape
        Probe grid shape ``(Nx, Ny, Nz)``. Default ``(16, 16, 16)``.
    bounds
        Per-axis probe bounds (``(min, max)`` tuples). Default
        ``((-2, 2), (-2, 2), (-2, 2))``.
    t
        Time coordinate for the probe slice (default ``0.0``).
    n_starts
        Number of BFGS starting points per probe point. Default ``4``
        (reduced from the paper-default ``16`` to keep the design loop
        cheap; optimizer in can override).

    Returns
    -------
    Float[Array, ""]
        Minimum (worst-point) EC margin across the probe grid.
    """
    if objective not in _OPTIMIZER_MAP:
        raise ValueError(
            f"ec_margin_objective: unknown objective {objective!r}; "
            f"expected one of {list(_OPTIMIZER_MAP)}."
        )
    optimizer = _OPTIMIZER_MAP[objective]

    # Build flat probe grid (N, 4)
    xs = jnp.linspace(bounds[0][0], bounds[0][1], grid_shape[0])
    ys = jnp.linspace(bounds[1][0], bounds[1][1], grid_shape[1])
    zs = jnp.linspace(bounds[2][0], bounds[2][1], grid_shape[2])
    XX, YY, ZZ = jnp.meshgrid(xs, ys, zs, indexing="ij")
    TT = jnp.full_like(XX, t)
    coords_stack = jnp.stack([TT, XX, YY, ZZ], axis=-1).reshape(-1, 4)

    def _per_point_margin(coords):
        cc = compute_curvature_chain(metric, coords)
        # Replace NaNs in T_ab for optimizer robustness
        T = jnp.where(jnp.isnan(cc.stress_energy), 0.0, cc.stress_energy)
        g = cc.metric
        res = optimizer(T, g, n_starts=n_starts)
        return res.margin

    margins = jax.vmap(_per_point_margin)(coords_stack)
    # Finite-filter: take min over finite margins only (defensive against
    # NaN at degenerate probe points).
    finite_mask = jnp.isfinite(margins)
    safe_margins = jnp.where(finite_mask, margins, jnp.inf)
    return jnp.min(safe_margins)


# ---------------------------------------------------------------------------
# Averaged / quantum composition
# ---------------------------------------------------------------------------


def averaged_objective(
    metric: MetricSpecification,
    geodesic,
    kind: str = "anec",
    **kwargs,
) -> Float[Array, ""]:
    """Compose ANEC / AWEC line integral as a DSGN objective.

    Parameters
    ----------
    metric
        The spacetime metric.
    geodesic
        Either a :class:`warpax.geodesics.GeodesicResult` or a callable
        ``geodesic(affine) -> coords`` (see API).
    kind
        ``'anec'`` | ``'awec'``.
    **kwargs
        Forwarded to :func:`warpax.averaged.anec` /
        :func:`warpax.averaged.awec` (e.g. ``affine_bounds``,
        ``n_samples``, ``tangent_norm``).

    Returns
    -------
    Float[Array, ""]
        The ``line_integral`` field of the underlying result.
    """
    from ..averaged import anec, awec

    if kind == "anec":
        return anec(metric, geodesic, **kwargs).line_integral
    elif kind == "awec":
        return awec(metric, geodesic, **kwargs).line_integral
    else:
        raise ValueError(
            f"averaged_objective: kind must be 'anec' or 'awec', got {kind!r}"
        )


def quantum_objective(
    metric: MetricSpecification,
    worldline,
    tau0: float = 1.0,
    **kwargs,
) -> Float[Array, ""]:
    """Compose Ford-Roman quantum inequality as a DSGN objective.

    Parameters
    ----------
    metric
        The spacetime metric.
    worldline
        Callable ``worldline(tau) -> coords`` (timelike worldline
        parametrized by proper time).
    tau0
        Characteristic sampling width.
    **kwargs
        Forwarded to :func:`warpax.quantum.ford_roman` (e.g.
        ``sampling``, ``n_samples``).

    Returns
    -------
    Float[Array, ""]
        The ``margin`` field of the underlying QIResult (positive =>
        QI satisfied).
    """
    from ..quantum import ford_roman

    return ford_roman(metric, worldline, tau0=tau0, **kwargs).margin


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


OBJECTIVE_REGISTRY: dict[str, Callable] = {
    "nec": partial(ec_margin_objective, objective="nec"),
    "wec": partial(ec_margin_objective, objective="wec"),
    "sec": partial(ec_margin_objective, objective="sec"),
    "dec": partial(ec_margin_objective, objective="dec"),
    "anec": partial(averaged_objective, kind="anec"),
    "awec": partial(averaged_objective, kind="awec"),
    "ford_roman": quantum_objective,
}
