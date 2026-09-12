"""Evaluate cap-free energy-condition margins and type fractions on a grid.

``certify(metric)`` computes stress-energy, Hawking-Ellis classifications, and
energy-condition margins. Well-conditioned Type-I points use eigenvalue
inequalities; other points use the numerical LMI route in
:mod:`.energy_conditions.frame_free`. The grid summaries include wall-restricted
volume fractions and minimum Type-I slacks, subject to numerical tolerances.
They do not certify unsampled spatial points.

The API also computes Eulerian single-frame miss rates when ``v_s < 1``.
This gate is a comparison policy. For a valid lapse and positive-definite
spatial metric, the Eulerian normal remains timelike at every warp speed.

Example
-------
>>> from warpax import certify
>>> from warpax.metrics import RodalMetric
>>> result = certify(RodalMetric(v_s=2.0, R=1.0, sigma=8.0))
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .analysis.invariant_verification import single_frame_miss
from .energy_conditions.filtering import shape_function_mask
from .energy_conditions.frame_free import (
    FrameFreeGridResult,
    certify_grid_frame_free,
    type_fractions,
    typeI_min_margins,
)
from .geometry import GridSpec, evaluate_curvature_grid
from .geometry.grid import build_coord_batch
from .grids import proper_volume_weights, wall_clustered


class CertifyResult(NamedTuple):
    """Grid energy-condition margins and wall-restricted summary for one metric."""

    v_s: float
    frame_free: FrameFreeGridResult
    type_fractions: dict  # wall-restricted, volume-weighted
    invariant_nec_min: float  # min(rho+p_i) over wall Type-I points
    invariant_dec_min: float  # min(rho-|p_i|) over wall Type-I points
    eulerian_available: bool  # v_s < 1 enables the API's single-frame comparison
    single_frame_miss: dict | None  # per-condition Eulerian miss rates (or None)


def certify(
    metric,
    *,
    v_s: float | None = None,
    bounds=None,
    shape: tuple[int, int, int] = (50, 50, 50),
    clustered: bool = True,
    solver: str = "auto",
    batch_size: int = 256,
    wall_bounds: tuple[float, float] = (0.1, 0.9),
) -> CertifyResult:
    """Evaluate the energy-condition structure of ``metric`` on a spatial grid.

    Parameters
    ----------
    metric : MetricSpecification
        Warp metric with a ``v_s`` parameter.
    v_s : float or None
        If given, rebuild the metric at this speed via ``eqx.tree_at``.
        Otherwise use the metric's speed. Values ``v_s >= 1`` are supported.
    bounds : list[(float, float)] or None
        Spatial box; defaults to ``[(-3, 3)] * 3``.
    shape : tuple[int, int, int]
        Grid resolution.
    clustered : bool
        Use a wall-clustered grid if true; otherwise use a uniform grid.
    solver : {"auto", "standard", "generalized"}
        Eigenvalue backend for classification.
    batch_size : int
        Curvature evaluation batch size.
    wall_bounds : (float, float)
        ``(f_low, f_high)`` selecting the wall for summary statistics.

    Returns
    -------
    CertifyResult
        Numerical grid margins and summaries. Eulerian miss rates are
        computed only when ``v_s < 1`` under the API's comparison policy.
    """
    if v_s is not None:
        metric = eqx.tree_at(lambda m: m.v_s, metric, v_s)
    v = float(metric.v_s)
    if bounds is None:
        bounds = [(-3, 3)] * 3

    grid = (
        wall_clustered(metric, bounds, shape, a=1.2)
        if clustered
        else GridSpec(bounds=bounds, shape=shape)
    )
    curv = evaluate_curvature_grid(metric, grid, batch_size=batch_size)
    T, g, gi = curv.stress_energy, curv.metric, curv.metric_inv

    ff = certify_grid_frame_free(T, g, gi, solver=solver)

    coords = build_coord_batch(grid, t=0.0)
    mask = shape_function_mask(metric, coords, shape, f_low=wall_bounds[0], f_high=wall_bounds[1])
    mask_flat = np.asarray(jnp.reshape(mask, (-1,))).astype(bool)
    vol_w = (
        None
        if grid.volume_weights_array is None
        else proper_volume_weights(grid.volume_weights_array, g)
    )

    fr = type_fractions(ff, mask=mask, volume_weights=vol_w)
    mm = typeI_min_margins(ff, mask=mask_flat)

    eulerian_available = v < 1.0
    miss = None
    if eulerian_available:
        vw_flat = None if vol_w is None else np.asarray(jnp.reshape(vol_w, (-1,)))
        miss = single_frame_miss(T, g, gi, mask=mask_flat, volume_weights=vw_flat)

    return CertifyResult(
        v_s=v,
        frame_free=ff,
        type_fractions=fr,
        invariant_nec_min=mm["nec_min"],
        invariant_dec_min=mm["dec_min"],
        eulerian_available=eulerian_available,
        single_frame_miss=miss,
    )
