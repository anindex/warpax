"""One-call, all-observer, all-velocity energy-condition certifier.

``certify(metric)`` is the public entry point intended for the GR community: an
independent, reproducible verifier that recomputes the all-observer energy-
condition truth of any warp-drive metric from the eigenstructure of ``T^a_b`` --
at any warp speed, including ``v_s >= 1``.

It wraps the certification engine:

- frame-independent Hawking-Ellis classification + Type-I eigenvalue margins
  (:func:`.energy_conditions.frame_free.certify_grid_frame_free`), valid at all
  velocities because no Eulerian normal is constructed;
- the volume-weighted Type census (where Type IV = no rest frame);
- when ``v_s < 1`` (the Eulerian congruence is timelike), the single-frame miss
  verification against the Eulerian frame
  (:func:`.analysis.invariant_verification.single_frame_miss`).

Example
-------
>>> from warpax import certify
>>> from warpax.metrics import RodalMetric
>>> r = certify(RodalMetric(v_s=2.0, R=1.0, sigma=8.0))   # superluminal
>>> r.type_fractions["frac_type_i"], r.invariant_nec_min
(1.0, -2.75...)
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
from .grids import wall_clustered


class CertifyResult(NamedTuple):
    """All-observer, all-velocity certification summary for one metric."""

    v_s: float
    frame_free: FrameFreeGridResult
    type_fractions: dict        # wall-restricted, volume-weighted
    invariant_nec_min: float    # min(rho+p_i) over wall Type-I points
    invariant_dec_min: float    # min(rho-|p_i|) over wall Type-I points
    eulerian_available: bool    # True iff v_s < 1 (Eulerian congruence timelike)
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
    """Certify the all-observer energy-condition structure of ``metric``.

    Parameters
    ----------
    metric : MetricSpecification
        Any warp-drive metric (e.g. from :mod:`warpax.metrics`).
    v_s : float or None
        If given, the metric is rebuilt at this warp speed (via ``eqx.tree_at``);
        otherwise the metric's own ``v_s`` is used. Works for ``v_s >= 1``.
    bounds : list[(float, float)] or None
        Spatial box; defaults to ``[(-3, 3)] * 3`` (matched compact domain).
    shape : tuple[int, int, int]
        Grid resolution.
    clustered : bool
        Use a wall-clustered grid (recommended; resolves the wall) vs uniform.
    solver : {"auto", "standard", "generalized"}
        Eigenvalue backend for classification.
    batch_size : int
        Curvature evaluation batch size.
    wall_bounds : (float, float)
        ``(f_low, f_high)`` defining the active wall region for the
        wall-restricted statistics.

    Returns
    -------
    CertifyResult
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
    mask = shape_function_mask(metric, coords, shape,
                               f_low=wall_bounds[0], f_high=wall_bounds[1])
    mask_flat = np.asarray(jnp.reshape(mask, (-1,))).astype(bool)
    vol_w = grid.volume_weights_array

    fr = type_fractions(ff, mask=mask, volume_weights=vol_w)
    mm = typeI_min_margins(ff, mask=mask_flat)

    eulerian_available = v < 1.0
    miss = None
    if eulerian_available:
        vw_flat = (None if vol_w is None
                   else np.asarray(jnp.reshape(vol_w, (-1,))))
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
