"""Input vs. derived stress-energy comparison.

The "source-consistency" diagnostic compares the stress-energy tensor
assumed by the metric constructor (T_input, if provided) against the
Einstein-derived tensor T_derived = G/8pi. The residual:

    DeltaT_{ab} = T_input_{ab} - (1/8pi) G_{ab}

should be identically zero for a self-consistent solution to Einstein's
field equations. A nonzero DeltaT indicates that the metric is NOT a
solution for the claimed source.

This is the central diagnostic from Barzegar-Buchert-Vigneron (2026):
many warp metrics are presented with claimed energy conditions based
on T = G/8pi, but when an explicit source is postulated, the actual
T_input may differ.
"""
from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry.geometry import compute_curvature_chain


def stress_energy_residual(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "4"],
    T_input: Float[Array, "4 4"] | None = None,
) -> dict[str, Float[Array, "..."]]:
    """Compute DeltaT = T_input - G/(8pi) at a single spacetime point.

    Parameters
    ----------
    metric_fn : callable mapping (4,) -> (4,4)
    coords : spacetime coordinates
    T_input : input stress-energy tensor (if None, defaults to vacuum T=0)

    Returns
    -------
    dict with keys:
        'delta_T' : DeltaT_{ab} residual (4,4)
        'T_derived' : G/(8pi) from the metric
        'T_input' : the input T_{ab}
        'max_residual' : max|DeltaT_{ab}|
        'relative_residual' : max|DeltaT| / (max|T_derived| + eps)
    """
    curv = compute_curvature_chain(metric_fn, coords)
    T_derived = curv.stress_energy  # G/(8pi), already symmetrized

    if T_input is None:
        T_input = jnp.zeros((4, 4), dtype=T_derived.dtype)

    delta_T = T_input - T_derived
    max_residual = jnp.max(jnp.abs(delta_T))
    T_scale = jnp.max(jnp.abs(T_derived)) + 1e-30
    relative_residual = max_residual / T_scale

    return {
        "delta_T": delta_T,
        "T_derived": T_derived,
        "T_input": T_input,
        "max_residual": max_residual,
        "relative_residual": relative_residual,
    }
