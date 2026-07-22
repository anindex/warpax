"""Wall-clustered radial grid generator.

Cosh stretching: per-axis uniform parameter ``u \\in [0, 1]`` is mapped to a
stretched coordinate by a ``sinh`` slice normalized to ``[0, 1]``,

.. math::
    x(u) \\;=\\; lo \\,+\\, (hi - lo) \\cdot
        \\frac{\\sinh(a(u - u_{\\mathrm{wall}})) - \\sinh(-a\\,u_{\\mathrm{wall}})}
             {\\sinh(a(1 - u_{\\mathrm{wall}})) - \\sinh(-a\\,u_{\\mathrm{wall}})},

whose derivative ``\\propto \\cosh(a(u-u_{\\mathrm{wall}}))`` is *minimal* at the
wall parameter and grows toward the tails, so the physical spacing is smallest
(densest sampling) at the wall. ``a`` controls clustering strength (larger =
tighter); ``u_wall \\in [0, 1]`` is the uniform-parameter location of the wall
radius along that axis. (A prior revision used a ``tanh`` slice, whose slope is
maximal at the wall, so it anti-clustered; ``sinh`` is the intended map.)

The returned :class:`GridSpec` carries ``coord_arrays`` + ``volume_weights``
as hashable tuples (static eqx fields), so JIT cache keys stay stable.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from warpax.geometry import GridSpec
from warpax.geometry.metric import MetricSpecification

from ._volume_weights import compute_volume_weights

__all__ = ["wall_clustered"]


def _cosh_stretch(
    u: Float[Array, "N"], u_wall: float, a: float
) -> Float[Array, "N"]:
    """Map uniform ``u \\in [0, 1]`` to stretched ``[0, 1]`` clustered at ``u_wall``.

    Uses a ``sinh`` slice normalized into ``[0, 1]``: the coordinate map's
    derivative is ``\\propto \\cosh(a(u - u_{wall}))``, which is *minimal* at
    ``u = u_wall`` and grows away from it. Since nodes are uniform in ``u``,
    the resulting physical spacing ``\\Delta x`` is *smallest* at the wall
    (dense clustering) and larger in the tails (sparse). Larger ``a`` means
    tighter clustering. Endpoints ``u=0`` and ``u=1`` map exactly to ``0`` and
    ``1`` for any ``u_wall in (0, 1)`` and ``a > 0``.

    (An earlier revision applied ``tanh`` here, whose slope is *maximal* at the
    wall, so it anti-clustered, sampling the wall *worse* than a uniform grid and
    getting worse as ``a`` grew. The ``sinh`` map is the intended "cosh
    stretching": slope ``= \\cosh``, densest at the wall.)
    """
    s_u = jnp.sinh(a * (u - u_wall))
    s_0 = jnp.sinh(a * (0.0 - u_wall))
    s_1 = jnp.sinh(a * (1.0 - u_wall))
    return (s_u - s_0) / (s_1 - s_0)


def _infer_wall_radius(
    metric: MetricSpecification, bounds: tuple[tuple[float, float], ...]
) -> float:
    """Best-effort wall-radius inference via ``shape_function_value`` scan.

    Samples ``metric.shape_function_value`` along the positive x-axis and
    returns the radius at which the absolute derivative is largest. Falls
    back to the midpoint of the first spatial axis plus a unit offset (1.0)
    when the metric has no
    shape function (e.g., :class:`warpax.io.InterpolatedADMMetric` raises
    :class:`NotImplementedError`).
    """
    r_lo = max(0.01, 0.01 * bounds[0][1])
    r_hi = 0.95 * bounds[0][1]
    r_samples = jnp.linspace(r_lo, r_hi, 101)

    def _f_at_r(r: Float[Array, ""]) -> Float[Array, ""]:
        coords = jnp.array([0.0, r, 0.0, 0.0])
        return metric.shape_function_value(coords)

    try:
        f_values = jax.vmap(_f_at_r)(r_samples)
    except NotImplementedError:
        return 0.5 * (bounds[0][0] + bounds[0][1]) + 1.0

    df = jnp.abs(jnp.diff(f_values))
    wall_idx = int(jnp.argmax(df))
    return float(r_samples[wall_idx])


def wall_clustered(
    metric: MetricSpecification,
    bounds: tuple[tuple[float, float], ...],
    shape: tuple[int, ...],
    clustering: str = "cosh",
    wall_radius: float | None = None,
    a: float = 1.2,
) -> GridSpec:
    """Build a radially wall-clustered :class:`GridSpec` for a warp-drive metric.

    Parameters
    ----------
    metric : MetricSpecification
        Source metric (used to infer wall radius if not supplied).
    bounds : tuple of (lo, hi) pairs
        Axis-aligned bounds (typically 3D spatial).
    shape : tuple of int
        Grid resolution per axis.
    clustering : {"cosh"}, default "cosh"
        Only ``"cosh"`` is currently supported.
    wall_radius : float | None, default None
        Wall radius in physical units. If ``None``, inferred via a
        ``metric.shape_function_value`` scan.
    a : float, default 1.2
        Cosh clustering strength (see module docstring).

    Returns
    -------
    GridSpec
        Static :class:`GridSpec` carrying ``coord_arrays`` (non-uniform 1D
        coords per axis, as a tuple of tuples) and ``volume_weights``
        (flattened per-cell weights; reshape via
        :attr:`GridSpec.volume_weights_array`).

    Raises
    ------
    ValueError
        If ``clustering`` is not ``"cosh"`` or if ``bounds`` and ``shape``
        have different lengths.
    """
    if clustering != "cosh":
        raise ValueError(
            f"Only clustering='cosh' is supported; got {clustering!r}"
        )
    if len(bounds) != len(shape):
        raise ValueError(
            f"bounds and shape must have the same length; got "
            f"{len(bounds)} vs {len(shape)}"
        )

    if wall_radius is None:
        wall_radius = _infer_wall_radius(metric, bounds)

    coord_arrays = []
    for axis_bounds, n in zip(bounds, shape):
        lo, hi = axis_bounds
        u_wall = (wall_radius - lo) / (hi - lo)
        u_wall = float(jnp.clip(u_wall, 0.05, 0.95))
        u = jnp.linspace(0.0, 1.0, n)
        stretched = _cosh_stretch(u, u_wall, a)
        coords_jnp = lo + (hi - lo) * stretched
        coord_arrays.append(tuple(float(x) for x in coords_jnp))
    coord_arrays_tuple = tuple(coord_arrays)

    volume_weights_tuple: tuple | None
    if len(bounds) == 3:
        vw_array = compute_volume_weights(
            jnp.asarray(coord_arrays[0]),
            jnp.asarray(coord_arrays[1]),
            jnp.asarray(coord_arrays[2]),
        )
        volume_weights_tuple = tuple(float(x) for x in vw_array.flatten())
    else:
        volume_weights_tuple = None

    return GridSpec(
        bounds=list(bounds),
        shape=shape,
        coord_arrays=coord_arrays_tuple,
        volume_weights=volume_weights_tuple,
    )
