"""Wall-clustered radial grid generator.

Anchored two-sided ``sinh`` stretching: the per-axis uniform parameter
``u \\in [0, 1]`` is mapped to a stretched coordinate that fixes the wall
parameter exactly (see :func:`_cosh_stretch`), so the physical spacing is
smallest *at the wall* and grows symmetrically toward both tails. ``a``
controls clustering strength (larger = tighter); ``u_wall \\in [0, 1]`` is the
uniform-parameter location of the wall radius along that axis.

The returned :class:`GridSpec` carries ``coord_arrays`` + ``volume_weights``
as hashable tuples (static eqx fields), so JIT cache keys stay stable.

Scope. The map is applied to each Cartesian axis independently, so it refines
the three coordinate *slabs* ``|x| \\approx R``, ``|y| \\approx R``,
``|z| \\approx R`` rather than the sphere ``r = R``. Along any axis crossing
(the wall normal that the resolution criterion measures) this is exact; for a
genuinely spherical refinement use a radial/angular grid instead.
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
    """Map uniform ``u \\in [0, 1]`` to stretched ``[0, 1]`` anchored at ``u_wall``.

    Two-sided ``sinh`` map, each side normalized independently so that the wall
    parameter is a *fixed point*:

    .. math::
        X(u) = \\begin{cases}
          q - q\\,\\sinh\\!\\big(a(q-u)/q\\big)/\\sinh a, & u \\le q,\\\\
          q + (1-q)\\,\\sinh\\!\\big(a(u-q)/(1-q)\\big)/\\sinh a, & u \\ge q,
        \\end{cases}

    with ``q = u_wall``. Then ``X(0)=0``, ``X(q)=q`` and ``X(1)=1`` exactly, and
    ``X'`` is ``\\propto \\cosh`` of a quantity vanishing at ``u=q`` on both
    sides, with the matching one-sided slope ``a/\\sinh a``. So the map is
    :math:`C^1`, the physical spacing is *smallest exactly at the wall*, and the
    two sides are refined symmetrically.

    History. An early revision applied ``tanh``, whose slope is maximal at the
    wall, so it anti-clustered. The replacement used a single ``sinh`` slice
    ``(\\sinh(a(u-u_w)) - \\sinh(-a u_w)) / (\\sinh(a(1-u_w)) - \\sinh(-a u_w))``,
    which clusters correctly but is *not anchored*: it does not fix ``u_wall``,
    so the densest sampling drifts off the wall. For ``bounds=(-3,3)``,
    ``wall_radius=1`` it landed at ``x = 1.266`` (a 27% offset) and for
    ``(-300,300)``, ``R=100`` at ``x = 126.6``. The two-sided form above fixes
    the wall by construction; see ``tests/test_grids_clustered.py``.
    """
    q = u_wall
    sa = jnp.sinh(a)
    lower = q - q * jnp.sinh(a * (q - u) / q) / sa
    upper = q + (1.0 - q) * jnp.sinh(a * (u - q) / (1.0 - q)) / sa
    return jnp.where(u <= q, lower, upper)


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

    This scan starts at the coordinate ORIGIN, so it assumes the bubble is centred
    there. Every construction that reaches this function is, Alcubierre, Natario,
    Van den Broeck, Rodal and the shells all sit at the origin at ``t = 0``, and the
    one that does not, Garattini-Zatrimaylov at ``r_0 = v_s / H``, is certified on the
    axisymmetric ladder with an explicit ``wall_radius`` and ``center`` instead. For an
    off-origin bubble the scan finds only the outer crossing, and anchoring the
    clustering there puts the COARSEST spacing on the inner one, so the refinement
    makes the resolution worse rather than better. Rather than leave that silent, the
    scan now refuses to guess when it cannot see a wall at all.
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

    # If the shape function never crosses the transition band on the scanned axis, the
    # argmax below is the argmax of rounding noise and the returned radius is
    # meaningless, GarattiniMetric(v_s=0.5, H=0.1) on bounds (-3, 3) has its walls at
    # x = 4 and x = 6, entirely outside the window, and this used to return 2.82.
    f_lo, f_hi = float(jnp.min(f_values)), float(jnp.max(f_values))
    if f_hi - f_lo < 0.1:
        raise ValueError(
            f"cannot infer a wall radius: the shape function spans only "
            f"[{f_lo:.3g}, {f_hi:.3g}] on the scanned axis x in "
            f"[{r_lo:.3g}, {r_hi:.3g}], so no transition is visible there. Pass "
            f"wall_radius= explicitly, or widen the bounds. A bubble that is not "
            f"centred on the coordinate origin needs the axisymmetric grid with an "
            f"explicit center= instead."
        )

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
        u = jnp.linspace(0.0, 1.0, n)
        centre = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        if lo < 0.0 < hi and 0.0 < wall_radius < half:
            # The wall is a sphere, so an axis crosses it twice, at +-R.
            # Fold about the centre and anchor on |x| = R so BOTH crossings are
            # refined identically; a one-sided anchor leaves the -x crossing
            # coarse (it gave 3.27 cells against 4.45 at N=80, below the
            # four-cell criterion the paper enforces).
            signed = 2.0 * u - 1.0
            v = jnp.abs(signed)
            q = float(jnp.clip(wall_radius / half, 0.05, 0.95))
            coords_jnp = centre + jnp.sign(signed) * half * _cosh_stretch(v, q, a)
        else:
            # Axis does not straddle the centre (or the wall is outside it):
            # single-sided anchored map.
            q = float(jnp.clip((wall_radius - lo) / (hi - lo), 0.05, 0.95))
            coords_jnp = lo + (hi - lo) * _cosh_stretch(u, q, a)
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
