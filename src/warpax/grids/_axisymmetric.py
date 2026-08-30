"""Exact axisymmetric reduction of a warp-drive slice to a ``(r, mu)`` sweep.

Every drive in this family is invariant under rotations about the propagation
axis, so a scalar built covariantly from the metric depends only on the distance
``r`` from the bubble centre and on ``mu = cos theta``, the cosine of the angle
to that axis. Sampling the half-plane ``phi = 0`` therefore sweeps the whole
orbit space: this is an *exact reduction*, not a slice through a 3-D problem.
The same argument already licenses the interval branch-and-bound in
:mod:`warpax.energy_conditions.enclosure`.

What it buys. A thin wall costs resolution in every dimension it is embedded in.
The Rodal sigma sweep needs four cells across a wall of width 7.32 inside a
600-wide box; on a uniform Cartesian grid that is ``N >= 329``, i.e. 3.6e7
points. In ``(r, mu)`` the same physics is 80x80 = 6400 points. That is the
difference between an intractable re-run and a three-level convergence ladder.

Quadrature. The radial axis carries the wall, so it uses the anchored clustering
map of :mod:`._clustered`; the angular axis is smooth, so it uses Gauss-Legendre
in ``mu``, which is exact for the polynomial part of the angular dependence and
converges spectrally otherwise. The proper-volume element of a flat slice in
these coordinates is

    dV = 2 pi r^2 dr dmu,

so the weights are ``w_ij = 2 pi r_i^2 w^r_i w^mu_j``. The ``2 pi`` cancels in
any volume *fraction* but is kept so that integrated quantities are absolute.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ._clustered import _cosh_stretch

__all__ = ["AxisymmetricGrid", "axisymmetric_grid"]


class AxisymmetricGrid(NamedTuple):
    """An exact ``(r, mu)`` sampling of an axisymmetric slice.

    Attributes
    ----------
    coords : ``(N, 4)`` batch of ``(t, x, y, 0)``, ready for
        :func:`warpax.geometry.evaluate_curvature_points`.
    weights : ``(N,)`` proper-volume weights ``2 pi r^2 w_r w_mu``.
    r, mu : the 1-D nodes, lengths ``n_r`` and ``n_mu``.
    shape : ``(n_r, n_mu)``, for reshaping flat results.
    """

    coords: jnp.ndarray
    weights: np.ndarray
    r: np.ndarray
    mu: np.ndarray
    shape: tuple[int, int]

    def reshape(self, flat):
        """Reshape a flat ``(N, ...)`` result back to ``(n_r, n_mu, ...)``."""
        arr = jnp.asarray(flat)
        return arr.reshape(*self.shape, *arr.shape[1:])


def _radial_nodes(r_max: float, n_r: int, wall_radius: float, a: float) -> np.ndarray:
    """Wall-clustered nodes on ``[0, r_max]``, anchored on ``wall_radius``."""
    if not (0.0 < wall_radius < r_max):
        return np.linspace(0.0, r_max, n_r)
    q = float(np.clip(wall_radius / r_max, 0.05, 0.95))
    u = jnp.linspace(0.0, 1.0, n_r)
    return np.asarray(r_max * _cosh_stretch(u, q, a), dtype=float)


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoidal cell widths for arbitrary (non-uniform) 1-D nodes."""
    w = np.zeros_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def axisymmetric_grid(
    r_max: float,
    n_r: int,
    n_mu: int,
    *,
    wall_radius: float | None = None,
    a: float = 2.0,
    t: float = 0.0,
    center: float = 0.0,
) -> AxisymmetricGrid:
    """Build the exact ``(r, mu)`` sampling of an axisymmetric slice.

    Parameters
    ----------
    r_max : outer radius of the sampled ball.
    n_r, n_mu : radial and angular node counts.
    wall_radius : radius to cluster the radial nodes on. ``None`` gives a
        uniform radial grid.
    a : radial clustering strength (see :mod:`._clustered`).
    t : the constant-time slice to sample.
    center : axial offset of the sampling origin, i.e. the reduction is performed
        about ``(center, 0, 0)`` rather than about the coordinate origin. The
        reduction stays *exact* under this shift as long as the configuration is
        axisymmetric about the same axis, which is what matters for a construction
        whose bubble does not sit at the origin: the Garattini-Zatrimaylov drive
        places its bubble at ``r_0 = v_s / H``, so an origin-centred radial grid
        clusters its nodes on a sphere the wall merely crosses, and the wall spans
        1.5 cells at ``n_r = 32`` where a bubble-centred grid spans 4.5.
    """
    r = _radial_nodes(r_max, n_r, wall_radius or -1.0, a)
    mu, w_mu = np.polynomial.legendre.leggauss(n_mu)
    w_r = _trapezoid_weights(r)

    R, MU = np.meshgrid(r, mu, indexing="ij")
    WR, WMU = np.meshgrid(w_r, w_mu, indexing="ij")

    x = center + R * MU
    y = R * np.sqrt(np.clip(1.0 - MU**2, 0.0, None))
    coords = jnp.stack(
        [
            jnp.full(x.size, t),
            jnp.asarray(x.ravel()),
            jnp.asarray(y.ravel()),
            jnp.zeros(x.size),
        ],
        axis=1,
    )
    weights = (2.0 * np.pi * R**2 * WR * WMU).ravel()
    return AxisymmetricGrid(coords=coords, weights=weights, r=r, mu=mu, shape=(n_r, n_mu))
