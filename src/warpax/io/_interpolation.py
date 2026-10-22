"""JIT-safe interpolation helpers for :class:`InterpolatedADMMetric`.

``"linear"`` uses :func:`jax.scipy.ndimage.map_coordinates` with ``order=1``
and ``mode="nearest"``; ``"cubic"`` evaluates a uniform cubic B-spline whose
coefficients come from :func:`_bspline_prefilter`. Both clamp out-of-bounds
coordinates to the grid boundary.

Private module; consumers should use :class:`warpax.io.InterpolatedADMMetric`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates
from jaxtyping import Array, Float, Real

from warpax.geometry import GridSpec

__all__ = [
    "_bspline_eval",
    "_bspline_prefilter",
    "_interpolate_scalar",
    "_interpolate_tensor",
    "_interpolate_vector",
]


def _coords_to_indices(coords: Real[Array, "4"], grid_spec: GridSpec) -> Float[Array, "4"]:
    """Map physical coords ``(t, x, y, z)`` to fractional grid indices; singleton axes map to 0."""
    bounds = jnp.asarray(grid_spec.bounds, dtype=jnp.result_type(coords.dtype, float))
    shape = jnp.asarray(grid_spec.shape)
    low = bounds[:, 0]
    span = jnp.where(shape > 1, bounds[:, 1] - low, 1.0)
    return jnp.where(shape > 1, (coords - low) / span * (shape - 1), 0.0)


def _bspline_prefilter(grid: np.ndarray) -> np.ndarray:
    """Solve for uniform cubic B-spline coefficients along the four leading axes.

    Per axis of length ``N >= 2`` the tridiagonal system with stencil
    ``[1/6, 2/3, 1/6]`` and mirror end rows (the coefficient at ``-1`` is
    reflected onto index ``1``) is solved so the spline reproduces the samples
    at the knots; the mirror rows match the index reflection in
    :func:`_reflect`, so ``N = 2`` and ``N = 3`` are exact too. Singleton axes
    are left untouched.
    """
    coeffs = np.asarray(grid, dtype=np.float64)
    for axis in range(4):
        n = coeffs.shape[axis]
        if n < 2:
            continue
        A = np.diag(np.full(n, 2.0 / 3.0)) + np.diag(np.full(n - 1, 1.0 / 6.0), 1)
        A += np.diag(np.full(n - 1, 1.0 / 6.0), -1)
        A[0, 1] += 1.0 / 6.0
        A[-1, -2] += 1.0 / 6.0
        moved = np.moveaxis(coeffs, axis, 0)
        solved = np.linalg.solve(A, moved.reshape(n, -1)).reshape(moved.shape)
        coeffs = np.moveaxis(solved, 0, axis)
    return coeffs


def _reflect(k: Array, n: int) -> Array:
    """Mirror-reflect integer indices into ``[0, n - 1]`` without repeating the endpoints."""
    if n == 1:
        return jnp.zeros_like(k)
    period = 2 * (n - 1)
    k = jnp.abs(k) % period
    return jnp.where(k > n - 1, period - k, k)


def _bspline_eval(
    coeffs: Float[Array, "C Nt Nx Ny Nz"], idx: Float[Array, "4"]
) -> Float[Array, "C"]:
    """Evaluate the tensor-product cubic B-spline at fractional index ``idx`` for every channel."""
    base = jnp.floor(idx)
    t = idx - base
    i0 = base.astype(jnp.int32) - 1
    offsets = jnp.arange(4, dtype=jnp.int32)
    ks = [_reflect(i0[a] + offsets, coeffs.shape[a + 1]) for a in range(4)]
    w = (
        jnp.stack(
            [
                (1.0 - t) ** 3,
                3.0 * t**3 - 6.0 * t**2 + 4.0,
                -3.0 * t**3 + 3.0 * t**2 + 3.0 * t + 1.0,
                t**3,
            ],
            axis=-1,
        )
        / 6.0
    )
    block = coeffs[
        :,
        ks[0][:, None, None, None],
        ks[1][None, :, None, None],
        ks[2][None, None, :, None],
        ks[3][None, None, None, :],
    ]
    return jnp.einsum("cijkl,i,j,k,l->c", block, w[0], w[1], w[2], w[3])


def _evaluate_channels(
    stacked: Float[Array, "C Nt Nx Ny Nz"],
    coords: Real[Array, "4"],
    grid_spec: GridSpec,
    method: str,
) -> Float[Array, "C"]:
    """Interpolate every leading channel independently at ``coords``."""
    idx = _coords_to_indices(coords, grid_spec)
    if method == "linear":
        return jax.vmap(lambda g: map_coordinates(g, idx[:, None], order=1, mode="nearest")[0])(
            stacked
        )
    if method == "cubic":
        upper = jnp.asarray(grid_spec.shape, dtype=idx.dtype) - 1
        return _bspline_eval(stacked, jnp.clip(idx, 0.0, upper))
    raise ValueError(f"Unknown interp_method: {method!r}; expected 'linear' or 'cubic'.")


def _interpolate_scalar(
    grid: Float[Array, "Nt Nx Ny Nz"],
    coords: Real[Array, "4"],
    grid_spec: GridSpec,
    method: str,
) -> Float[Array, ""]:
    """Interpolate a scalar field at coords. Returns a 0-d array."""
    return _evaluate_channels(grid[None], coords, grid_spec, method)[0]


def _interpolate_vector(
    grid: Float[Array, "Nt Nx Ny Nz 3"],
    coords: Real[Array, "4"],
    grid_spec: GridSpec,
    method: str,
) -> Float[Array, "3"]:
    """Interpolate a 3-vector field, one channel at a time."""
    return _evaluate_channels(jnp.moveaxis(grid, -1, 0), coords, grid_spec, method)


def _interpolate_tensor(
    grid: Float[Array, "Nt Nx Ny Nz 3 3"],
    coords: Real[Array, "4"],
    grid_spec: GridSpec,
    method: str,
) -> Float[Array, "3 3"]:
    """Interpolate a 3x3 tensor field, one of the nine channels at a time."""
    flat = grid.reshape(*grid.shape[:4], 9)
    return _evaluate_channels(jnp.moveaxis(flat, -1, 0), coords, grid_spec, method).reshape(3, 3)
