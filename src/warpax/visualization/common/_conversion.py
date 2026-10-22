"""JAX-to-NumPy conversion: freeze curvature and EC results into FrameData."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from warpax.geometry.grid import GridCurvatureResult
from warpax.geometry.types import GridSpec

from ._frame_data import FrameData


def eulerian_energy_density_grid(
    stress_energy: jax.Array,
    metric_inv: jax.Array,
) -> np.ndarray:
    """Eulerian energy density rho = T_{ab} n^a n^b on a 3D grid.

    The future-directed Eulerian observer is the unit normal to constant-t
    slices: ``n^a = (1/alpha, -beta^i/alpha)`` with
    ``alpha = 1/sqrt(-g^{00})``. For zero shift (Schwarzschild, Minkowski)
    this reduces to ``T_{00}/alpha**2``; for warp metrics with non-trivial beta
    (Alcubierre, Natario, ...) the proper density differs from the bare
    covariant component.
    """
    from warpax.energy_conditions.observer import eulerian_normal

    _, n_up = eulerian_normal(metric_inv)
    rho = jnp.einsum("...a,...ab,...b->...", n_up, stress_energy, n_up)
    return np.asarray(rho)


def _symmetric_clim(arr: np.ndarray) -> tuple[float, float]:
    """Symmetric color limits centered at 0 for diverging colormaps.

    Returns ``(-max_abs, max_abs)`` where ``max_abs = max(|min|, |max|)``.
    Falls back to ``(-1.0, 1.0)`` if data is near-zero.
    """
    max_abs = max(abs(float(np.nanmin(arr))), abs(float(np.nanmax(arr))))
    if max_abs < 1e-15:
        max_abs = 1.0
    return (-max_abs, max_abs)


def _magnitude_clim(arr: np.ndarray) -> tuple[float, float]:
    """Color limits for magnitude (sequential) colormaps."""
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1.0
    return (vmin, vmax)


def _oneside_neg_clim(arr: np.ndarray) -> tuple[float, float]:
    """One-sided color limits ``(vmin, 0)`` for a strictly-non-positive field.

    Correct for fields like the Eulerian energy density and NEC/WEC margins,
    which are ``<= 0`` everywhere for the Alcubierre bubble (0 in flat regions,
    negative in the wall): a diverging ``+/-`` scale would imply a positive
    "satisfied" half that the data never reaches.
    """
    finite = arr[np.isfinite(arr)]
    vmin = float(np.nanmin(finite)) if finite.size else -1.0
    if vmin >= 0.0:
        vmin = -1.0
    return (vmin, 0.0)


def _worst_boost_direction_grid(
    stress_energy: jax.Array,
    metric: jax.Array,
    metric_inv: jax.Array | None = None,
) -> np.ndarray:
    """Unit spatial worst-WEC-boost direction ``e_{i*}`` per grid point ``(...,3)``.

    The worst timelike boost is along the principal eigenvector of the
    most-violating principal pressure (closed form, no rapidity cap). Computed at
    frame-build time via a single batched eigendecomposition; non-Type-I points
    fall back to a (masked-away) approximate direction.
    """
    from warpax.energy_conditions.worst_observer_analytic import worst_observer_typeI

    grid_shape = stress_energy.shape[:-2]
    flat_T = jnp.reshape(stress_energy, (-1, 4, 4))
    flat_g = jnp.reshape(metric, (-1, 4, 4))
    if metric_inv is None:
        flat_ginv = jax.vmap(jnp.linalg.inv)(flat_g)
    else:
        flat_ginv = jnp.reshape(metric_inv, (-1, 4, 4))
    flat_Tmixed = jnp.einsum("nac,ncb->nab", flat_ginv, flat_T)

    def _dir(T_mixed: jax.Array, g_ab: jax.Array) -> jax.Array:
        evals, evecs = jnp.linalg.eig(T_mixed)
        wo = worst_observer_typeI(evals.real, evecs.real, g_ab, condition="wec")
        return wo["boost_direction"]

    dirs = jax.vmap(_dir)(flat_Tmixed, flat_g)
    return np.asarray(dirs)[:, 1:].reshape(*grid_shape, 3)


def eulerian_wec_fields(
    stress_energy: jax.Array,
    metric: jax.Array,
    metric_inv: jax.Array | None = None,
    *,
    atol: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Bounded, cap-free WEC diagnostics on a grid (no rapidity cutoff).

    The naive "worst-case WEC margin over ``zeta <= zeta_max``" is misleading:
    wherever the NEC is violated, ``min_u T_{ab} u^a u^b -> -inf`` as the boost
    rapidity grows, so a finite-cap value encodes only the arbitrary cutoff. This
    returns the physically well-posed alternatives instead:

    - ``wec_margin_eulerian`` : invariant Type-I WEC slack ``min(rho, rho+p_i)``,
      the bounded rest-frame margin (``<= 0`` where violated; ``NaN`` for
      non-Type-I matter, which has no rest frame).
    - ``zeta_th`` : threshold rapidity, the smallest boost at which *some*
      observer first measures negative energy. Closed form
      ``sinh^2(zeta_th) = rho / |rho + p_*|`` for ``rho > 0``; ``0`` where the
      rest frame already violates; ``+inf`` where WEC holds for all boosts.
    - ``boost_dir`` : ``(..., 3)`` unit spatial worst-boost direction ``e_{i*}``.
    - ``he_type`` : Hawking-Ellis algebraic type (1-4).
    """
    from warpax.energy_conditions.frame_free import certify_grid_frame_free

    ff = certify_grid_frame_free(stress_energy, metric, metric_inv)
    rho = np.asarray(ff.rho)
    rho_plus_p = np.asarray(ff.nec_margins)
    wec = np.asarray(ff.wec_margins)
    he = np.asarray(ff.he_types)

    violated = rho_plus_p < -atol
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = rho / np.maximum(-rho_plus_p, atol)
        zeta_th = np.where(
            violated & (rho > 0.0),
            np.arcsinh(np.sqrt(np.maximum(ratio, 0.0))),
            np.where((rho < -atol) | violated, 0.0, np.inf),
        )
    zeta_th = np.where(np.isfinite(rho), zeta_th, np.nan)

    boost_dir = _worst_boost_direction_grid(stress_energy, metric, metric_inv)

    return {
        "wec_margin_eulerian": wec,
        "zeta_th": zeta_th,
        "boost_dir": boost_dir,
        "he_type": he,
    }


def _extract_coordinates(grid_spec: GridSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract spatial coordinates from GridSpec as NumPy arrays."""
    X, Y, Z = grid_spec.meshgrid
    return np.asarray(X), np.asarray(Y), np.asarray(Z)


_CURVATURE_COLORMAPS: dict[str, str] = {
    "ricci_scalar": "RdBu_r",
    "kretschmann": "RdBu_r",
    "ricci_squared": "RdBu_r",
    "weyl_squared": "RdBu_r",
    "energy_density": "RdBu_r",
    "T_00_covariant": "RdBu_r",
}
"""Colormaps per curvature field: ``energy_density`` is the proper Eulerian density
T_{ab} n^a n^b, ``T_00_covariant`` the bare covariant component."""

DIVERGING_FIELDS: frozenset[str] = frozenset(
    {
        *_CURVATURE_COLORMAPS,
        "theta",
        "wec_margin_sweep",
        "nec_margin_sweep",
    }
)
"""Signed fields that get symmetric, zero-centered color limits."""


def freeze_curvature(
    grid_result: GridCurvatureResult,
    grid_spec: GridSpec,
    *,
    metric_name: str,
    v_s: float,
    t: float = 0.0,
) -> FrameData:
    """Freeze a GridCurvatureResult into a FrameData snapshot.

    Converts all JAX arrays to NumPy and computes rendering hints.

    Parameters
    ----------
    grid_result : GridCurvatureResult
        Output from ``evaluate_curvature_grid``.
    grid_spec : GridSpec
        Grid specification used for the computation.
    metric_name : str
        Warp metric identifier (e.g. ``"alcubierre"``).
    v_s : float
        Warp bubble velocity parameter.
    t : float, optional
        Time coordinate of the evaluation (default 0.0).

    Returns
    -------
    FrameData
        Frozen snapshot with curvature invariants as scalar fields.
    """
    x, y, z = _extract_coordinates(grid_spec)

    rho_eul = eulerian_energy_density_grid(grid_result.stress_energy, grid_result.metric_inv)
    scalar_fields: dict[str, np.ndarray] = {
        "ricci_scalar": np.asarray(grid_result.ricci_scalar),
        "kretschmann": np.asarray(grid_result.kretschmann),
        "ricci_squared": np.asarray(grid_result.ricci_squared),
        "weyl_squared": np.asarray(grid_result.weyl_squared),
        "energy_density": rho_eul,
        "T_00_covariant": np.asarray(grid_result.stress_energy[..., 0, 0]),
    }

    colormaps = dict(_CURVATURE_COLORMAPS)
    clim = {
        name: _symmetric_clim(arr) if name in DIVERGING_FIELDS else _magnitude_clim(arr)
        for name, arr in scalar_fields.items()
    }

    return FrameData(
        x=x,
        y=y,
        z=z,
        scalar_fields=scalar_fields,
        metric_name=metric_name,
        v_s=v_s,
        grid_shape=grid_spec.shape,
        t=t,
        colormaps=colormaps,
        clim=clim,
    )
