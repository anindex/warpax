"""Observer sweep computation for EC margins across families of observers.

Evaluates energy condition margins for a coarse grid of observer 4-velocities,
enabling animation of observer-dependent violations and fast approximate
worst-case detection.

Two observer family generators are provided:
- ``make_rapidity_observers``: rapidity sweep at fixed angular directions
- ``make_angular_observers``: angular sweep at fixed rapidity

Vectorized margin computation uses double vmap: outer over grid points (N),
inner over observer samples (K). The tetrad is computed once per grid point.

Cross-validation against BFGS optimization verifies that the coarse sweep
catches violations wherever the optimizer does.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .observer import (
    compute_orthonormal_tetrad,
    null_from_angles,
    timelike_from_rapidity,
)


def make_rapidity_observers(
    n_rapidity: int = 12,
    n_directions: int = 3,
    zeta_max: float = 5.0,
) -> Float[Array, "K 3"]:
    """Generate observer parameter sets: rapidity sweep at fixed directions.

    Parameters
    ----------
    n_rapidity : int
        Number of rapidity values linearly spaced from 0 to zeta_max.
    n_directions : int
        Number of fixed angular directions. Default 3: along x, y, z axes.
    zeta_max : float
        Maximum rapidity.

    Returns
    -------
    Float[Array, "K 3"]
        Observer parameters (zeta, theta, phi) of shape (K, 3)
        where K = n_rapidity * n_directions.
    """
    zetas = jnp.linspace(0.0, zeta_max, n_rapidity)

    directions = jnp.array(
        [
            [jnp.pi / 2, 0.0],
            [jnp.pi / 2, jnp.pi / 2],
            [0.0, 0.0],
        ]
    )[:n_directions]

    n_dirs = min(n_directions, directions.shape[0])
    zetas_col = jnp.broadcast_to(zetas[:, None, None], (n_rapidity, n_dirs, 1))
    dirs_col = jnp.broadcast_to(directions[None, :n_dirs, :], (n_rapidity, n_dirs, 2))
    return jnp.concatenate([zetas_col, dirs_col], axis=-1).reshape(-1, 3)


def make_angular_observers(
    zeta_fixed: float = 1.0,
    n_theta: int = 6,
    n_phi: int = 6,
) -> Float[Array, "K 3"]:
    """Generate observer parameter sets: angular sweep at fixed rapidity.

    Parameters
    ----------
    zeta_fixed : float
        Fixed rapidity for all observers.
    n_theta : int
        Number of polar angle values in [0, pi].
    n_phi : int
        Number of azimuthal angle values in [0, 2*pi) (endpoint excluded).

    Returns
    -------
    Float[Array, "K 3"]
        Observer parameters (zeta, theta, phi) of shape (K, 3)
        where K = n_theta * n_phi.
    """
    thetas = jnp.linspace(0.0, jnp.pi, n_theta)
    phis = jnp.linspace(0.0, 2 * jnp.pi, n_phi, endpoint=False)

    THETA, PHI = jnp.meshgrid(thetas, phis, indexing="ij")
    zeta_col = jnp.full(THETA.shape, zeta_fixed)
    return jnp.stack([zeta_col.ravel(), THETA.ravel(), PHI.ravel()], axis=-1)


def _point_sweep_wec(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    obs_params: Float[Array, "K 3"],
) -> Float[Array, "K"]:
    """WEC margin for all observers at a single grid point."""
    tetrad = compute_orthonormal_tetrad(g_ab)

    def eval_one(params):
        u = timelike_from_rapidity(params[0], params[1], params[2], tetrad)
        return jnp.einsum("a,ab,b->", u, T_ab, u)

    return jax.vmap(eval_one)(obs_params)


@jax.jit
def sweep_wec_margins(
    T_field: Float[Array, "N 4 4"],
    g_field: Float[Array, "N 4 4"],
    observer_params: Float[Array, "K 3"],
) -> Float[Array, "N K"]:
    """Compute WEC margins for all observers at all grid points.

    Parameters
    ----------
    T_field : Float[Array, "N 4 4"]
        Stress-energy tensor field (flattened grid), N points.
    g_field : Float[Array, "N 4 4"]
        Metric tensor field (flattened grid), N points.
    observer_params : Float[Array, "K 3"]
        Observer parameters (zeta, theta, phi), K observers.

    Returns
    -------
    Float[Array, "N K"]
        WEC margin at each point for each observer.
        Negative values indicate WEC violation.
    """
    return jax.vmap(_point_sweep_wec, in_axes=(0, 0, None))(T_field, g_field, observer_params)


def _point_sweep_nec(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    obs_params: Float[Array, "K 2"],
) -> Float[Array, "K"]:
    """NEC margin for all null observers at a single grid point."""
    tetrad = compute_orthonormal_tetrad(g_ab)

    def eval_one(params):
        k = null_from_angles(params[0], params[1], tetrad)
        return jnp.einsum("a,ab,b->", k, T_ab, k)

    return jax.vmap(eval_one)(obs_params)


@jax.jit
def sweep_nec_margins(
    T_field: Float[Array, "N 4 4"],
    g_field: Float[Array, "N 4 4"],
    observer_params: Float[Array, "K 2"],
) -> Float[Array, "N K"]:
    """Compute NEC margins for all null directions at all grid points.

    Parameters
    ----------
    T_field : Float[Array, "N 4 4"]
        Stress-energy tensor field (flattened grid), N points.
    g_field : Float[Array, "N 4 4"]
        Metric tensor field (flattened grid), N points.
    observer_params : Float[Array, "K 2"]
        Null direction parameters (theta, phi), K directions.

    Returns
    -------
    Float[Array, "N K"]
        NEC margin at each point for each null direction.
    """
    return jax.vmap(_point_sweep_nec, in_axes=(0, 0, None))(T_field, g_field, observer_params)


def _point_sweep_sec(
    sec_tensor: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    obs_params: Float[Array, "K 3"],
) -> Float[Array, "K"]:
    """SEC margin for all observers at a single grid point."""
    tetrad = compute_orthonormal_tetrad(g_ab)

    def eval_one(params):
        u = timelike_from_rapidity(params[0], params[1], params[2], tetrad)
        return jnp.einsum("a,ab,b->", u, sec_tensor, u)

    return jax.vmap(eval_one)(obs_params)


def sweep_all_margins(
    T_field: Float[Array, "N 4 4"],
    g_field: Float[Array, "N 4 4"],
    g_inv_field: Float[Array, "N 4 4"],
    observer_params: Float[Array, "K 3"],
) -> dict[str, Float[Array, "N K"]]:
    """Compute WEC, NEC, SEC margins for all observers at all grid points.

    Parameters
    ----------
    T_field : Float[Array, "N 4 4"]
        Stress-energy tensor field (flattened grid).
    g_field : Float[Array, "N 4 4"]
        Metric tensor field (flattened grid).
    g_inv_field : Float[Array, "N 4 4"]
        Inverse metric tensor field (flattened grid).
    observer_params : Float[Array, "K 3"]
        Observer parameters (zeta, theta, phi).

    Returns
    -------
    dict[str, Float[Array, "N K"]]
        Dictionary with keys ``"wec"``, ``"nec"``, ``"sec"`` mapping to
        (N, K) margin arrays.

    WEC is ``T_ab u^a u^b`` over timelike observers; NEC is ``T_ab k^a k^b`` over
    null directions, using only the ``(theta, phi)`` columns of
    ``observer_params``; SEC is ``(T_ab - T g_ab / 2) u^a u^b`` with
    ``T = g^{ab} T_ab``.
    """
    wec = sweep_wec_margins(T_field, g_field, observer_params)

    nec_params = observer_params[:, 1:]
    nec = sweep_nec_margins(T_field, g_field, nec_params)

    T_trace = jnp.einsum("nab,nab->n", g_inv_field, T_field)
    sec_tensor = T_field - 0.5 * T_trace[:, None, None] * g_field
    sec = jax.vmap(_point_sweep_sec, in_axes=(0, 0, None))(sec_tensor, g_field, observer_params)

    return {"wec": wec, "nec": nec, "sec": sec}


def cross_validate_sweep(
    T_field: Float[Array, "N 4 4"],
    g_field: Float[Array, "N 4 4"],
    observer_params: Float[Array, "K 3"],
    *,
    n_validation_points: int = 50,
    n_starts: int = 16,
    zeta_max: float = 5.0,
    rtol: float = 0.1,
    key=None,
) -> dict:
    """Cross-validate sweep WEC margins against BFGS optimization.

    Selects random grid points, compares sweep worst-case margin with
    BFGS optimizer result at each point.

    Parameters
    ----------
    T_field : Float[Array, "N 4 4"]
        Stress-energy tensor field (flattened grid).
    g_field : Float[Array, "N 4 4"]
        Metric tensor field (flattened grid).
    observer_params : Float[Array, "K 3"]
        Observer parameters for sweep.
    n_validation_points : int
        Number of random grid points to validate.
    n_starts : int
        Multi-start count for BFGS optimizer.
    zeta_max : float
        Rapidity cap for BFGS optimizer.
    rtol : float
        Relative tolerance for "matching" (default 0.1 = 10%).
    key : PRNGKey or None
        Random key for point selection.

    Returns
    -------
    dict
        Dictionary with:
        - ``max_relative_error``: float
        - ``mean_relative_error``: float
        - ``sign_agreement_fraction``: float
        - ``sweep_worse_count``: int

    Sign agreement counts points where both margins share a sign or both are
    within ``1e-12`` of zero; ``sweep_worse_count`` counts points where both are
    negative and the sweep is shallower; the relative error uses BFGS as reference
    and is absolute when ``|bfgs| <= 1e-12``.
    """
    from .optimization import optimize_wec

    if key is None:
        key = jax.random.PRNGKey(42)

    N = T_field.shape[0]
    n_validation_points = min(n_validation_points, N)

    key, subkey = jax.random.split(key)
    indices = jax.random.choice(subkey, N, shape=(n_validation_points,), replace=False)

    sweep_margins = sweep_wec_margins(T_field, g_field, observer_params)
    sweep_mins = jnp.min(sweep_margins, axis=-1)

    sign_agree = 0
    sweep_worse = 0
    rel_errors = []

    for i in range(n_validation_points):
        idx = int(indices[i])
        T_point = T_field[idx]
        g_point = g_field[idx]

        key, subkey = jax.random.split(key)
        bfgs_result = optimize_wec(
            T_point,
            g_point,
            n_starts=n_starts,
            zeta_max=zeta_max,
            key=subkey,
        )
        bfgs_min = float(bfgs_result.margin)
        sweep_min = float(sweep_mins[idx])

        if (sweep_min < 0) == (bfgs_min < 0) or (abs(sweep_min) < 1e-12 and abs(bfgs_min) < 1e-12):
            sign_agree += 1

        if bfgs_min < 0 and sweep_min > bfgs_min:
            sweep_worse += 1

        if abs(bfgs_min) > 1e-12:
            rel_err = abs(sweep_min - bfgs_min) / abs(bfgs_min)
        else:
            rel_err = abs(sweep_min - bfgs_min)
        rel_errors.append(rel_err)

    import numpy as np

    rel_errors = np.array(rel_errors)

    return {
        "max_relative_error": float(np.max(rel_errors)),
        "mean_relative_error": float(np.mean(rel_errors)),
        "sign_agreement_fraction": sign_agree / n_validation_points,
        "sweep_worse_count": sweep_worse,
    }
