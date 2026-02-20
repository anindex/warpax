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


# ---------------------------------------------------------------------------
# Observer family generators
# ---------------------------------------------------------------------------


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

    # Fixed directions: along x (theta=pi/2, phi=0), y (theta=pi/2, phi=pi/2), z (theta=0, phi=0)
    directions = jnp.array([
        [jnp.pi / 2, 0.0],          # +x
        [jnp.pi / 2, jnp.pi / 2],   # +y
        [0.0, 0.0],                  # +z
    ])[:n_directions]

    # Outer product: all combinations of rapidity x direction
    # Result shape: (n_rapidity, n_directions, 3) -> flatten to (K, 3)
    params = []
    for i in range(n_rapidity):
        for j in range(min(n_directions, directions.shape[0])):
            params.append(jnp.array([zetas[i], directions[j, 0], directions[j, 1]]))

    return jnp.stack(params)  # (K, 3)


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

    params = []
    for i in range(n_theta):
        for j in range(n_phi):
            params.append(jnp.array([zeta_fixed, thetas[i], phis[j]]))

    return jnp.stack(params)  # (K, 3)


# ---------------------------------------------------------------------------
# Vectorized margin computation (double vmap)
# ---------------------------------------------------------------------------


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
    return jax.vmap(_point_sweep_wec, in_axes=(0, 0, None))(
        T_field, g_field, observer_params
    )


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
    return jax.vmap(_point_sweep_nec, in_axes=(0, 0, None))(
        T_field, g_field, observer_params
    )


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
    """
    # WEC: T_{ab} u^a u^b for timelike observers
    wec = sweep_wec_margins(T_field, g_field, observer_params)

    # NEC: T_{ab} k^a k^b for null directions
    # Use (theta, phi) columns from observer_params (zeta irrelevant for null)
    nec_params = observer_params[:, 1:]  # (K, 2)
    nec = sweep_nec_margins(T_field, g_field, nec_params)

    # SEC: (T_{ab} - 0.5 T g_{ab}) u^a u^b for timelike observers
    T_trace = jnp.einsum("nab,nab->n", g_inv_field, T_field)  # (N,)
    sec_tensor = T_field - 0.5 * T_trace[:, None, None] * g_field  # (N, 4, 4)
    sec = jax.vmap(_point_sweep_sec, in_axes=(0, 0, None))(
        sec_tensor, g_field, observer_params
    )

    return {"wec": wec, "nec": nec, "sec": sec}


# ---------------------------------------------------------------------------
# Cross-validation against BFGS
# ---------------------------------------------------------------------------


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
    """
    from .optimization import optimize_wec

    if key is None:
        key = jax.random.PRNGKey(42)

    N = T_field.shape[0]
    n_validation_points = min(n_validation_points, N)

    # Select random grid points
    key, subkey = jax.random.split(key)
    indices = jax.random.choice(subkey, N, shape=(n_validation_points,), replace=False)

    # Compute sweep margins at validation points
    sweep_margins = sweep_wec_margins(T_field, g_field, observer_params)  # (N, K)
    sweep_mins = jnp.min(sweep_margins, axis=-1)  # (N,) worst across observers

    # Collect validation results
    sign_agree = 0
    sweep_worse = 0
    rel_errors = []

    for i in range(n_validation_points):
        idx = int(indices[i])
        T_point = T_field[idx]
        g_point = g_field[idx]

        key, subkey = jax.random.split(key)
        bfgs_result = optimize_wec(
            T_point, g_point,
            n_starts=n_starts,
            zeta_max=zeta_max,
            key=subkey,
        )
        bfgs_min = float(bfgs_result.margin)
        sweep_min = float(sweep_mins[idx])

        # Sign agreement: both agree on violation (negative) or satisfaction (positive)
        if (sweep_min < 0) == (bfgs_min < 0) or (abs(sweep_min) < 1e-12 and abs(bfgs_min) < 1e-12):
            sign_agree += 1

        # Sweep missed deeper violation (sweep_min > bfgs_min when both negative)
        if bfgs_min < 0 and sweep_min > bfgs_min:
            sweep_worse += 1

        # Relative error (use BFGS as reference)
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
