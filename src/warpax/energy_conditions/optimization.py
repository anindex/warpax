"""Optimization-based energy condition verification (pure JAX + Optimistix).

Uses unconstrained 3-vector parameterization for timelike observers and
stereographic projection for null directions, fully JIT/vmap-compatible.

For each energy condition, we minimize the corresponding functional over the
observer parameter space to find the worst-case (violation-maximizing) observer:

- WEC: min_{w in R^3} T_{ab} u^a u^b  (timelike u, 3D)
- NEC: min_{w in R^2} T_{ab} k^a k^b  (null k, 2D via stereographic)
- SEC: min_{w in R^3} (T_{ab} - 1/2 T g_{ab}) u^a u^b  (timelike u, 3D)
- DEC: max_{w in R^3} g_{ab} j^a j^b  (flux causality, 3D)

The boost 3-vector w encodes both rapidity (|w|) and direction (w/|w|).
When w = 0, the observer is exactly Eulerian.  Rapidity is capped smoothly
via zeta = zeta_max * tanh(|w| / zeta_max).

Multi-start uses a mixture of deterministic starts (Eulerian + axis-aligned)
and random starts, replacing the sigmoid-biased initialization.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float

from .observer import (
    boost_vector_to_params,
    compute_orthonormal_tetrad,
    null_from_stereo,
    stereo_to_params,
    timelike_from_boost_vector,
    timelike_from_rapidity,
)


class OptimizationResult(NamedTuple):
    """Result of optimization-based energy condition check at a single point."""

    margin: Float[Array, ""]  # Minimum value found (negative = violated)
    worst_observer: Float[Array, "4"]  # 4-vector achieving the minimum
    worst_params: Float[Array, "3"]  # (zeta, theta, phi) at minimum
    converged: Float[Array, ""]  # 1.0 if converged, 0.0 otherwise
    n_steps: Float[Array, ""]  # Number of optimizer steps taken


# ---------------------------------------------------------------------------
# Internal helpers: objectives (3-vector parameterization)
# ---------------------------------------------------------------------------


def _wec_objective(w, args):
    """WEC objective: T_{ab} u^a u^b from boost 3-vector w."""
    T_ab, tetrad, zeta_max = args
    u = timelike_from_boost_vector(w, tetrad, zeta_max)
    return jnp.einsum("a,ab,b->", u, T_ab, u)


def _nec_objective(w, args):
    """NEC objective: T_{ab} k^a k^b from stereographic 2-vector w."""
    T_ab, tetrad = args
    k = null_from_stereo(w, tetrad)
    return jnp.einsum("a,ab,b->", k, T_ab, k)


def _sec_objective(w, args):
    """SEC objective: (T_{ab} - 0.5 T g_{ab}) u^a u^b from boost 3-vector w."""
    sec_tensor, tetrad, zeta_max = args
    u = timelike_from_boost_vector(w, tetrad, zeta_max)
    return jnp.einsum("a,ab,b->", u, sec_tensor, u)


def _dec_objective(w, args):
    """DEC objective: min(flux_causality_margin, future_directedness_margin).

    Flux causality: -g_{ab} j^a j^b (positive when flux is causal).
    Future-directedness: -j^a n_a (positive when flux is future-directed,
        where n = tetrad[0] is the future-pointing unit normal).

    We minimize the worse (smaller) of the two margins to find the
    observer that most violates either aspect of DEC.
    """
    T_mixed, g_ab, tetrad, zeta_max = args
    u = timelike_from_boost_vector(w, tetrad, zeta_max)
    j = -jnp.einsum("ac,c->a", T_mixed, u)

    # Causality margin: positive when flux is timelike/null (causal)
    flux_causality = -jnp.einsum("a,ab,b->", j, g_ab, j)

    # Future-directedness margin: j . n < 0 means future-directed
    # (our convention: n = tetrad[0] is future-pointing, signature -+++)
    # n_a = g_{ab} n^b (lower the index)
    n_up = tetrad[0]
    n_down = jnp.einsum("ab,b->a", g_ab, n_up)
    future_margin = -jnp.einsum("a,a->", j, n_down)

    return jnp.minimum(flux_causality, future_margin)


# ---------------------------------------------------------------------------
# Internal helpers: initialization
# ---------------------------------------------------------------------------


def _make_initial_conditions_3d(n_starts, zeta_max, key):
    """Mixture initial conditions for 3-vector observer optimization.

    Start 0: w = 0 (Eulerian observer, always included).
    Starts 1-6: axis-aligned boosts at scale zeta_max (tanh(1) ~ 0.76 zeta_max).
    Remaining: random with scale ~ zeta_max.
    """
    parts = []

    # Always include Eulerian (w = 0)
    parts.append(jnp.zeros((1, 3)))
    remaining = n_starts - 1

    if remaining > 0:
        # Axis-aligned at scale zeta_max
        axes = jnp.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ], dtype=jnp.float64) * zeta_max
        n_axes = min(6, remaining)
        parts.append(axes[:n_axes])
        remaining -= n_axes

    if remaining > 0:
        # Random with scale ~ zeta_max to cover the full rapidity range
        parts.append(jax.random.normal(key, shape=(remaining, 3)) * zeta_max)

    return jnp.concatenate(parts, axis=0)


def _make_initial_conditions_2d(n_starts, key):
    """Mixture initial conditions for stereographic null direction optimization.

    Start 0: w = 0 (north pole / e_3 direction).
    Starts 1-6: axis-aligned and spread directions on S^2.
    Remaining: random.
    """
    parts = []

    # Origin (north pole)
    parts.append(jnp.zeros((1, 2)))
    remaining = n_starts - 1

    if remaining > 0:
        # Spread across S^2 via stereographic coordinates
        axes = jnp.array([
            [1, 0], [-1, 0],    # ±e_1 on equator
            [0, 1], [0, -1],    # ±e_2 on equator
            [3, 0], [0, 3],     # near south pole
        ], dtype=jnp.float64)
        n_axes = min(6, remaining)
        parts.append(axes[:n_axes])
        remaining -= n_axes

    if remaining > 0:
        parts.append(jax.random.normal(key, shape=(remaining, 2)) * 2.0)

    return jnp.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Internal helpers: multi-start solvers
# ---------------------------------------------------------------------------


def _solve_multistart_3d(objective_fn, args, n_starts, zeta_max, rtol, atol,
                         max_steps, key):
    """Multi-start BFGS for 3D boost-vector optimization.

    Returns (best_obj, best_raw, best_physical, best_converged, best_n_steps).
    """
    solver = optx.BFGS(rtol=rtol, atol=atol)
    zeta_max_arr = jnp.float64(zeta_max)

    def solve_one(w0):
        sol = optx.minimise(
            objective_fn, solver, w0, args=args,
            max_steps=max_steps, throw=False
        )
        obj_val = objective_fn(sol.value, args)
        converged = (sol.result == optx.RESULTS.successful).astype(jnp.float64)
        n_steps = sol.stats["num_steps"].astype(jnp.float64)
        physical = boost_vector_to_params(sol.value, zeta_max_arr)
        return sol.value, obj_val, converged, n_steps, physical

    w0_batch = _make_initial_conditions_3d(n_starts, zeta_max, key)

    raw_opt, obj_vals, convergeds, n_steps_all, physicals = jax.vmap(solve_one)(
        w0_batch
    )

    # Select best (minimum objective value)
    best_idx = jnp.argmin(obj_vals)
    best_obj = obj_vals[best_idx]
    best_raw = raw_opt[best_idx]
    best_physical = physicals[best_idx]
    best_converged = convergeds[best_idx]
    best_n_steps = n_steps_all[best_idx]

    return best_obj, best_raw, best_physical, best_converged, best_n_steps


def _solve_multistart_2d(objective_fn, args, n_starts, rtol, atol,
                         max_steps, key):
    """Multi-start BFGS for 2D stereographic null direction optimization.

    Returns (best_obj, best_physical, best_converged, best_n_steps).
    """
    solver = optx.BFGS(rtol=rtol, atol=atol)

    def solve_one(w0):
        sol = optx.minimise(
            objective_fn, solver, w0, args=args,
            max_steps=max_steps, throw=False
        )
        obj_val = objective_fn(sol.value, args)
        converged = (sol.result == optx.RESULTS.successful).astype(jnp.float64)
        n_steps = sol.stats["num_steps"].astype(jnp.float64)
        physical = stereo_to_params(sol.value)
        return sol.value, obj_val, converged, n_steps, physical

    w0_batch = _make_initial_conditions_2d(n_starts, key)

    raw_opt, obj_vals, convergeds, n_steps_all, physicals = jax.vmap(solve_one)(
        w0_batch
    )

    best_idx = jnp.argmin(obj_vals)
    best_obj = obj_vals[best_idx]
    best_physical = physicals[best_idx]
    best_converged = convergeds[best_idx]
    best_n_steps = n_steps_all[best_idx]

    return best_obj, best_physical, best_converged, best_n_steps


# ---------------------------------------------------------------------------
# Public API: per-condition optimizers
# ---------------------------------------------------------------------------


def optimize_wec(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    n_starts: int = 16,
    zeta_max: float = 5.0,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 256,
    key=None,
) -> OptimizationResult:
    """Find the worst-case WEC observer via Optimistix BFGS.

    Minimizes T_{ab} u^a u^b over timelike observers parameterized by
    an unconstrained boost 3-vector w, with smooth rapidity cap.
    w = 0 corresponds exactly to the Eulerian observer.

    Parameters
    ----------
    T_ab : Float[Array, "4 4"]
        Stress-energy tensor (covariant) at a single point.
    g_ab : Float[Array, "4 4"]
        Metric tensor (covariant) at the same point.
    n_starts : int
        Number of initial conditions for multi-start optimization.
    zeta_max : float
        Smooth rapidity cap: zeta = zeta_max * tanh(|w| / zeta_max).
    rtol, atol : float
        Convergence tolerances for BFGS.
    max_steps : int
        Maximum number of BFGS iterations.
    key : PRNGKey or None
        Random key for initial conditions. If None, uses PRNGKey(42).

    Returns
    -------
    OptimizationResult
        Contains margin, worst observer 4-vector, params, convergence info.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    tetrad = compute_orthonormal_tetrad(g_ab)
    zeta_max_arr = jnp.float64(zeta_max)
    args = (T_ab, tetrad, zeta_max_arr)

    best_obj, best_raw, best_physical, best_converged, best_n_steps = (
        _solve_multistart_3d(
            _wec_objective, args, n_starts, zeta_max,
            rtol, atol, max_steps, key
        )
    )

    # Reconstruct worst observer 4-vector from best boost vector
    worst_u = timelike_from_boost_vector(best_raw, tetrad, zeta_max_arr)

    return OptimizationResult(
        margin=best_obj,
        worst_observer=worst_u,
        worst_params=best_physical,
        converged=best_converged,
        n_steps=best_n_steps,
    )


def optimize_nec(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    n_starts: int = 16,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 256,
    key=None,
) -> OptimizationResult:
    """Find the worst-case NEC null direction via Optimistix BFGS.

    Minimizes T_{ab} k^a k^b over null vectors parameterized by
    stereographic projection from R^2 to S^2. 2D optimization
    without polar coordinate singularities.

    Parameters
    ----------
    T_ab : Float[Array, "4 4"]
        Stress-energy tensor (covariant) at a single point.
    g_ab : Float[Array, "4 4"]
        Metric tensor (covariant) at the same point.
    n_starts : int
        Number of initial conditions.
    rtol, atol : float
        Convergence tolerances for BFGS.
    max_steps : int
        Maximum number of BFGS iterations.
    key : PRNGKey or None
        Random key for initial conditions.

    Returns
    -------
    OptimizationResult
        Contains margin, worst null 4-vector, params (0, theta, phi),
        convergence info.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    tetrad = compute_orthonormal_tetrad(g_ab)
    args = (T_ab, tetrad)

    best_obj, best_physical, best_converged, best_n_steps = (
        _solve_multistart_2d(
            _nec_objective, args, n_starts, rtol, atol, max_steps, key
        )
    )

    # Reconstruct worst null vector from physical (theta, phi) params
    from .observer import null_from_angles
    worst_k = null_from_angles(best_physical[1], best_physical[2], tetrad)

    return OptimizationResult(
        margin=best_obj,
        worst_observer=worst_k,
        worst_params=best_physical,
        converged=best_converged,
        n_steps=best_n_steps,
    )


def optimize_sec(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    n_starts: int = 16,
    zeta_max: float = 5.0,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 256,
    key=None,
) -> OptimizationResult:
    """Find the worst-case SEC observer via Optimistix BFGS.

    Minimizes (T_{ab} - 0.5 T g_{ab}) u^a u^b over timelike observers
    parameterized by an unconstrained boost 3-vector.

    Parameters
    ----------
    T_ab : Float[Array, "4 4"]
        Stress-energy tensor (covariant) at a single point.
    g_ab : Float[Array, "4 4"]
        Metric tensor (covariant) at the same point.
    n_starts, zeta_max, rtol, atol, max_steps, key
        Same as optimize_wec.

    Returns
    -------
    OptimizationResult
        Contains margin, worst observer 4-vector, params, convergence info.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    tetrad = compute_orthonormal_tetrad(g_ab)
    g_inv = jnp.linalg.inv(g_ab)
    T_trace = jnp.einsum("ab,ab->", g_inv, T_ab)
    sec_tensor = T_ab - 0.5 * T_trace * g_ab
    zeta_max_arr = jnp.float64(zeta_max)

    args = (sec_tensor, tetrad, zeta_max_arr)

    best_obj, best_raw, best_physical, best_converged, best_n_steps = (
        _solve_multistart_3d(
            _sec_objective, args, n_starts, zeta_max,
            rtol, atol, max_steps, key
        )
    )

    worst_u = timelike_from_boost_vector(best_raw, tetrad, zeta_max_arr)

    return OptimizationResult(
        margin=best_obj,
        worst_observer=worst_u,
        worst_params=best_physical,
        converged=best_converged,
        n_steps=best_n_steps,
    )


def optimize_dec(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    n_starts: int = 16,
    zeta_max: float = 5.0,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 256,
    key=None,
) -> OptimizationResult:
    """Find the worst-case DEC observer via Optimistix BFGS.

    DEC requires T^a_b u^b to be causal (timelike or null) for all timelike u.
    We maximize g_{ab} j^a j^b where j^a = -T^a_c u^c. DEC violated when
    the maximum is positive (spacelike flux).

    The returned margin = -max(g_{ab} j^a j^b), so margin >= 0 means satisfied.

    Parameters
    ----------
    T_ab : Float[Array, "4 4"]
        Stress-energy tensor (covariant) at a single point.
    g_ab : Float[Array, "4 4"]
        Metric tensor (covariant) at the same point.
    n_starts, zeta_max, rtol, atol, max_steps, key
        Same as optimize_wec.

    Returns
    -------
    OptimizationResult
        Contains margin (negative of max flux norm), worst observer, etc.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    tetrad = compute_orthonormal_tetrad(g_ab)
    g_inv = jnp.linalg.inv(g_ab)
    T_mixed = jnp.einsum("ac,cb->ab", g_inv, T_ab)  # T^a_b
    zeta_max_arr = jnp.float64(zeta_max)

    args = (T_mixed, g_ab, tetrad, zeta_max_arr)

    best_obj, best_raw, best_physical, best_converged, best_n_steps = (
        _solve_multistart_3d(
            _dec_objective, args, n_starts, zeta_max,
            rtol, atol, max_steps, key
        )
    )

    worst_u = timelike_from_boost_vector(best_raw, tetrad, zeta_max_arr)

    return OptimizationResult(
        margin=best_obj,
        worst_observer=worst_u,
        worst_params=best_physical,
        converged=best_converged,
        n_steps=best_n_steps,
    )


# ---------------------------------------------------------------------------
# Combined optimizer
# ---------------------------------------------------------------------------


def optimize_point(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    conditions: tuple[str, ...] = ("nec", "wec", "sec", "dec"),
    n_starts: int = 16,
    zeta_max: float = 5.0,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 256,
    key=None,
) -> dict[str, OptimizationResult]:
    """Run optimization for multiple energy conditions at a single point.

    Parameters
    ----------
    T_ab, g_ab : Float[Array, "4 4"]
        Stress-energy and metric tensors at a point.
    conditions : tuple of str
        Which conditions to check. Subset of ('nec', 'wec', 'sec', 'dec').
    n_starts, zeta_max, rtol, atol, max_steps : optimization parameters.
    key : PRNGKey or None.

    Returns
    -------
    dict mapping condition name to OptimizationResult.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    _optimizers = {
        "nec": lambda k: optimize_nec(T_ab, g_ab, n_starts, rtol, atol, max_steps, k),
        "wec": lambda k: optimize_wec(T_ab, g_ab, n_starts, zeta_max, rtol, atol, max_steps, k),
        "sec": lambda k: optimize_sec(T_ab, g_ab, n_starts, zeta_max, rtol, atol, max_steps, k),
        "dec": lambda k: optimize_dec(T_ab, g_ab, n_starts, zeta_max, rtol, atol, max_steps, k),
    }

    results = {}
    for cond in conditions:
        key, subkey = jax.random.split(key)
        results[cond] = _optimizers[cond](subkey)

    return results


# ---------------------------------------------------------------------------
# Adaptive rapidity (Python control flow not JIT-compilable)
# ---------------------------------------------------------------------------


def optimize_wec_adaptive(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    initial_zeta_max: float = 5.0,
    extension_factor: float = 2.0,
    max_extensions: int = 3,
    n_starts: int = 16,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 256,
    key=None,
    boundary_threshold: float = 0.05,
) -> OptimizationResult:
    """WEC optimization with adaptive rapidity range extension.

    First pass uses initial_zeta_max. If the best solution has zeta near
    the boundary (within boundary_threshold fraction of zeta_max) AND the
    gradient of the objective at that point indicates the minimum may be
    further out, extends zeta_max by extension_factor and re-runs.

    This function uses Python control flow and is NOT JIT-compilable.

    Parameters
    ----------
    T_ab, g_ab : Float[Array, "4 4"]
        Stress-energy and metric tensors.
    initial_zeta_max : float
        Starting rapidity range.
    extension_factor : float
        Multiply zeta_max by this factor on each extension.
    max_extensions : int
        Maximum number of range extensions.
    n_starts, rtol, atol, max_steps, key : optimization parameters.
    boundary_threshold : float
        Fraction of zeta_max considered "near boundary" (default 5%).

    Returns
    -------
    OptimizationResult
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    zeta_max = initial_zeta_max

    tetrad = compute_orthonormal_tetrad(g_ab)

    for _ in range(max_extensions + 1):
        key, subkey = jax.random.split(key)
        result = optimize_wec(
            T_ab, g_ab, n_starts, zeta_max, rtol, atol, max_steps, subkey
        )

        best_zeta = result.worst_params[0]
        boundary_val = zeta_max * (1.0 - boundary_threshold)

        # Check if best zeta is near boundary
        near_boundary = float(best_zeta) > boundary_val

        if not near_boundary:
            return result

        # Check if gradient points toward lower objective at boundary
        def wec_zeta_grad(zeta_val):
            u = timelike_from_rapidity(
                zeta_val, result.worst_params[1], result.worst_params[2], tetrad
            )
            return jnp.einsum("a,ab,b->", u, T_ab, u)

        grad_at_boundary = jax.grad(wec_zeta_grad)(best_zeta)

        if float(grad_at_boundary) < 0:
            zeta_max *= extension_factor
        else:
            return result

    return result
