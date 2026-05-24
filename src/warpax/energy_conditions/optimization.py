"""Optimization-based EC verification (JAX + Optimistix).

Minimises WEC/NEC/SEC/DEC functionals over observer boost vectors
``w in R^3`` (null NEC via stereographic ``R^2``). Multistart BFGS;
optional projected gradient for ``|w| <= zeta_max``; JIT/vmap-compatible.
DEC is the joint minimum of the WEC margin, flux causality
``-g(j, j)`` with ``j^a = -T^a_b u^b``, and future-directedness
``-j_a n^a``. See ``optimize_wec`` / ``optimize_dec`` for ``strategy``
(``'tanh'`` smooth cap or ``'hard_bound'`` projected gradient) and
``warm_start`` / ``starts`` options.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float


_VALID_WARM_STARTS: frozenset[str] = frozenset({"cold", "spatial_neighbor"})
_VALID_STARTS: frozenset[str] = frozenset({"axis+gaussian", "fibonacci+bfgs_top_k"})


def _validate_warm_start(warm_start: str) -> None:
    """raise ValueError if warm_start is not in the allowed set."""
    if warm_start not in _VALID_WARM_STARTS:
        raise ValueError(
            f"warm_start must be one of {{'cold', 'spatial_neighbor'}}, "
            f"got {warm_start!r}"
        )


def _validate_neighbor_fraction(neighbor_fraction: float) -> None:
    """raise ValueError on out-of-range neighbor_fraction (0 < f <= 1)."""
    if not (0.0 < float(neighbor_fraction) <= 1.0):
        raise ValueError(
            "neighbor_fraction must satisfy 0 < neighbor_fraction <= 1, "
            f"got {neighbor_fraction!r}"
        )


def _validate_starts(starts: str) -> None:
    """raise ValueError if starts is not in the allowed set."""
    if starts not in _VALID_STARTS:
        raise ValueError(
            f"starts must be one of {{'axis+gaussian', 'fibonacci+bfgs_top_k'}}, "
            f"got {starts!r}"
        )

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
    """DEC objective: ``min(WEC, flux_causality, future_directedness)``.

    Each sub-margin is positive iff the corresponding DEC sub-condition
    holds; minimum exposes the worst sub-condition for the boost ``w``.
    """
    T_ab, T_mixed, g_ab, tetrad, zeta_max = args
    u = timelike_from_boost_vector(w, tetrad, zeta_max)
    j = -jnp.einsum("ac,c->a", T_mixed, u)
    wec_margin = jnp.einsum("a,ab,b->", u, T_ab, u)
    flux_causality = -jnp.einsum("a,ab,b->", j, g_ab, j)
    n_up = tetrad[0]
    n_down = jnp.einsum("ab,b->a", g_ab, n_up)
    future_margin = -jnp.einsum("a,a->", j, n_down)
    return jnp.minimum(wec_margin, jnp.minimum(flux_causality, future_margin))


# Fixed fold_in salts (literal constants, not py3.12 randomised ``hash``).
_DEC_SUB_SALTS = {
    'wec':    0x57_45_43_01,
    'flux':   0x46_4C_58_02,
    'future': 0x46_55_54_03,
}


def _dec_wec_subobjective(w, args):
    """DEC sub: ``T_{ab} u^a u^b``. Positive when WEC is satisfied (sub of DEC)."""
    T_ab, _T_mixed, _g_ab, tetrad, zeta_max = args
    u = timelike_from_boost_vector(w, tetrad, zeta_max)
    return jnp.einsum("a,ab,b->", u, T_ab, u)


def _dec_flux_subobjective(w, args):
    """DEC sub: ``-g_{ab} j^a j^b``. Positive when flux is causal (timelike/null)."""
    _T_ab, T_mixed, g_ab, tetrad, zeta_max = args
    u = timelike_from_boost_vector(w, tetrad, zeta_max)
    j = -jnp.einsum("ac,c->a", T_mixed, u)
    return -jnp.einsum("a,ab,b->", j, g_ab, j)


def _dec_future_subobjective(w, args):
    """DEC sub: ``-j^a n_a``. Positive when flux is future-directed."""
    _T_ab, T_mixed, g_ab, tetrad, zeta_max = args
    u = timelike_from_boost_vector(w, tetrad, zeta_max)
    j = -jnp.einsum("ac,c->a", T_mixed, u)
    n_up = tetrad[0]
    n_down = jnp.einsum("ab,b->a", g_ab, n_up)
    return -jnp.einsum("a,a->", j, n_down)


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
        axes = jnp.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ], dtype=jnp.float64) * zeta_max
        n_axes = min(6, remaining)
        parts.append(axes[:n_axes])
        remaining -= n_axes

    if remaining > 0:
        parts.append(jax.random.normal(key, shape=(remaining, 3)) * zeta_max)

    return jnp.concatenate(parts, axis=0)


def _make_initial_conditions_2d(n_starts, key):
    """Mixture initial conditions for stereographic null direction optimization.

    Start 0: w = 0 (north pole / e_3 direction).
    Starts 1-6: axis-aligned and spread directions on S^2.
    Remaining: random.
    """
    parts = []

    parts.append(jnp.zeros((1, 2)))
    remaining = n_starts - 1

    if remaining > 0:
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


def _fibonacci_sphere_angles(n: int) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
    """Approximately uniform (theta, phi) pairs on S^2 via Fibonacci lattice."""
    n_arr = jnp.float64(n)
    indices = jnp.arange(n, dtype=jnp.float64)
    golden_ratio = (1.0 + jnp.sqrt(5.0)) / 2.0
    theta = jnp.arccos(1.0 - 2.0 * (indices + 0.5) / n_arr)
    phi = (2.0 * jnp.pi * indices / golden_ratio) % (2.0 * jnp.pi)
    return theta, phi


def _fibonacci_boost_vectors(
    n: int,
    zeta_max: float,
    key,
) -> Float[Array, "n 3"]:
    """Map Fibonacci directions on S^2 to boost 3-vectors at ~0.76 zeta_max."""
    if n <= 0:
        return jnp.zeros((0, 3), dtype=jnp.float64)
    theta, phi = _fibonacci_sphere_angles(n)
    target_zeta = zeta_max * jnp.tanh(1.0)
    w_mag = zeta_max * jnp.arctanh(jnp.minimum(target_zeta / zeta_max, 0.999))
    dirs = jnp.stack(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ],
        axis=-1,
    )
    # Small jitter breaks exact symmetries without changing coverage much.
    jitter = jax.random.normal(key, shape=(n, 3)) * (0.01 * zeta_max)
    return w_mag * dirs + jitter


def _params_to_boost_vector(
    params: Float[Array, "3"],
    zeta_max: float,
) -> Float[Array, "3"]:
    """Convert (zeta, theta, phi) to unconstrained boost vector w."""
    zeta, theta, phi = params[0], params[1], params[2]
    zeta_clamped = jnp.minimum(jnp.maximum(zeta, 0.0), zeta_max * 0.999)
    w_mag = zeta_max * jnp.arctanh(jnp.minimum(zeta_clamped / zeta_max, 0.999))
    return w_mag * jnp.array([
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.sin(phi),
        jnp.cos(theta),
    ])


def _observer_to_boost_vector(
    u: Float[Array, "4"],
    tetrad: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    zeta_max: float,
) -> Float[Array, "3"]:
    """Invert a timelike 4-vector into boost 3-vector w (approximate inverse)."""
    c = jnp.einsum("a,ab,ib->i", u, g_ab, tetrad)
    zeta = jnp.arccosh(jnp.clip(jnp.abs(c[0]), 1.0, None))
    sinh_z = jnp.maximum(jnp.sinh(jnp.minimum(zeta, 50.0)), 1e-12)
    s_hat = c[1:4] / sinh_z
    w_mag = zeta_max * jnp.arctanh(jnp.minimum(zeta / zeta_max, 1.0 - 1e-15))
    return w_mag * s_hat


def _inject_neighbor_start(
    w0_batch: Float[Array, "n 3"],
    neighbor_observer: Float[Array, "4"] | None,
    tetrad: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    zeta_max: float,
    neighbor_fraction: float,
    warm_start: str,
) -> Float[Array, "n 3"]:
    """Replace a fraction of multistart seeds with a neighbor warm-start."""
    if warm_start != "spatial_neighbor" or neighbor_observer is None:
        return w0_batch
    n_starts = w0_batch.shape[0]
    n_swap = max(1, int(round(neighbor_fraction * n_starts)))
    w_neighbor = _observer_to_boost_vector(
        neighbor_observer, tetrad, g_ab, zeta_max,
    )
    swap_slice = jnp.s_[-n_swap:]
    return w0_batch.at[swap_slice].set(
        jnp.broadcast_to(w_neighbor, (n_swap, 3))
    )


def _fibonacci_top_k_compose(
    n_starts: int,
    dim: int,
    key,
    objective_fn,
    args,
    rtol: float,
    atol: float,
    max_steps: int,
    make_cold,
    make_fib,
    pad_scale: float,
) -> Float[Array, "n d"]:
    """Compose multistart pool: origin + top-k of cold BFGS + Fibonacci seeds.

    ``make_cold(n, key) -> (n, dim)`` builds the axis+gaussian cold pool.
    ``make_fib(n, key) -> (n, dim)`` builds the Fibonacci-sphere seeds.
    Pads with Gaussian noise scaled by ``pad_scale`` if the concatenation
    falls short of ``n_starts``.
    """
    k_top = max(1, min(4, n_starts // 4))
    n_fib = max(1, n_starts - k_top - 1)
    key_cold, key_fib = jax.random.split(key)
    w0_cold = make_cold(n_starts, key_cold)
    solver = optx.BFGS(rtol=rtol, atol=atol)

    def _obj_at(w0):
        sol = optx.minimise(
            objective_fn, solver, w0, args=args,
            max_steps=max_steps, throw=False,
        )
        return objective_fn(sol.value, args), sol.value

    obj_vals, raw_opts = jax.vmap(_obj_at)(w0_cold)
    top_idx = jnp.argsort(obj_vals)[:k_top]
    w0_top = raw_opts[top_idx]
    w0_fib = make_fib(n_fib, key_fib)
    w0_batch = jnp.concatenate([jnp.zeros((1, dim)), w0_top, w0_fib], axis=0)
    if w0_batch.shape[0] > n_starts:
        return w0_batch[:n_starts]
    if w0_batch.shape[0] < n_starts:
        pad_key = jax.random.fold_in(key, 99)
        pad = jax.random.normal(
            pad_key, shape=(n_starts - w0_batch.shape[0], dim),
        ) * pad_scale
        return jnp.concatenate([w0_batch, pad], axis=0)
    return w0_batch


def _build_initial_conditions_3d(
    n_starts: int,
    zeta_max: float,
    key,
    starts: str,
    objective_fn,
    args,
    rtol: float,
    atol: float,
    max_steps: int,
) -> Float[Array, "n 3"]:
    """Build 3D multistart pool for ``starts`` mode."""
    if starts == "axis+gaussian":
        return _make_initial_conditions_3d(n_starts, zeta_max, key)
    return _fibonacci_top_k_compose(
        n_starts, 3, key, objective_fn, args, rtol, atol, max_steps,
        make_cold=lambda n, k: _make_initial_conditions_3d(n, zeta_max, k),
        make_fib=lambda n, k: _fibonacci_boost_vectors(n, zeta_max, k),
        pad_scale=zeta_max,
    )


def _build_initial_conditions_2d(
    n_starts: int,
    key,
    starts: str,
    objective_fn,
    args,
    rtol: float,
    atol: float,
    max_steps: int,
) -> Float[Array, "n 2"]:
    """Build 2D stereographic multistart pool for ``starts`` mode."""
    if starts == "axis+gaussian":
        return _make_initial_conditions_2d(n_starts, key)

    def _make_fib_stereo(n, _k):
        theta, phi = _fibonacci_sphere_angles(n)
        return jnp.stack([
            3.0 * jnp.tan(theta / 2.0) * jnp.cos(phi),
            3.0 * jnp.tan(theta / 2.0) * jnp.sin(phi),
        ], axis=-1)

    return _fibonacci_top_k_compose(
        n_starts, 2, key, objective_fn, args, rtol, atol, max_steps,
        make_cold=lambda n, k: _make_initial_conditions_2d(n, k),
        make_fib=_make_fib_stereo,
        pad_scale=2.0,
    )


class ProjectedBFGSSolver(optx.BFGS):
    """BFGS with hard projection onto the rapidity ball ``|w|_2 <= zeta_max``.

    Bound-inactive iterates pass through unchanged; bound-active iterates
    are clipped to the radially-nearest feasible point. KKT contract is
    pinned by ``test_projected_bfgs_hits_kkt_at_bound_active``. See
    https://docs.kidger.site/optimistix/examples/custom_solver/ for the
    Optimistix custom-solver pattern.
    """

    zeta_max: float

    def __init__(self, rtol, atol, zeta_max, **kwargs):
        super().__init__(rtol=rtol, atol=atol, **kwargs)
        self.zeta_max = zeta_max

    def step(self, fn, y, args, options, state, tags):
        y_new, state_new, aux = super().step(fn, y, args, options, state, tags)
        # Radial projection onto |y|_2 <= zeta_max (identity when bound-inactive).
        norm_y = jnp.linalg.norm(y_new)
        scale_factor = jnp.minimum(
            jnp.float64(1.0),
            jnp.float64(self.zeta_max) / jnp.maximum(norm_y, 1e-12),
        )
        y_projected = y_new * scale_factor
        return y_projected, state_new, aux


def _solve_multistart_3d(objective_fn, args, n_starts, zeta_max, rtol, atol,
                         max_steps, key, *,
                         starts="axis+gaussian",
                         warm_start="cold",
                         neighbor_fraction=1.0 / 16.0,
                         neighbor_observer=None,
                         g_ab=None,
                         tetrad=None):
    """Multi-start BFGS for 3D boost-vector optimization.

    Returns (best_obj, best_raw, best_physical, best_converged, best_n_steps).
    """
    solver = optx.BFGS(rtol=rtol, atol=atol)
    zeta_max_arr = jnp.float64(zeta_max)

    w0_batch = _build_initial_conditions_3d(
        n_starts, zeta_max, key, starts,
        objective_fn, args, rtol, atol, max_steps,
    )
    if tetrad is not None and g_ab is not None:
        w0_batch = _inject_neighbor_start(
            w0_batch, neighbor_observer, tetrad, g_ab, zeta_max,
            neighbor_fraction, warm_start,
        )

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
                         max_steps, key, *,
                         starts="axis+gaussian"):
    """Multi-start BFGS for 2D stereographic null direction optimization.

    Returns (best_obj, best_physical, best_converged, best_n_steps).
    """
    solver = optx.BFGS(rtol=rtol, atol=atol)

    w0_batch = _build_initial_conditions_2d(
        n_starts, key, starts, objective_fn, args, rtol, atol, max_steps,
    )

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

    raw_opt, obj_vals, convergeds, n_steps_all, physicals = jax.vmap(solve_one)(
        w0_batch
    )

    best_idx = jnp.argmin(obj_vals)
    best_obj = obj_vals[best_idx]
    best_physical = physicals[best_idx]
    best_converged = convergeds[best_idx]
    best_n_steps = n_steps_all[best_idx]

    return best_obj, best_physical, best_converged, best_n_steps


def _solve_multistart_3d_projected(objective_fn, args, n_starts, zeta_max,
                                   rtol, atol, max_steps, key, *,
                                   starts="axis+gaussian",
                                   warm_start="cold",
                                   neighbor_fraction=1.0 / 16.0,
                                   neighbor_observer=None,
                                   g_ab=None,
                                   tetrad=None):
    """Projected BFGS variant of :func:`_solve_multistart_3d`."""
    solver = ProjectedBFGSSolver(rtol=rtol, atol=atol, zeta_max=zeta_max)
    zeta_max_arr = jnp.float64(zeta_max)

    w0_batch = _build_initial_conditions_3d(
        n_starts, zeta_max, key, starts,
        objective_fn, args, rtol, atol, max_steps,
    )
    if tetrad is not None and g_ab is not None:
        w0_batch = _inject_neighbor_start(
            w0_batch, neighbor_observer, tetrad, g_ab, zeta_max,
            neighbor_fraction, warm_start,
        )

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

    raw_opt, obj_vals, convergeds, n_steps_all, physicals = jax.vmap(solve_one)(
        w0_batch
    )

    best_idx = jnp.argmin(obj_vals)
    best_obj = obj_vals[best_idx]
    best_raw = raw_opt[best_idx]
    best_physical = physicals[best_idx]
    best_converged = convergeds[best_idx]
    best_n_steps = n_steps_all[best_idx]

    return best_obj, best_raw, best_physical, best_converged, best_n_steps


def _dispatch_multistart_3d(strategy, objective_fn, args, n_starts, zeta_max,
                            rtol, atol, max_steps, key, *,
                            starts="axis+gaussian",
                            warm_start="cold",
                            neighbor_fraction=1.0 / 16.0,
                            neighbor_observer=None,
                            g_ab=None,
                            tetrad=None):
    """strategy dispatcher for 3D multistart BFGS.

    Routes ``strategy='tanh'`` to :func:`_solve_multistart_3d` and
    ``strategy='hard_bound'`` to :func:`_solve_multistart_3d_projected`.
    Any other value raises a ``ValueError`` with the allowed set.
    """
    common = dict(
        starts=starts,
        warm_start=warm_start,
        neighbor_fraction=neighbor_fraction,
        neighbor_observer=neighbor_observer,
        g_ab=g_ab,
        tetrad=tetrad,
    )
    if strategy == "tanh":
        return _solve_multistart_3d(
            objective_fn, args, n_starts, zeta_max, rtol, atol, max_steps, key,
            **common,
        )
    elif strategy == "hard_bound":
        return _solve_multistart_3d_projected(
            objective_fn, args, n_starts, zeta_max, rtol, atol, max_steps, key,
            **common,
        )
    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Must be one of {{'tanh', 'hard_bound'}}."
        )


def _dec_per_subcondition_min(T_ab, T_mixed, g_ab, tetrad, n_starts, zeta_max,
                              rtol, atol, max_steps, key, strategy):
    """Three independent DEC sub-optimizations; outer ``min`` of mins.

    Each sub-optimization uses a seed-isolated PRNG key via
    ``jax.random.fold_in(key, salt)`` and the strategy-dispatched
    multistart solver (BFGS / ProjectedBFGS).
    """
    args = (T_ab, T_mixed, g_ab, tetrad, jnp.float64(zeta_max))

    if strategy == "tanh":
        multistart = _solve_multistart_3d
    elif strategy == "hard_bound":
        multistart = _solve_multistart_3d_projected
    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Must be one of {{'tanh', 'hard_bound'}}."
        )

    sub_objs = {
        'wec':    _dec_wec_subobjective,
        'flux':   _dec_flux_subobjective,
        'future': _dec_future_subobjective,
    }

    sub_obj_vals = []
    sub_raws = []
    sub_physicals = []
    sub_convergeds = []
    sub_n_steps_all = []
    for name in ('wec', 'flux', 'future'):
        sub_key = jax.random.fold_in(key, _DEC_SUB_SALTS[name])
        obj_val, raw, physical, converged, n_steps = multistart(
            sub_objs[name], args, n_starts, zeta_max,
            rtol, atol, max_steps, sub_key,
        )
        sub_obj_vals.append(obj_val)
        sub_raws.append(raw)
        sub_physicals.append(physical)
        sub_convergeds.append(converged)
        sub_n_steps_all.append(n_steps)

    margins = jnp.stack(sub_obj_vals)
    worst_idx = jnp.argmin(margins)
    worst_margin = margins[worst_idx]
    worst_raw = jnp.stack(sub_raws)[worst_idx]
    worst_physical = jnp.stack(sub_physicals)[worst_idx]
    worst_converged = jnp.stack(sub_convergeds)[worst_idx]
    worst_n_steps = jnp.stack(sub_n_steps_all)[worst_idx]

    return (worst_margin, worst_raw, worst_physical,
            worst_converged, worst_n_steps)


def optimize_wec(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    n_starts: int = 16,
    zeta_max: float = 5.0,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 256,
    key=None,
    strategy: str = "tanh",
    warm_start: str = "cold",
    neighbor_fraction: float = 1.0 / 16.0,
    starts: str = "axis+gaussian",
    neighbor_observer: Float[Array, "4"] | None = None,
    tetrad: Float[Array, "4 4"] | None = None,
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
    strategy : {'tanh', 'hard_bound'}
        ``'tanh'`` smooth rapidity cap vs ``'hard_bound'``
        projected gradient on the closed ball ``|w|_2 <= zeta_max``.

    Returns
    -------
    OptimizationResult
        Contains margin, worst observer 4-vector, params, convergence info.
    """
    _validate_warm_start(warm_start)
    _validate_neighbor_fraction(neighbor_fraction)
    _validate_starts(starts)

    if key is None:
        key = jax.random.PRNGKey(42)

    if tetrad is None:
        tetrad = compute_orthonormal_tetrad(g_ab)
    zeta_max_arr = jnp.float64(zeta_max)
    args = (T_ab, tetrad, zeta_max_arr)

    best_obj, best_raw, best_physical, best_converged, best_n_steps = (
        _dispatch_multistart_3d(
            strategy, _wec_objective, args, n_starts, zeta_max,
            rtol, atol, max_steps, key,
            starts=starts,
            warm_start=warm_start,
            neighbor_fraction=neighbor_fraction,
            neighbor_observer=neighbor_observer,
            g_ab=g_ab,
            tetrad=tetrad,
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


def optimize_nec(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    n_starts: int = 16,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 256,
    key=None,
    strategy: str = "tanh",
    warm_start: str = "cold",
    neighbor_fraction: float = 1.0 / 16.0,
    starts: str = "axis+gaussian",
    neighbor_observer: Float[Array, "4"] | None = None,
    tetrad: Float[Array, "4 4"] | None = None,
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
    strategy : {'tanh', 'hard_bound'}
        Validated for typos; both route through 2D stereographic
        multistart (no bound-active distinction on ``S^2``).

    Returns
    -------
    OptimizationResult
        Contains margin, worst null 4-vector, params (0, theta, phi),
        convergence info.
    """
    if strategy not in ("tanh", "hard_bound"):
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Must be one of {{'tanh', 'hard_bound'}}."
        )

    _validate_warm_start(warm_start)
    _validate_neighbor_fraction(neighbor_fraction)
    _validate_starts(starts)

    if key is None:
        key = jax.random.PRNGKey(42)

    if tetrad is None:
        tetrad = compute_orthonormal_tetrad(g_ab)
    args = (T_ab, tetrad)

    best_obj, best_physical, best_converged, best_n_steps = (
        _solve_multistart_2d(
            _nec_objective, args, n_starts, rtol, atol, max_steps, key,
            starts=starts,
        )
    )

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
    strategy: str = "tanh",
    warm_start: str = "cold",
    neighbor_fraction: float = 1.0 / 16.0,
    starts: str = "axis+gaussian",
    neighbor_observer: Float[Array, "4"] | None = None,
    tetrad: Float[Array, "4 4"] | None = None,
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
    n_starts, zeta_max, rtol, atol, max_steps, key, strategy
        Same as optimize_wec (strategy is the dispatch).
    warm_start, neighbor_fraction, starts, neighbor_observer
        Same semantics as ``optimize_wec``.

    Returns
    -------
    OptimizationResult
        Contains margin, worst observer 4-vector, params, convergence info.
    """
    _validate_warm_start(warm_start)
    _validate_neighbor_fraction(neighbor_fraction)
    _validate_starts(starts)

    if key is None:
        key = jax.random.PRNGKey(42)

    if tetrad is None:
        tetrad = compute_orthonormal_tetrad(g_ab)
    g_inv = jnp.linalg.inv(g_ab)
    T_trace = jnp.einsum("ab,ab->", g_inv, T_ab)
    sec_tensor = T_ab - 0.5 * T_trace * g_ab
    zeta_max_arr = jnp.float64(zeta_max)

    args = (sec_tensor, tetrad, zeta_max_arr)

    best_obj, best_raw, best_physical, best_converged, best_n_steps = (
        _dispatch_multistart_3d(
            strategy, _sec_objective, args, n_starts, zeta_max,
            rtol, atol, max_steps, key,
            starts=starts,
            warm_start=warm_start,
            neighbor_fraction=neighbor_fraction,
            neighbor_observer=neighbor_observer,
            g_ab=g_ab,
            tetrad=tetrad,
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
    strategy: str = "tanh",
    mode: str = "three_term_min",
    warm_start: str = "cold",
    neighbor_fraction: float = 1.0 / 16.0,
    starts: str = "axis+gaussian",
    neighbor_observer: Float[Array, "4"] | None = None,
    tetrad: Float[Array, "4 4"] | None = None,
) -> OptimizationResult:
    """Find the worst-case DEC observer via Optimistix BFGS.

    DEC requires three conditions on every future-directed timelike
    observer ``u``: (i) WEC margin ``T_{ab} u^a u^b >= 0``, (ii) flux
    causality ``-g_{ab} j^a j^b >= 0``, and (iii) future-directedness
    ``-j^a n_a >= 0`` with ``j^a = -T^a_b u^b``. ``_dec_objective``
    returns the minimum of these three; the optimizer minimizes that
    composite over ``u`` to obtain the **DEC margin** (positive iff DEC
    holds at the point). See Hawking & Ellis §4.3.

    Parameters
    ----------
    T_ab : Float[Array, "4 4"]
        Stress-energy tensor (covariant) at a single point.
    g_ab : Float[Array, "4 4"]
        Metric tensor (covariant) at the same point.
    n_starts, zeta_max, rtol, atol, max_steps, key, strategy
        Same as optimize_wec (strategy is the dispatch).
    mode : {'three_term_min', 'per_subcondition_min'}
        Single composite objective vs three seed-isolated
        sub-optimizations combined by outer ``min``.

    Returns
    -------
    OptimizationResult
        ``margin`` is the minimum across the three DEC sub-conditions
        evaluated at the worst-case observer; positive iff DEC holds
        at the point. ``worst_observer`` is the timelike four-velocity
        that realises the minimum.
    """
    _validate_warm_start(warm_start)
    _validate_neighbor_fraction(neighbor_fraction)
    _validate_starts(starts)

    if key is None:
        key = jax.random.PRNGKey(42)

    if tetrad is None:
        tetrad = compute_orthonormal_tetrad(g_ab)
    g_inv = jnp.linalg.inv(g_ab)
    T_mixed = jnp.einsum("ac,cb->ab", g_inv, T_ab)
    zeta_max_arr = jnp.float64(zeta_max)

    if mode == "three_term_min":
        args = (T_ab, T_mixed, g_ab, tetrad, zeta_max_arr)
        best_obj, best_raw, best_physical, best_converged, best_n_steps = (
            _dispatch_multistart_3d(
                strategy, _dec_objective, args, n_starts, zeta_max,
                rtol, atol, max_steps, key,
                starts=starts,
                warm_start=warm_start,
                neighbor_fraction=neighbor_fraction,
                neighbor_observer=neighbor_observer,
                g_ab=g_ab,
                tetrad=tetrad,
            )
        )
    elif mode == "per_subcondition_min":
        best_obj, best_raw, best_physical, best_converged, best_n_steps = (
            _dec_per_subcondition_min(
                T_ab, T_mixed, g_ab, tetrad,
                n_starts, zeta_max, rtol, atol, max_steps, key, strategy,
            )
        )
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. "
            f"Must be one of {{'three_term_min', 'per_subcondition_min'}}."
        )

    worst_u = timelike_from_boost_vector(best_raw, tetrad, zeta_max_arr)

    return OptimizationResult(
        margin=best_obj,
        worst_observer=worst_u,
        worst_params=best_physical,
        converged=best_converged,
        n_steps=best_n_steps,
    )


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
    strategy: str = "tanh",
    warm_start: str = "cold",
    neighbor_fraction: float = 1.0 / 16.0,
    starts: str = "axis+gaussian",
    neighbor_observer: Float[Array, "4"] | None = None,
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
    strategy : {'tanh', 'hard_bound'}
        Solver dispatch forwarded to each per-condition optimizer.

    Returns
    -------
    dict mapping condition name to OptimizationResult.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    # Hoist the orthonormal tetrad once and share across the four optimizers.
    tetrad = compute_orthonormal_tetrad(g_ab)

    _optimizers = {
        "nec": lambda k: optimize_nec(
            T_ab, g_ab, n_starts, rtol, atol, max_steps, k, strategy=strategy,
            starts=starts, tetrad=tetrad,
        ),
        "wec": lambda k: optimize_wec(
            T_ab, g_ab, n_starts, zeta_max, rtol, atol, max_steps, k,
            strategy=strategy,
            warm_start=warm_start,
            neighbor_fraction=neighbor_fraction,
            starts=starts,
            neighbor_observer=neighbor_observer,
            tetrad=tetrad,
        ),
        "sec": lambda k: optimize_sec(
            T_ab, g_ab, n_starts, zeta_max, rtol, atol, max_steps, k,
            strategy=strategy,
            warm_start=warm_start,
            neighbor_fraction=neighbor_fraction,
            starts=starts,
            neighbor_observer=neighbor_observer,
            tetrad=tetrad,
        ),
        "dec": lambda k: optimize_dec(
            T_ab, g_ab, n_starts, zeta_max, rtol, atol, max_steps, k,
            strategy=strategy,
            warm_start=warm_start,
            neighbor_fraction=neighbor_fraction,
            starts=starts,
            neighbor_observer=neighbor_observer,
            tetrad=tetrad,
        ),
    }

    results = {}
    for cond in conditions:
        key, subkey = jax.random.split(key)
        results[cond] = _optimizers[cond](subkey)

    return results


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
