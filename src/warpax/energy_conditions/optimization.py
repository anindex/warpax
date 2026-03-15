"""Optimization-based energy condition verification (pure JAX + Optimistix).

Uses unconstrained 3-vector parameterization for timelike observers and
stereographic projection for null directions, fully JIT/vmap-compatible.

For each energy condition, we minimize the corresponding functional over the
observer parameter space to find the worst-case (violation-maximizing) observer:

- WEC: min_{w in R^3} T_{ab} u^a u^b (timelike u, 3D)
- NEC: min_{w in R^2} T_{ab} k^a k^b (null k, 2D via stereographic)
- SEC: min_{w in R^3} (T_{ab} - 1/2 T g_{ab}) u^a u^b (timelike u, 3D)
- DEC: max_{w in R^3} g_{ab} j^a j^b (flux causality, 3D)

The boost 3-vector w encodes both rapidity (|w|) and direction (w/|w|).
When w = 0, the observer is exactly Eulerian. Rapidity is capped smoothly
via zeta = zeta_max * tanh(|w| / zeta_max).

Multi-start uses a mixture of deterministic starts (Eulerian + axis-aligned)
and random starts, replacing the sigmoid-biased initialization.

``strategy='hard_bound'`` adds a projected-gradient BFGS solver
(:class:`ProjectedBFGSSolver`) that projects every iterate onto the closed
rapidity ball ``|w|_2 <= zeta_max``.  Default ``strategy='tanh'`` preserves
the original behavior.

``warm_start='spatial_neighbor'`` enables grid-aware multistart seeding.
Default ``warm_start='cold'`` preserves the original behavior.
``neighbor_fraction`` defaults to 1/16; the pool prevents basin-clone
overlap.

``starts='fibonacci+bfgs_top_k'`` adds a Fibonacci lattice + BFGS-top-k
blended starter pool.  Default ``starts='axis+gaussian'`` preserves the
original behavior.

``backend='cpu'|'gpu'`` selects the JAX device context (default ``'cpu'``).
The env var ``WARPAX_PERF_BACKEND`` overrides the kwarg.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# Additive-kwarg validation helpers
# ---------------------------------------------------------------------------


_VALID_WARM_STARTS: frozenset[str] = frozenset({"cold", "spatial_neighbor"})
_VALID_STARTS: frozenset[str] = frozenset({"axis+gaussian", "fibonacci+bfgs_top_k"})
_VALID_BACKENDS: frozenset[str] = frozenset({"cpu", "gpu"})


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


def _validate_backend(backend: str) -> None:
    """Raise ValueError if backend is not in the allowed set."""
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"backend must be one of {{'cpu', 'gpu'}}, got {backend!r}"
        )


def _resolve_backend(backend: str) -> str:
    """WARPAX_PERF_BACKEND env var overrides kwarg."""
    effective = os.environ.get("WARPAX_PERF_BACKEND", backend)
    _validate_backend(effective)
    return effective

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
    """DEC objective: min(wec_margin, flux_causality_margin, future_directedness_margin).

    WEC margin: T_{ab} u^a u^b (positive when WEC is satisfied).
    Flux causality: -g_{ab} j^a j^b (positive when flux is causal).
    Future-directedness: -j^a n_a (positive when flux is future-directed,
        where n = tetrad[0] is the future-pointing unit normal).

    We minimize the worst (smallest) of the three margins to find the
    observer that most violates any aspect of DEC (including WEC violations,
    since DEC implies WEC).
    """
    T_ab, T_mixed, g_ab, tetrad, zeta_max = args
    u = timelike_from_boost_vector(w, tetrad, zeta_max)
    j = -jnp.einsum("ac,c->a", T_mixed, u)

    # WEC margin: T_{ab} u^a u^b >= 0 required
    wec_margin = jnp.einsum("a,ab,b->", u, T_ab, u)

    # Causality margin: positive when flux is timelike/null (causal)
    flux_causality = -jnp.einsum("a,ab,b->", j, g_ab, j)

    # Future-directedness margin: j . n < 0 means future-directed
    # (our convention: n = tetrad[0] is future-pointing, signature -+++)
    # n_a = g_{ab} n^b (lower the index)
    n_up = tetrad[0]
    n_down = jnp.einsum("ab,b->a", g_ab, n_up)
    future_margin = -jnp.einsum("a,a->", j, n_down)

    return jnp.minimum(wec_margin, jnp.minimum(flux_causality, future_margin))


# ---------------------------------------------------------------------------
# per-subcondition DEC objectives + seed salts
# Each sub returns its margin with sign convention `positive = sub satisfied,
# negative = violated` - byte-equivalent to the corresponding ``jnp.minimum``
# argument in the v0.1.x ``_dec_objective`` above.
# ---------------------------------------------------------------------------

# Canonical salts: ``hash('label') & 0x7FFFFFFF`` gives a positive 32-bit
# integer accepted by :func:`jax.random.fold_in`. Python 3.12 ``hash`` is
# process-randomised, so values VARY across processes - but reproducibility
# WITHIN a single process is guaranteed (tests compute expected keys at run
# time, not hardcoded bit patterns).
_DEC_SUB_SALTS = {
    'wec':    hash('dec:wec_margin')          & 0x7FFFFFFF,
    'flux':   hash('dec:flux_causality')      & 0x7FFFFFFF,
    'future': hash('dec:future_directedness') & 0x7FFFFFFF,
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
# hard-bound projected-gradient BFGS solver
# ---------------------------------------------------------------------------


class ProjectedBFGSSolver(optx.BFGS):
    """BFGS with hard projection onto the closed rapidity ball ``|w|_2 <= zeta_max``.

    Subclasses Optimistix's concrete :class:`optimistix.BFGS` and overrides
    :meth:`step` to project the candidate iterate radially after the parent
    BFGS step computes it. Subclassing :class:`BFGS` directly (rather than
    :class:`AbstractQuasiNewton`) inherits concrete field defaults (``norm``,
    ``use_inverse``, ``descent``, ``search``, ``verbose``) and the concrete
    ``update_hessian``/``init_hessian`` methods - the only new behavior is
    the post-step radial projection.

    At bound-inactive points the projection is the identity; at bound-active
    KKT points the iterate lands on the sphere ``|w|_2 = zeta_max`` at the
    radially-nearest feasible point. The 1D-quadratic KKT contract is pinned
    by ``test_projected_bfgs_hits_kkt_at_bound_active``.

    Notes
    -----
    - ``super.step(...)`` returns the unprojected BFGS iterate; the radial
      clip is the only post-processing.
    - All upstream Optimistix calls use ``throw=False``.
    - The class-level BFGS MRO is
      ``BFGS -> AbstractBFGS -> AbstractQuasiNewton``, so the ``step``
       override composes cleanly with the BFGS solver form.

    References
    ----------
    - Custom solvers in Optimistix:
      https://docs.kidger.site/optimistix/examples/custom_solver/
    """

    zeta_max: float

    def __init__(self, rtol, atol, zeta_max, **kwargs):
        super().__init__(rtol=rtol, atol=atol, **kwargs)
        self.zeta_max = zeta_max

    def step(self, fn, y, args, options, state, tags):
        y_new, state_new, aux = super().step(fn, y, args, options, state, tags)
        # radial projection onto |y|_2 <= zeta_max.
        # Using jnp.minimum on the scale factor makes the projection the
        # identity when |y| <= zeta_max (bound-inactive) and clips to the
        # sphere when |y| > zeta_max (bound-active).
        norm_y = jnp.linalg.norm(y_new)
        scale_factor = jnp.minimum(
            jnp.float64(1.0),
            jnp.float64(self.zeta_max) / jnp.maximum(norm_y, 1e-12),
        )
        y_projected = y_new * scale_factor
        return y_projected, state_new, aux


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


def _solve_multistart_3d_projected(objective_fn, args, n_starts, zeta_max,
                                   rtol, atol, max_steps, key):
    """projected-gradient variant of :func:`_solve_multistart_3d`.

    Mirrors the original multistart pool composition (same
    :func:`_make_initial_conditions_3d`, same key, same vmap structure) - ONLY
    the solver is swapped for :class:`ProjectedBFGSSolver`. This keeps
    bound-inactive points bit-equivalent to the tanh path within
    ``5e-6 * scale``.

    Returns (best_obj, best_raw, best_physical, best_converged, best_n_steps).
    """
    solver = ProjectedBFGSSolver(rtol=rtol, atol=atol, zeta_max=zeta_max)
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

    best_idx = jnp.argmin(obj_vals)
    best_obj = obj_vals[best_idx]
    best_raw = raw_opt[best_idx]
    best_physical = physicals[best_idx]
    best_converged = convergeds[best_idx]
    best_n_steps = n_steps_all[best_idx]

    return best_obj, best_raw, best_physical, best_converged, best_n_steps


def _dispatch_multistart_3d(strategy, objective_fn, args, n_starts, zeta_max,
                            rtol, atol, max_steps, key):
    """strategy dispatcher for 3D multistart BFGS.

    Routes ``strategy='tanh'`` to the original :func:`_solve_multistart_3d` (bit-
    exact preservation) and ``strategy='hard_bound'`` to
    :func:`_solve_multistart_3d_projected`. Any other value raises a
    ``ValueError`` with the allowed set.
    """
    if strategy == "tanh":
        return _solve_multistart_3d(
            objective_fn, args, n_starts, zeta_max, rtol, atol, max_steps, key
        )
    elif strategy == "hard_bound":
        return _solve_multistart_3d_projected(
            objective_fn, args, n_starts, zeta_max, rtol, atol, max_steps, key
        )
    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Must be one of {{'tanh', 'hard_bound'}}."
        )


# ---------------------------------------------------------------------------
# per-subcondition DEC driver
# ---------------------------------------------------------------------------


def _dec_per_subcondition_min(T_ab, T_mixed, g_ab, tetrad, n_starts, zeta_max,
                              rtol, atol, max_steps, key, strategy):
    """per-subcondition DEC: 3 independent BFGS multistarts, outer min.

    Each sub gets a seed-isolated PRNG key via ``jax.random.fold_in(key, salt)``.
    Combines via outer min-of-mins: the overall DEC margin is the MOST violated
    sub. Returns the worst sub's margin + observer + convergence info,
    matching the :class:`OptimizationResult` shape from the v0.1.x path.

    Parameters mirror :func:`_solve_multistart_3d`. ``strategy`` is passed
    through to select the v0.1.x BFGS (``'tanh'``) vs ProjectedBFGS
    (``'hard_bound'``) for each sub-multistart (so + compose:
    ``mode='per_subcondition_min', strategy='hard_bound'`` runs three
    ProjectedBFGS multistarts with seed isolation).

    Returns (best_obj, best_raw, best_physical, best_converged, best_n_steps).
    """
    args = (T_ab, T_mixed, g_ab, tetrad, jnp.float64(zeta_max))

    # Pick the inner solver according to the strategy (composes).
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

    # Outer min-of-mins: pick the most violated sub.
    margins = jnp.stack(sub_obj_vals)
    worst_idx = jnp.argmin(margins)
    worst_margin = margins[worst_idx]
    worst_raw = jnp.stack(sub_raws)[worst_idx]
    worst_physical = jnp.stack(sub_physicals)[worst_idx]
    worst_converged = jnp.stack(sub_convergeds)[worst_idx]
    worst_n_steps = jnp.stack(sub_n_steps_all)[worst_idx]

    return (worst_margin, worst_raw, worst_physical,
            worst_converged, worst_n_steps)


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
    strategy: str = "tanh",
    warm_start: str = "cold",
    neighbor_fraction: float = 1.0 / 16.0,
    starts: str = "axis+gaussian",
    backend: str = "cpu",
    neighbor_observer: Float[Array, "4"] | None = None,
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
        Solver dispatch. ``'tanh'`` (default): unconstrained
        R^3 boost vector ``w`` mapped to physical rapidity via
        ``zeta = zeta_max * tanh(|w| / zeta_max)``; smooth cap with a small
        bias toward the ball interior at bound-active points. ``'hard_bound'``:
        projected-gradient BFGS with ``|w|_2 <= zeta_max``
        enforced by radial clip every step; KKT-correct at bound-active
        optima. Bound-inactive points agree between the two strategies to
        ``5e-6 * scale``.

    Returns
    -------
    OptimizationResult
        Contains margin, worst observer 4-vector, params, convergence info.
    """
    # Validate additive kwargs (fail-fast; fp64 defaults preserve original behavior)
    _validate_warm_start(warm_start)
    _validate_neighbor_fraction(neighbor_fraction)
    _validate_starts(starts)
    effective_backend = _resolve_backend(backend)
    del effective_backend  # reserved for future backend-context dispatch

    if key is None:
        key = jax.random.PRNGKey(42)

    tetrad = compute_orthonormal_tetrad(g_ab)
    zeta_max_arr = jnp.float64(zeta_max)
    args = (T_ab, tetrad, zeta_max_arr)

    best_obj, best_raw, best_physical, best_converged, best_n_steps = (
        _dispatch_multistart_3d(
            strategy, _wec_objective, args, n_starts, zeta_max,
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
    strategy: str = "tanh",
    warm_start: str = "cold",
    neighbor_fraction: float = 1.0 / 16.0,
    starts: str = "axis+gaussian",
    backend: str = "cpu",
    neighbor_observer: Float[Array, "4"] | None = None,
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
        Solver dispatch. For NEC the stereographic 2D parameterization has
        no analogous bound-active issue - the null direction is unbounded on
        ``S^2``. Both ``'tanh'`` and ``'hard_bound'`` route through the
        :func:`_solve_multistart_2d` path. Validated for strategy-name typos.

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

    # Validate additive kwargs (fail-fast; defaults preserve original behavior)
    _validate_warm_start(warm_start)
    _validate_neighbor_fraction(neighbor_fraction)
    _validate_starts(starts)
    effective_backend = _resolve_backend(backend)
    del effective_backend  # reserved for future backend-context dispatch

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
    strategy: str = "tanh",
    warm_start: str = "cold",
    neighbor_fraction: float = 1.0 / 16.0,
    starts: str = "axis+gaussian",
    backend: str = "cpu",
    neighbor_observer: Float[Array, "4"] | None = None,
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
    warm_start, neighbor_fraction, starts, backend, neighbor_observer
        Additive kwargs - same semantics as optimize_wec.

    Returns
    -------
    OptimizationResult
        Contains margin, worst observer 4-vector, params, convergence info.
    """
    # Validate additive kwargs (fail-fast; defaults preserve original behavior)
    _validate_warm_start(warm_start)
    _validate_neighbor_fraction(neighbor_fraction)
    _validate_starts(starts)
    effective_backend = _resolve_backend(backend)
    del effective_backend  # reserved for future backend-context dispatch

    if key is None:
        key = jax.random.PRNGKey(42)

    tetrad = compute_orthonormal_tetrad(g_ab)
    g_inv = jnp.linalg.inv(g_ab)
    T_trace = jnp.einsum("ab,ab->", g_inv, T_ab)
    sec_tensor = T_ab - 0.5 * T_trace * g_ab
    zeta_max_arr = jnp.float64(zeta_max)

    args = (sec_tensor, tetrad, zeta_max_arr)

    best_obj, best_raw, best_physical, best_converged, best_n_steps = (
        _dispatch_multistart_3d(
            strategy, _sec_objective, args, n_starts, zeta_max,
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
    strategy: str = "tanh",
    mode: str = "three_term_min",
    warm_start: str = "cold",
    neighbor_fraction: float = 1.0 / 16.0,
    starts: str = "axis+gaussian",
    backend: str = "cpu",
    neighbor_observer: Float[Array, "4"] | None = None,
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
    n_starts, zeta_max, rtol, atol, max_steps, key, strategy
        Same as optimize_wec (strategy is the dispatch).
    mode : {'three_term_min', 'per_subcondition_min'}
        DEC composite mode.

        ``'three_term_min'`` (default):
            Single BFGS multistart on the three-way-min composition inside
            :func:`_dec_objective`. Smooth except at sub-crossover surfaces
            (kinks in the landscape where the active sub switches).
        ``'per_subcondition_min'`` :
            Three independent BFGS multistarts, one per sub-inequality (WEC
            margin, flux causality, future-directedness), with
            ``jax.random.fold_in(key, salt)`` seed isolation per sub. Combined
            via outer min-of-mins. Eliminates kink-bias at sub-crossover
            surfaces; matches the ``three_term_min`` result within
            ``1e-6 * scale`` at well-behaved (DEC-satisfying) points.

    Returns
    -------
    OptimizationResult
        Contains margin (negative of max flux norm), worst observer, etc.
    """
    # Validate additive kwargs (fail-fast; defaults preserve original behavior)
    _validate_warm_start(warm_start)
    _validate_neighbor_fraction(neighbor_fraction)
    _validate_starts(starts)
    effective_backend = _resolve_backend(backend)
    del effective_backend  # reserved for future backend-context dispatch

    if key is None:
        key = jax.random.PRNGKey(42)

    tetrad = compute_orthonormal_tetrad(g_ab)
    g_inv = jnp.linalg.inv(g_ab)
    # NOTE (W-1): The original _dec_objective closes over T_mixed computed via
    # ``jnp.einsum("ac,cb->ab", g_inv, T_ab)``. Do NOT substitute
    # ``jnp.linalg.solve(g_ab, T_ab)`` - float-point residuals differ at the
    # last few ULPs and would break the bit-exact regression sentinel
    # (``test_default_mode_preserves_v1_0``).
    T_mixed = jnp.einsum("ac,cb->ab", g_inv, T_ab)  # T^a_b
    zeta_max_arr = jnp.float64(zeta_max)

    if mode == "three_term_min":
        # Default path - preserved bit-exact via existing _dispatch_multistart_3d
        # routing (strategy='tanh' hits _solve_multistart_3d byte-identically).
        args = (T_ab, T_mixed, g_ab, tetrad, zeta_max_arr)
        best_obj, best_raw, best_physical, best_converged, best_n_steps = (
            _dispatch_multistart_3d(
                strategy, _dec_objective, args, n_starts, zeta_max,
                rtol, atol, max_steps, key
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
    strategy: str = "tanh",
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
        Solver dispatch forwarded to each per-condition optimizer. Default
        ``'tanh'`` preserves the original behavior.

    Returns
    -------
    dict mapping condition name to OptimizationResult.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    _optimizers = {
        "nec": lambda k: optimize_nec(
            T_ab, g_ab, n_starts, rtol, atol, max_steps, k, strategy=strategy
        ),
        "wec": lambda k: optimize_wec(
            T_ab, g_ab, n_starts, zeta_max, rtol, atol, max_steps, k,
            strategy=strategy,
        ),
        "sec": lambda k: optimize_sec(
            T_ab, g_ab, n_starts, zeta_max, rtol, atol, max_steps, k,
            strategy=strategy,
        ),
        "dec": lambda k: optimize_dec(
            T_ab, g_ab, n_starts, zeta_max, rtol, atol, max_steps, k,
            strategy=strategy,
        ),
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
