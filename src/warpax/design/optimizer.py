"""optimizer - ``design_metric`` with sigmoid reparameterization.

Reuses the ``hard_bound`` strategy name 
via sigmoid-reparameterized shape-function parameters: unconstrained
raw params ``theta`` map to physical values via ``tanh(theta)``, so the
Optimistix BFGS solver sees an unconstrained loss surface while the
physical shape-function values stay in ``[-1, 1]`` .

Per the design specification. The concrete bound here is
  the sigmoid reparameterization (shape-function param space is
  N-dim, not the 3-vector observer space that
  ``_solve_multistart_3d`` targets, so we do NOT reuse that specific
  helper - instead we reuse its *pattern*: Optimistix BFGS +
  multistart + best-of-N selection).
- Multistart ``n_starts=16`` default; per-start seed via
  ``jax.random.fold_in(key, start_idx)`` .
- Each iterate constructs a ``ShapeFunctionMetric(strict=False)`` so
  the optimizer can observe ``verify_physical`` verdicts rather than
  crash on unphysical configurations.
- Returns ``(ShapeFunctionMetric, OptimizationReport)``.

Reference:
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float

from .constraints import CONSTRAINT_REGISTRY
from .metrics import PhysicalityVerdict, ShapeFunctionMetric
from .objectives import OBJECTIVE_REGISTRY
from .shape_functions import ShapeFunction


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class OptimizationReport(NamedTuple):
    """Convergence summary for :func:`design_metric`.

    Attributes
    ----------
    converged
        True iff the best-of-N start converged.
    final_margin
        Best objective value achieved (scalar). Sign convention per the
        underlying :data:`OBJECTIVE_REGISTRY` entry - e.g. for
        ``objective='nec'`` this is the worst-point NEC margin
        (positive = satisfied, negative = violated).
    n_steps
        BFGS iteration count of the best start.
    physicality
        :class:`PhysicalityVerdict` of the final
        :class:`ShapeFunctionMetric`.
    strategy
        The strategy label passed in (for audit).
    n_starts
        The ``n_starts`` value used.
    """
    converged: bool
    final_margin: Float[Array, ""]
    n_steps: int
    physicality: PhysicalityVerdict
    strategy: str
    n_starts: int


# ---------------------------------------------------------------------------
# Sigmoid reparameterization 
# ---------------------------------------------------------------------------


def _sigmoid_reparam(theta_raw, lo=-1.0, hi=1.0):
    """Map unconstrained ``theta_raw`` to ``[lo, hi]`` via ``tanh``.

    Using ``tanh`` (instead of ``sigmoid``) centers the map at ``theta=0``
    => ``value=0``, which is a natural "do nothing" prior for
    gradient-descent starting points.
    """
    return lo + 0.5 * (hi - lo) * (1.0 + jnp.tanh(theta_raw))


def _sigmoid_inverse(value, lo=-1.0, hi=1.0):
    """Inverse of :func:`_sigmoid_reparam` for recovering raw theta
    from a given physical value (used to seed Alcubierre-centerd starts).
    """
    # value = lo + 0.5*(hi - lo)*(1 + tanh(theta))
    # => tanh(theta) = 2*(value - lo)/(hi - lo) - 1
    # Clamp for numerical stability at the sigmoid tails.
    normalized = jnp.clip(2.0 * (value - lo) / (hi - lo) - 1.0, -0.999, 0.999)
    return jnp.arctanh(normalized)


# ---------------------------------------------------------------------------
# design_metric
# ---------------------------------------------------------------------------


def design_metric(
    shape: ShapeFunction,
    objective: str = "nec",
    constraints: list[str] | None = None,
    strategy: str = "hard_bound",
    n_starts: int = 16,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 256,
    v_s: float = 0.5,
    key=None,
) -> tuple[ShapeFunctionMetric, OptimizationReport]:
    """Constrained-BFGS shape-function optimizer.

    Parameters
    ----------
    shape
        Starting :class:`ShapeFunction`. Must be a ``'spline'`` basis
        in this release (Bernstein/GMM accepted but the optimizer
        only varies spline ``values``).
    objective
        One of :data:`OBJECTIVE_REGISTRY` keys.
    constraints
        List of string keys into :data:`CONSTRAINT_REGISTRY`. Each
        constraint is added to the objective as a soft penalty:
        ``obj + sum_k max(0, -margin_k)^2``. ``None`` (default) =>
        unconstrained.
    strategy
        Strategy-dispatch label. ``'hard_bound'`` (default) applies
        sigmoid reparameterization to bound physical values to
        ``[-1, 1]``. Other values currently fall through to the same
        path (reserved for future).
    n_starts
        Multistart count . Default ``16``.
    rtol, atol, max_steps
        Optimistix BFGS convergence kwargs.
    v_s
        Bubble velocity for the :class:`ShapeFunctionMetric` instances
        constructed during optimization. Default ``0.5`` (matches
        ``paper_params.py.V_S_PRIMARY``).
    key
        ``jax.random.PRNGKey`` for multistart randomness. Default
        ``PRNGKey(42)`` (matches ``paper_params.py.PRNG_SEED_MULTISTART``).

    Returns
    -------
    (ShapeFunctionMetric, OptimizationReport)
        Best-of-N optimized metric + convergence report.

    Notes
    -----
    Only the spline ``values`` parameter is optimized; ``knots`` are
    held fixed (static interpolation grid). Non-spline bases are
    returned unchanged with a warning-free no-op report (future FUT).
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    if constraints is None:
        constraints = []
    if objective not in OBJECTIVE_REGISTRY:
        raise ValueError(
            f"design_metric: unknown objective {objective!r}; "
            f"expected one of {list(OBJECTIVE_REGISTRY)}."
        )
    obj_fn = OBJECTIVE_REGISTRY[objective]
    v_s_arr = jnp.asarray(v_s)

    # max_steps=0 short-circuit: return the input shape unchanged.
    # This is the reproduction path : the optimizer
    # verifies that the basis preserves the input profile bit-exactly
    # when no BFGS step is taken (at the extreme
    # trivially a local minimum under zero step budget).
    if max_steps <= 0:
        sfm0 = ShapeFunctionMetric(shape, v_s=v_s_arr, strict=False)
        # Evaluate the objective on the input metric so the report
        # records a real margin (supports catalog generation).
        try:
            margin0 = OBJECTIVE_REGISTRY[objective](
                sfm0, grid_shape=(6, 6, 6), bounds=((-1.5, 1.5),) * 3,
            )
        except Exception:  # noqa: BLE001
            margin0 = jnp.asarray(jnp.nan)
        return sfm0, OptimizationReport(
            converged=True,
            final_margin=margin0,
            n_steps=0,
            physicality=sfm0.verify_physical(),
            strategy=strategy,
            n_starts=n_starts,
        )

    # Non-spline bases: return unchanged (future FUT to extend)
    if shape.basis != "spline":
        sfm = ShapeFunctionMetric(shape, v_s=v_s_arr, strict=False)
        report = OptimizationReport(
            converged=False,
            final_margin=jnp.asarray(jnp.nan),
            n_steps=0,
            physicality=sfm.verify_physical(),
            strategy=strategy,
            n_starts=n_starts,
        )
        return sfm, report

    # Extract spline params
    knots = shape.params["knots"]
    values_init = shape.params["values"]

    # Sigmoid-inverse starting point: optimizer starts at the input shape
    theta_init = _sigmoid_inverse(values_init)

    # Loss function: objective(metric(theta)) + sum of constraint penalties
    def _loss(theta_raw, args):
        values = _sigmoid_reparam(theta_raw)
        sf = ShapeFunction(
            basis="spline",
            params={"knots": knots, "values": values},
            order=shape.order,
        )
        # Construct metric with strict=False so the optimizer can see
        # unphysical iterates as soft penalties rather than crashing.
        sfm = ShapeFunctionMetric(sf, v_s=v_s_arr, strict=False)
        # Keep ec_margin_objective cheap during optimization
        obj_val = obj_fn(sfm, grid_shape=(6, 6, 6), bounds=((-1.5, 1.5),) * 3)
        # Constraint penalties
        penalty = jnp.asarray(0.0)
        for c_name in constraints:
            if c_name not in CONSTRAINT_REGISTRY:
                raise ValueError(f"Unknown constraint {c_name!r}")
            c_fn = CONSTRAINT_REGISTRY[c_name]
            # bubble_size and boundedness take (sf, ...); velocity takes (v_s, ...)
            if c_name == "velocity":
                c_res = c_fn(v_s_arr)
            else:
                c_res = c_fn(sf)
            # Soft quadratic penalty: max(0, -margin)^2
            penalty = penalty + jnp.square(jnp.maximum(0.0, -c_res.margin))
        return obj_val + penalty

    solver = optx.BFGS(rtol=rtol, atol=atol)

    # Multistart: theta_init + small-radius Gaussian perturbations
    keys = jax.random.split(key, n_starts)
    # For the first start, use exact theta_init (no perturbation) so that
    # reproduction preserves the Alcubierre starting point bit-exactly.
    # Subsequent starts perturb for local-optimum exploration .

    # Always evaluate the starting shape as the initial best baseline.
    # This implements : the optimizer respects the seed's local minimum
    # by never accepting a worse iterate than the starting point.
    best_obj = _loss(theta_init, None)
    best_theta = theta_init
    best_converged = False
    best_n_steps = 0

    for i in range(n_starts):
        if i == 0:
            theta_start = theta_init
        else:
            perturb = 0.1 * jax.random.normal(
                jax.random.fold_in(keys[i], i), theta_init.shape
            )
            theta_start = theta_init + perturb

        try:
            sol = optx.minimise(
                _loss, solver, theta_start, args=None,
                max_steps=max_steps, throw=False,
            )
            sol_value = sol.value
            converged = bool(sol.result == optx.RESULTS.successful)
            n_steps_i = int(sol.stats["num_steps"])
        except Exception:  # noqa: BLE001
            # Optimistix sometimes crashes on noisy objectives; fall back
            # to the starting point (which gives the best-known margin
            # for the default Alcubierre reproduction case).
            sol_value = theta_start
            converged = False
            n_steps_i = 0

        # Evaluate outside the try-block so we always have an obj_val
        obj_val = _loss(sol_value, None)
        if float(obj_val) < float(best_obj):
            best_obj = obj_val
            best_theta = sol_value
            best_converged = converged
            best_n_steps = n_steps_i

    # Defensive fallback: if nothing ever evaluated (e.g. total loss failure),
    # keep the input shape unchanged with a diagnostic infinity margin.
    if not jnp.isfinite(best_obj):
        best_obj = _loss(theta_init, None)
        best_theta = theta_init
        best_converged = False
        best_n_steps = 0

    best_values = _sigmoid_reparam(best_theta)
    best_shape = ShapeFunction(
        basis="spline",
        params={"knots": knots, "values": best_values},
        order=shape.order,
    )
    best_metric = ShapeFunctionMetric(best_shape, v_s=v_s_arr, strict=False)
    verdict = best_metric.verify_physical()

    report = OptimizationReport(
        converged=best_converged,
        final_margin=best_obj,
        n_steps=best_n_steps,
        physicality=verdict,
        strategy=strategy,
        n_starts=n_starts,
    )
    return best_metric, report
