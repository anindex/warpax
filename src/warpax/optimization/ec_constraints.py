"""EC constraint enforcement for shell optimization.

Soft penalty: sum of softplus(-margin)^2 for gradient-based optimization.
Hard feasibility: binary certification with the full warpax EC pipeline.

The per-probe BFGS multistart is wrapped in :func:`equinox.filter_jit`
so the inner trace is compiled once and reused across probes.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..energy_conditions.optimization import optimize_dec, optimize_nec, optimize_wec
from ..geometry.geometry import compute_curvature_chain
from ..geometry.metric import MetricSpecification


class ECFeasibilityResult(NamedTuple):
    """Result of a hard EC feasibility check.

    feasible : True iff all conditions satisfied at all probe points.
    margins : per-condition margin arrays, shape (N_probes,).
    worst_point_idx : index of the worst-violating probe point.
    worst_condition : name of the worst-violated condition.
    worst_margin : value of the worst margin (negative = violated).
    """

    feasible: bool
    margins: dict[str, Float[Array, "N"]]
    worst_point_idx: int
    worst_condition: str
    worst_margin: float


_OPTIMIZER_MAP = {"nec": optimize_nec, "wec": optimize_wec, "dec": optimize_dec}


def _probe_coords(r_probes: Float[Array, "N"]) -> Float[Array, "N 4"]:
    """Build (N, 4) coord batch on the x-axis at t=0 from radial probes."""
    N = r_probes.shape[0]
    zeros = jnp.zeros_like(r_probes)
    return jnp.stack([zeros, r_probes, zeros, zeros], axis=-1).reshape(N, 4)


@eqx.filter_jit
def _probe_T_g(
    metric: MetricSpecification,
    r_probes: Float[Array, "N"],
) -> tuple[Float[Array, "N 4 4"], Float[Array, "N 4 4"]]:
    """Vmap the curvature chain across radial probes; NaN-clean T."""
    coords_batch = _probe_coords(r_probes)
    cc_batch = jax.vmap(lambda c: compute_curvature_chain(metric, c))(coords_batch)
    T = jnp.where(jnp.isnan(cc_batch.stress_energy), 0.0, cc_batch.stress_energy)
    return T, cc_batch.metric


@partial(jax.jit, static_argnames=("conditions", "n_starts"))
def _ec_margins_at_point(
    T: Float[Array, "4 4"],
    g: Float[Array, "4 4"],
    conditions: tuple[str, ...],
    n_starts: int,
    key=None,
) -> dict[str, Float[Array, ""]]:
    """Per-point dict of EC margins; traced once, cached across probes.

    Callers thread a per-point key; each condition gets a distinct
    ``fold_in`` of it. ``key=None`` falls back to ``PRNGKey(42)``
    (legacy callers), still with per-condition decorrelation.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    out = {}
    for idx, cond in enumerate(conditions):
        cond_key = jax.random.fold_in(key, idx)
        out[cond] = _OPTIMIZER_MAP[cond](T, g, n_starts=n_starts, key=cond_key).margin
    return out


def ec_penalty(
    metric: MetricSpecification,
    r_probes: Float[Array, "N"],
    *,
    conditions: tuple[str, ...] = ("nec", "wec", "dec"),
    n_starts: int = 4,
    key=None,
) -> Float[Array, ""]:
    """Soft EC penalty: ``sum_i sum_c softplus(-margin_{i,c})^2``.

    Vectorizes the curvature chain across probes and JIT-caches the
    per-point margin evaluation, so repeated invocations during
    optimization re-use the compiled trace instead of re-tracing.

    ``key`` seeds the per-point multistart randomness; ``None`` keeps
    the deterministic default ``PRNGKey(42)``. Each probe point gets a
    distinct ``fold_in(key, i)`` so multistarts decorrelate across
    points.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    T_batch, g_batch = _probe_T_g(metric, r_probes)
    total_penalty = jnp.float64(0.0)
    for i in range(r_probes.shape[0]):
        point_key = jax.random.fold_in(key, i)
        margins = _ec_margins_at_point(
            T_batch[i], g_batch[i], conditions, n_starts, point_key
        )
        for cond in conditions:
            total_penalty = total_penalty + jax.nn.softplus(-margins[cond]) ** 2

    return total_penalty


def ec_feasibility_check(
    metric: MetricSpecification,
    r_probes: Float[Array, "N"],
    *,
    conditions: tuple[str, ...] = ("nec", "wec", "dec"),
    n_starts: int = 16,
    key=None,
) -> ECFeasibilityResult:
    """Hard EC check with per-point, per-condition margins.

    Uses n_starts=16 by default for certification quality.

    ``key`` seeds the per-point multistart randomness; ``None`` keeps
    the deterministic default ``PRNGKey(42)``. Each probe point gets a
    distinct ``fold_in(key, i)``.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    T_batch, g_batch = _probe_T_g(metric, r_probes)
    margins = {cond: [] for cond in conditions}
    for i in range(r_probes.shape[0]):
        point_key = jax.random.fold_in(key, i)
        per_point = _ec_margins_at_point(
            T_batch[i], g_batch[i], conditions, n_starts, point_key
        )
        for cond in conditions:
            margins[cond].append(per_point[cond])
    margins = {cond: jnp.stack(vals) for cond, vals in margins.items()}

    worst_condition = conditions[0]
    worst_margin = float("inf")
    worst_idx = 0

    for cond in conditions:
        min_val = float(jnp.min(margins[cond]))
        if min_val < worst_margin:
            worst_margin = min_val
            worst_condition = cond
            worst_idx = int(jnp.argmin(margins[cond]))

    feasible = all(float(jnp.min(margins[c])) >= 0.0 for c in conditions)

    return ECFeasibilityResult(
        feasible=feasible,
        margins=margins,
        worst_point_idx=worst_idx,
        worst_condition=worst_condition,
        worst_margin=worst_margin,
    )


class FrameFreeFeasibility(NamedTuple):
    """Frame-free (Hawking--Ellis) EC feasibility at a set of radial probes.

    feasible : True iff every probe is Hawking--Ellis Type I with non-negative
        cap-free NEC/WEC/DEC slacks. A Type-IV probe (no rest frame) is
        infeasible by construction.
    worst_margin : signed severity for the heatmap. At Type-I probes it is the
        worst cap-free eigenvalue slack; where any probe is Type-IV it is the
        negative of the largest imaginary-eigenvalue scale (an invariant,
        frame-free severity proxy, not a capped optimizer value).
    n_type_iv : number of non-Type-I probes (Type II/III/IV).
    """

    feasible: bool
    worst_margin: float
    n_type_iv: int
    n_type_i: int
    n_total: int
    max_imag: float


def ec_feasibility_frame_free(
    metric: MetricSpecification,
    r_probes: Float[Array, "N"],
    *,
    conditions: tuple[str, ...] = ("nec", "wec", "dec"),
) -> FrameFreeFeasibility:
    """Frame-free EC feasibility via the Hawking--Ellis certifier.

    Unlike :func:`ec_feasibility_check` (rapidity-capped BFGS optimizer), the
    verdict here is observer-independent: the algebraic Type of ``T^a_b`` plus
    the cap-free Type-I eigenvalue slacks (Hawking & Ellis 1973). No Eulerian
    normal, no timelike tetrad, valid at all warp speeds.
    """
    from ..energy_conditions.frame_free import certify_grid_frame_free

    T_batch, g_batch = _probe_T_g(metric, r_probes)
    res = certify_grid_frame_free(T_batch, g_batch)
    he = jnp.asarray(res.he_types)
    is_type_i = he == 1.0
    n_total = int(he.size)
    n_type_i = int(jnp.sum(is_type_i))
    n_type_iv = n_total - n_type_i

    margin_cols = {"nec": res.nec_margins, "wec": res.wec_margins, "dec": res.dec_margins}
    # Worst cap-free Type-I slack over the requested conditions (NaN at non-Type-I).
    stacked = jnp.stack([margin_cols[c] for c in conditions])  # (C, N)
    typeI_worst = float(jnp.nanmin(jnp.where(is_type_i[None, :], stacked, jnp.nan)))
    if not jnp.isfinite(typeI_worst):
        typeI_worst = float("nan")
    max_imag = float(jnp.nanmax(jnp.abs(jnp.asarray(res.eigenvalues_imag))))

    feasible = bool(n_type_iv == 0 and jnp.isfinite(typeI_worst) and typeI_worst >= 0.0)
    # Heatmap severity: Type-I slack if all Type-I, else negative imaginary scale.
    if n_type_iv > 0:
        worst_margin = -abs(max_imag)
    else:
        worst_margin = typeI_worst

    return FrameFreeFeasibility(
        feasible=feasible,
        worst_margin=worst_margin,
        n_type_iv=n_type_iv,
        n_type_i=n_type_i,
        n_total=n_total,
        max_imag=max_imag,
    )
