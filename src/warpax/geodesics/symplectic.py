"""Symplectic (canonical Hamiltonian) null/timelike geodesic integrator.

Geodesics are the trajectories of the Hamiltonian

.. math::

    H(x, p) = \\tfrac12 g^{ab}(x)\\, p_a p_b ,
    \\qquad \\dot x^a = g^{ab} p_b, \\quad
    \\dot p_a = -\\tfrac12 (\\partial_a g^{bc}) p_b p_c ,

whose value ``H = \\tfrac12 g(k, k)`` (with ``k^a = g^{ab} p_b``) is an exact
constant of motion. A symplectic integrator therefore keeps ``g(k,k)`` at its
initial value to machine precision -- in particular a *null* geodesic stays on
the null cone -- whereas the second-order geodesic-equation form integrated with
an adaptive Runge-Kutta scheme (:func:`warpax.geodesics.integrator.integrate_geodesic`)
lets ``g(k,k)`` drift secularly by the integrator tolerance. This is what makes a
rigorous, geodesic-integrated ANEC possible for strong-shift warp bubbles (see
:func:`warpax.averaged.anec.anec_rigorous`).

``H`` is *non-separable* (the kinetic term depends on ``x`` through
``g^{ab}(x)``), so plain leapfrog does not apply. We use the explicit
extended-phase-space scheme of Tao (PRE 94, 043303, 2016; arXiv:1609.02212),
realised with JAX autodiff in the spirit of the FANTASY integrator
(Christian & Chan, ApJ 2021; arXiv:2010.02237): two copies ``(x, p)`` and
``(\\bar x, \\bar p)`` are evolved by the exactly-integrable mixed Hamiltonians
``H_A = H(x, \\bar p)`` and ``H_B = H(\\bar x, p)`` plus an ``\\omega``-binding
rotation ``H_C`` that keeps the copies together. The metric gradient enters only
through ``g^{ab}`` and ``\\partial_c g^{ab}`` (one ``jax.jacfwd`` of the metric);
the full curvature chain is never invoked inside the inner loop.

References
----------
- Tao, "Explicit symplectic approximation of nonseparable Hamiltonians,"
  PRE 94, 043303 (2016).
- Christian & Chan, "FANTASY: A geodesic integrator with automatic
  differentiation," ApJ 919, 41 (2021).
"""
from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .initial_conditions import null_ic, timelike_ic

# Yoshida (1990) 4th-order triple-jump coefficients.
_YOSHIDA_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA_W0 = 1.0 - 2.0 * _YOSHIDA_W1

_NULL_TOL = 1e-10


class SymplecticGeodesicResult(NamedTuple):
    """Result of a symplectic geodesic integration.

    Attributes
    ----------
    ts : Float[Array, "M"]
        Saved affine-parameter values.
    positions : Float[Array, "M 4"]
        Coordinate positions ``x^a``.
    momenta : Float[Array, "M 4"]
        Canonical momenta ``p_a`` (lower index).
    velocities : Float[Array, "M 4"]
        Contravariant tangents ``k^a = g^{ab} p_b`` (for ANEC reuse).
    H_values : Float[Array, "M"]
        ``H = 1/2 g^{ab} p_a p_b = 1/2 g(k, k)`` at each saved point -- the
        conserved quantity. For a null geodesic this is the on-cone witness;
        ``max|2 H|`` is the worst ``|g(k,k)|`` along the path.
    complete : bool
        True iff the trajectory is finite throughout (no NaN/Inf) -- i.e. the
        initial condition was sub/luminal and integration did not blow up.
    termination_reason : str
        ``'complete'``, ``'superluminal'`` (NaN initial momentum), or ``'nan'``.
    """

    ts: Float[Array, "M"]
    positions: Float[Array, "M 4"]
    momenta: Float[Array, "M 4"]
    velocities: Float[Array, "M 4"]
    H_values: Float[Array, "M"]
    complete: bool
    termination_reason: str


def _g_inv_and_dginv(
    metric_fn, x: Float[Array, "4"]
) -> tuple[Float[Array, "4 4"], Float[Array, "4 4 4"]]:
    """Return ``g^{ab}(x)`` and ``\\partial_c g^{ab}(x)`` (derivative index last).

    Uses ``\\partial_c g^{ab} = -g^{ad} (\\partial_c g_{de}) g^{eb}`` so only the
    metric and one ``jax.jacfwd`` of it are needed (more stable than
    differentiating an explicit matrix inverse near a near-singular ``g``).
    """
    g = metric_fn(x)
    g_inv = jnp.linalg.inv(g)
    dg = jax.jacfwd(metric_fn)(x)  # dg[d, e, c] = d g_{de} / d x^c
    dginv = -jnp.einsum("ad,dec,eb->abc", g_inv, dg, g_inv)  # [a, b, c]
    return g_inv, dginv


def _phi_HA(state, delta, metric_fn):
    """Flow of ``H_A = 1/2 g^{ab}(x) p_bar_a p_bar_b`` for step ``delta``.

    Updates ``x_bar`` (drift using real ``x``) and ``p`` (force), holding
    ``x`` and ``p_bar`` fixed.
    """
    x, p, xb, pb = state
    g_inv, dginv = _g_inv_and_dginv(metric_fn, x)
    dxb = jnp.einsum("ab,b->a", g_inv, pb)               # dH_A/dp_bar
    dp = -0.5 * jnp.einsum("bca,b,c->a", dginv, pb, pb)  # -dH_A/dx
    return (x, p + delta * dp, xb + delta * dxb, pb)


def _phi_HB(state, delta, metric_fn):
    """Flow of ``H_B = 1/2 g^{ab}(x_bar) p_a p_b`` for step ``delta``.

    Updates ``x`` (drift using copy ``x_bar``) and ``p_bar`` (force), holding
    ``x_bar`` and ``p`` fixed.
    """
    x, p, xb, pb = state
    g_inv, dginv = _g_inv_and_dginv(metric_fn, xb)
    dx = jnp.einsum("ab,b->a", g_inv, p)                 # dH_B/dp
    dpb = -0.5 * jnp.einsum("bca,b,c->a", dginv, p, p)   # -dH_B/dx_bar
    return (x + delta * dx, p, xb, pb + delta * dpb)


def _phi_HC(state, delta, omega):
    """Flow of the binding Hamiltonian ``H_C`` (rotation of the copy mismatch)."""
    x, p, xb, pb = state
    ang = 2.0 * omega * delta
    ca, sa = jnp.cos(ang), jnp.sin(ang)
    sx, dx = x + xb, x - xb
    sp, dp = p + pb, p - pb
    dx_new = ca * dx + sa * dp
    dp_new = -sa * dx + ca * dp
    return (
        0.5 * (sx + dx_new),
        0.5 * (sp + dp_new),
        0.5 * (sx - dx_new),
        0.5 * (sp - dp_new),
    )


def _strang_step(state, delta, omega, metric_fn):
    """Symmetric 2nd-order step: A(d/2) B(d/2) C(d) B(d/2) A(d/2)."""
    state = _phi_HA(state, delta / 2.0, metric_fn)
    state = _phi_HB(state, delta / 2.0, metric_fn)
    state = _phi_HC(state, delta, omega)
    state = _phi_HB(state, delta / 2.0, metric_fn)
    state = _phi_HA(state, delta / 2.0, metric_fn)
    return state


def _yoshida4_step(state, delta, omega, metric_fn):
    """4th-order Yoshida triple-jump of the symmetric Strang step."""
    state = _strang_step(state, _YOSHIDA_W1 * delta, omega, metric_fn)
    state = _strang_step(state, _YOSHIDA_W0 * delta, omega, metric_fn)
    state = _strang_step(state, _YOSHIDA_W1 * delta, omega, metric_fn)
    return state


def _half_H(g_inv, p):
    return 0.5 * jnp.einsum("ab,a,b->", g_inv, p, p)


@eqx.filter_jit
def _integrate_core(
    metric_fn,
    x0: Float[Array, "4"],
    p0: Float[Array, "4"],
    lam0: float,
    lam1: float,
    omega: float,
    *,
    num_steps: int,
    num_save: int,
    order: int,
):
    """Fixed-step symplectic integration; returns saved arrays (jit core)."""
    delta = (lam1 - lam0) / num_steps
    step_fn = _strang_step if order == 2 else _yoshida4_step

    def scan_body(state, _):
        new_state = step_fn(state, delta, omega, metric_fn)
        x, p, _xb, _pb = new_state
        return new_state, (x, p)

    init = (x0, p0, x0, p0)  # both copies start equal
    _, (xs, ps) = jax.lax.scan(scan_body, init, xs=None, length=num_steps)

    # Prepend the initial point, then subsample to num_save evenly.
    xs = jnp.concatenate([x0[None, :], xs], axis=0)   # (num_steps+1, 4)
    ps = jnp.concatenate([p0[None, :], ps], axis=0)
    lam_full = jnp.linspace(lam0, lam1, num_steps + 1)
    save_idx = jnp.linspace(0, num_steps, num_save).astype(jnp.int32)

    xs_s = xs[save_idx]
    ps_s = ps[save_idx]
    ts_s = lam_full[save_idx]

    def per_point(x, p):
        g_inv = jnp.linalg.inv(metric_fn(x))
        k = jnp.einsum("ab,b->a", g_inv, p)
        H = _half_H(g_inv, p)
        return k, H

    ks, Hs = jax.vmap(per_point)(xs_s, ps_s)
    return ts_s, xs_s, ps_s, ks, Hs


def null_ic_canonical(
    metric_fn,
    x0: Float[Array, "4"],
    n_spatial: Float[Array, "3"],
) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
    """Null initial ``(x0, p0)`` in canonical (lower-index momentum) form.

    Reuses :func:`null_ic` (which carries the superluminal NaN sentinel) and
    lowers the index, ``p_a = g_{ab} k^b``.
    """
    x0, k0 = null_ic(metric_fn, x0, n_spatial)
    p0 = jnp.einsum("ab,b->a", metric_fn(x0), k0)
    return x0, p0


def timelike_ic_canonical(
    metric_fn,
    x0: Float[Array, "4"],
    v_spatial: Float[Array, "3"],
) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
    """Timelike initial ``(x0, p0)`` in canonical form (``p_a = g_{ab} v^b``)."""
    x0, v0 = timelike_ic(metric_fn, x0, v_spatial)
    p0 = jnp.einsum("ab,b->a", metric_fn(x0), v0)
    return x0, p0


def integrate_geodesic_symplectic(
    metric_fn,
    x0: Float[Array, "4"],
    p0: Float[Array, "4"],
    affine_bounds: tuple[float, float] = (-30.0, 30.0),
    *,
    num_steps: int = 8192,
    num_save: int = 512,
    order: int = 4,
    omega: float = 1.0,
) -> SymplecticGeodesicResult:
    """Integrate a geodesic with the extended-phase-space symplectic scheme.

    Empirically, on a long crossing of a large warp bubble
    (Alcubierre ``v_s=0.1, R=20, sigma=2`` over affine span 60) the adaptive
    Tsit5 integrator drifts off the null cone to ``max|g(k,k)| ~ 0.2``, while
    this scheme at ``num_steps=8192, omega=1, order=4`` holds it at ``~1e-10``
    -- the difference that makes a rigorous geodesic-integrated ANEC possible.

    Parameters
    ----------
    metric_fn : MetricSpecification
        Spacetime metric (Equinox module), ``coords (4,) -> g_ab (4,4)``.
    x0 : Float[Array, "4"]
        Initial position.
    p0 : Float[Array, "4"]
        Initial canonical momentum ``p_a`` (e.g. from :func:`null_ic_canonical`).
        A non-finite ``p0`` (superluminal null IC) yields an all-NaN result with
        ``termination_reason='superluminal'`` rather than garbage.
    affine_bounds : tuple[float, float]
        ``(lam0, lam1)`` integration interval.
    num_steps : int
        Number of fixed symplectic steps (static; keys the compile cache).
    num_save : int
        Number of evenly-spaced saved points (static).
    order : int
        ``2`` (Strang) or ``4`` (Yoshida); default 4.
    omega : float
        Tao binding strength for the extended phase space (dynamic; tune for
        strong-shift metrics). Larger ``omega`` binds the two copies more
        tightly; the conserved ``H_values`` witness is the objective acceptance
        criterion regardless of ``omega``.

    Returns
    -------
    SymplecticGeodesicResult
    """
    if order not in (2, 4):
        raise ValueError(f"order must be 2 or 4; got {order}")

    finite_ic = bool(jnp.all(jnp.isfinite(p0)) & jnp.all(jnp.isfinite(x0)))
    if not finite_ic:
        nan_v = jnp.full((num_save, 4), jnp.nan)
        nan_s = jnp.full((num_save,), jnp.nan)
        return SymplecticGeodesicResult(
            ts=jnp.linspace(affine_bounds[0], affine_bounds[1], num_save),
            positions=nan_v, momenta=nan_v, velocities=nan_v, H_values=nan_s,
            complete=False, termination_reason="superluminal",
        )

    ts, xs, ps, ks, Hs = _integrate_core(
        metric_fn, x0, p0, affine_bounds[0], affine_bounds[1], omega,
        num_steps=num_steps, num_save=num_save, order=order,
    )

    complete = bool(jnp.all(jnp.isfinite(xs)) & jnp.all(jnp.isfinite(ks)))
    reason = "complete" if complete else "nan"
    return SymplecticGeodesicResult(
        ts=ts, positions=xs, momenta=ps, velocities=ks, H_values=Hs,
        complete=complete, termination_reason=reason,
    )


def integrate_geodesic_symplectic_family(
    metric_fn,
    x0_batch: Float[Array, "N 4"],
    p0_batch: Float[Array, "N 4"],
    affine_bounds: tuple[float, float] = (-30.0, 30.0),
    *,
    num_steps: int = 8192,
    num_save: int = 512,
    order: int = 4,
    omega: float = 1.0,
):
    """Vectorized batch of symplectic geodesics sharing one metric.

    Returns ``(ts, positions, momenta, velocities, H_values)`` with a leading
    batch axis. (The host-side ``complete``/``termination_reason`` flags are not
    returned in the batched form; check ``jnp.isfinite`` per row.)
    """
    def core(x0, p0):
        return _integrate_core(
            metric_fn, x0, p0, affine_bounds[0], affine_bounds[1], omega,
            num_steps=num_steps, num_save=num_save, order=order,
        )

    return jax.vmap(core)(x0_batch, p0_batch)
