"""Grid EC verification: Hawking-Ellis + eigenvalue + BFGS pipeline.

Strategy: classify each point, then for Type I take the algebraic
eigenvalue margins (which are exact for WEC, NEC, and SEC) and run the
optimizer in parallel for diagnostics in ``*_opt_margins``. Non-Type-I
points fall back to optimizer margins. The DEC margin always takes a
final ``min(WEC, DEC)`` because DEC implies WEC.

Eulerian-frame margins are exposed separately by
:func:`compute_eulerian_ec` for clean single-frame comparisons.

:func:`verify_point` uses host-side control flow on the Hawking-Ellis
type, so it is not JIT-compilable on the outer call. :func:`verify_grid`
handles vectorization by splitting the eigenvalue and optimizer branches
and merging with ``jnp.where``; the inner kernels are still vmapped.
"""
from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Float

from .classification import (
    classify_hawking_ellis,
    classify_with_solver,
    _standard_solver_unreliable_mask,
)
from .eigenvalue_checks import check_all
from .optimization import (
    optimize_point,
)
from .observer import compute_orthonormal_tetrad
from .types import ECGridResult, ECPointResult, ECSummary


def _classify_grid_batch(
    flat_T_mixed,
    flat_g,
    flat_T,
    *,
    solver: str,
):
    """Classify a flattened grid with optional auto generalized fallback."""
    if solver == "generalized":
        def _classify_gen(T_mixed_i, g_i, T_ab_i):
            return classify_hawking_ellis(
                T_mixed_i, g_i, solver="generalized", T_ab=T_ab_i,
            )
        return jax.vmap(_classify_gen)(flat_T_mixed, flat_g, flat_T)

    if solver == "standard":
        return jax.vmap(classify_hawking_ellis)(flat_T_mixed, flat_g)

    # auto: standard everywhere, generalized pencil only on unreliable points.
    #
    # We deliberately do NOT vmap a lax.cond here: under vmap, lax.cond becomes
    # a select that evaluates BOTH branches for every point, which would run the
    # (slow, sequential pure_callback) generalized pencil on the entire grid.
    # Instead, batch the generalized solve over just the unreliable subset and
    # scatter the results back with a single vectorized ``.at[idx].set`` -- this
    # removes the historical Python per-point loop and its N device->host
    # gathers while keeping the generalized solve restricted to the points that
    # need it. Byte-identical to the old host loop (same standard classification
    # at imag_rtol=3e-3, same _standard_solver_unreliable_mask decision at
    # imag_rtol=0.05, same per-point generalized classify on the covariant T).
    cls_std = jax.vmap(classify_hawking_ellis)(flat_T_mixed, flat_g)
    unreliable = np.asarray(
        _standard_solver_unreliable_mask(
            cls_std.he_type,
            cls_std.eigenvalues,
            cls_std.eigenvalues_imag,
        ),
        dtype=bool,
    )
    if not unreliable.any():
        return cls_std

    idx = jnp.asarray(np.where(unreliable)[0])

    def _classify_gen(T_mixed_i, g_i, T_ab_i):
        return classify_hawking_ellis(
            T_mixed_i, g_i, solver="generalized", T_ab=T_ab_i,
        )

    sub = jax.vmap(_classify_gen)(
        flat_T_mixed[idx], flat_g[idx], flat_T[idx]
    )

    return type(cls_std)(
        he_type=cls_std.he_type.at[idx].set(sub.he_type),
        eigenvalues=cls_std.eigenvalues.at[idx].set(sub.eigenvalues),
        eigenvectors=cls_std.eigenvectors.at[idx].set(sub.eigenvectors),
        rho=cls_std.rho.at[idx].set(sub.rho),
        pressures=cls_std.pressures.at[idx].set(sub.pressures),
        eigenvalues_imag=cls_std.eigenvalues_imag.at[idx].set(
            sub.eigenvalues_imag
        ),
        is_vacuum=cls_std.is_vacuum.at[idx].set(sub.is_vacuum),
    )


def _run_grid_optimization(
    flat_T,
    flat_g,
    subkeys,
    *,
    n_starts,
    zeta_max,
    strategy,
    batch_size,
    skip_type_i_optimization,
    he_types,
    warm_start,
    neighbor_fraction,
    starts,
):
    """Run per-point optimization, optionally skipping Type-I points."""
    n_points = flat_T.shape[0]
    nan = jnp.nan

    def _empty_opt_tuple():
        return (
            jnp.full(n_points, nan),
            jnp.full(n_points, nan),
            jnp.full(n_points, nan),
            jnp.full(n_points, nan),
            jnp.zeros((n_points, 4)),
            jnp.zeros((n_points, 3)),
            jnp.zeros(n_points),
            jnp.zeros(n_points),
            jnp.zeros(n_points),
            jnp.zeros(n_points),
            jnp.zeros(n_points),
            jnp.zeros(n_points),
            jnp.zeros(n_points),
            jnp.zeros(n_points),
        )

    def optimize_single(args):
        T_i, g_i, subkey, neighbor_obs = args
        opt = optimize_point(
            T_i, g_i,
            conditions=("nec", "wec", "sec", "dec"),
            n_starts=n_starts,
            zeta_max=zeta_max,
            key=subkey,
            strategy=strategy,
            warm_start=warm_start,
            neighbor_fraction=neighbor_fraction,
            starts=starts,
            neighbor_observer=neighbor_obs,
        )
        return (
            opt["nec"].margin,
            opt["wec"].margin,
            opt["sec"].margin,
            opt["dec"].margin,
            opt["wec"].worst_observer,
            opt["wec"].worst_params,
            opt["nec"].converged,
            opt["wec"].converged,
            opt["sec"].converged,
            opt["dec"].converged,
            opt["nec"].n_steps,
            opt["wec"].n_steps,
            opt["sec"].n_steps,
            opt["dec"].n_steps,
        )

    is_type_i = np.asarray(he_types == 1.0)
    if skip_type_i_optimization and is_type_i.all():
        return _empty_opt_tuple()

    if warm_start == "spatial_neighbor":
        results = []
        prev_worst = None
        for i in range(n_points):
            if skip_type_i_optimization and is_type_i[i]:
                results.append((
                    nan, nan, nan, nan,
                    jnp.zeros(4), jnp.zeros(3),
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                ))
                continue
            neighbor = prev_worst if i > 0 else None
            results.append(
                optimize_single((flat_T[i], flat_g[i], subkeys[i], neighbor))
            )
            prev_worst = results[-1][4]
        stacked = tuple(jnp.stack([r[k] for r in results], axis=0) for k in range(14))
        return stacked

    if skip_type_i_optimization and (~is_type_i).any():
        out = _empty_opt_tuple()
        idx = np.where(~is_type_i)[0]
        sub_T = flat_T[idx]
        sub_g = flat_g[idx]
        sub_keys = subkeys[idx]
        sub_results = jax.vmap(
            lambda args: optimize_single((args[0], args[1], args[2], None))
        )((sub_T, sub_g, sub_keys))

        idx_dev = jnp.asarray(idx)

        def _scatter(full, partial, indices):
            # Single vectorized scatter (full[indices[j]] = partial[j]) instead
            # of a Python loop of N full-array .at[i].set copies. Same result,
            # O(N) -> one fused op.
            return full.at[idx_dev].set(partial)

        nec_opt, wec_opt, sec_opt, dec_opt, worst_obs, worst_par = out[:6]
        nec_opt = _scatter(nec_opt, sub_results[0], idx)
        wec_opt = _scatter(wec_opt, sub_results[1], idx)
        sec_opt = _scatter(sec_opt, sub_results[2], idx)
        dec_opt = _scatter(dec_opt, sub_results[3], idx)
        worst_obs = _scatter(worst_obs, sub_results[4], idx)
        worst_par = _scatter(worst_par, sub_results[5], idx)
        nec_conv = _scatter(out[6], sub_results[6], idx)
        wec_conv = _scatter(out[7], sub_results[7], idx)
        sec_conv = _scatter(out[8], sub_results[8], idx)
        dec_conv = _scatter(out[9], sub_results[9], idx)
        nec_nsteps = _scatter(out[10], sub_results[10], idx)
        wec_nsteps = _scatter(out[11], sub_results[11], idx)
        sec_nsteps = _scatter(out[12], sub_results[12], idx)
        dec_nsteps = _scatter(out[13], sub_results[13], idx)
        return (
            nec_opt, wec_opt, sec_opt, dec_opt, worst_obs, worst_par,
            nec_conv, wec_conv, sec_conv, dec_conv,
            nec_nsteps, wec_nsteps, sec_nsteps, dec_nsteps,
        )

    args = (flat_T, flat_g, subkeys, jnp.full((n_points, 4), jnp.nan))
    if batch_size is not None:
        return lax.map(
            lambda a: optimize_single(a),
            args,
            batch_size=batch_size,
        )
    return jax.vmap(optimize_single)(args)


def verify_point(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    g_inv: Float[Array, "4 4"] | None = None,
    n_starts: int = 16,
    zeta_max: float = 5.0,
    key=None,
    *,
    solver: str = 'auto',
) -> ECPointResult:
    """Verify all energy conditions at a single spacetime point.

    Two-tier strategy:
    - Type I: eigenvalue margins (exact for WEC/NEC/SEC) plus parallel
      optimizer diagnostics in ``*_opt_margins``. ``dec_margins`` uses
      ``min(wec_margin, dec_eigenvalue_margin)``.
    - Non-Type-I: optimizer margins only.

    Parameters
    ----------
    T_ab : Float[Array, "4 4"]
        Covariant stress-energy tensor at a single point.
    g_ab : Float[Array, "4 4"]
        Covariant metric tensor at the same point.
    g_inv : Float[Array, "4 4"] or None
        Inverse metric. Computed from ``g_ab`` if not provided.
    n_starts : int
        Multi-start count for optimization.
    zeta_max : float
        Maximum rapidity.
    key : PRNGKey or None
        Random key for optimization initial conditions.
    solver : {'standard', 'generalized', 'auto'}, keyword-only
        Eigenvalue backend forwarded to :func:`classify_with_solver`.
        ``'auto'`` (default) uses standard eig with generalized fallback
        on ill-conditioned pencils.

    Returns
    -------
    ECPointResult
        Complete per-point energy condition result.

    Note
    ----
    Uses Python control flow on JAX values - not ``jit``-able. For
    batched evaluation, use :func:`verify_grid`.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    if g_inv is None:
        g_inv = jnp.linalg.inv(g_ab)

    T_mixed = jnp.einsum("ac,cb->ab", g_inv, T_ab)
    cls = classify_with_solver(
        T_mixed, g_ab, T_ab,
        solver=solver,
    )
    he_type = cls.he_type
    rho = cls.rho
    pressures = cls.pressures

    opt_results = optimize_point(
        T_ab, g_ab,
        conditions=("nec", "wec", "sec", "dec"),
        n_starts=n_starts,
        zeta_max=zeta_max,
        key=key,
    )

    nec_opt = opt_results["nec"].margin
    wec_opt = opt_results["wec"].margin
    sec_opt = opt_results["sec"].margin
    dec_opt = opt_results["dec"].margin

    # WEC worst observer is the timelike one we expose downstream.
    worst_observer = opt_results["wec"].worst_observer
    worst_params = opt_results["wec"].worst_params

    he_type_int = int(he_type)
    if he_type_int == 1:
        # Type I: algebraic NEC/WEC/SEC margins are exact. The algebraic
        # DEC proxy ``rho - max|p_i|`` is necessary only for flux
        # causality, so combine with the optimizer for tightness.
        nec_eig, wec_eig, sec_eig, dec_eig = check_all(rho, pressures)
        nec_margin = nec_eig
        wec_margin = wec_eig
        sec_margin = sec_eig
        dec_margin = jnp.minimum(dec_eig, dec_opt)
    else:
        nec_margin = nec_opt
        wec_margin = wec_opt
        sec_margin = sec_opt
        dec_margin = dec_opt

    # DEC implies WEC; the WEC slack can dominate, so re-merge here.
    dec_margin = jnp.minimum(wec_margin, dec_margin)

    return ECPointResult(
        he_type=he_type,
        eigenvalues=cls.eigenvalues,
        rho=rho,
        pressures=pressures,
        nec_margin=nec_margin,
        wec_margin=wec_margin,
        sec_margin=sec_margin,
        dec_margin=dec_margin,
        worst_observer=worst_observer,
        worst_params=worst_params,
        nec_opt_margin=nec_opt,
        wec_opt_margin=wec_opt,
        sec_opt_margin=sec_opt,
        dec_opt_margin=dec_opt,
    )


def _compute_summary(margins: Float[Array, "N"], atol: float = 1e-10) -> ECSummary:
    """Compute per-condition summary statistics over flattened margins."""
    n = margins.shape[0]
    violated = margins < -atol
    frac = jnp.sum(violated.astype(jnp.float64)) / n
    violation_magnitudes = jnp.where(violated, jnp.abs(margins), 0.0)
    max_viol = jnp.max(violation_magnitudes)
    min_margin = jnp.min(margins)
    return ECSummary(
        fraction_violated=frac,
        max_violation=max_viol,
        min_margin=min_margin,
    )


def verify_grid(
    T_field: Float[Array, "... 4 4"],
    g_field: Float[Array, "... 4 4"],
    g_inv_field: Float[Array, "... 4 4"] | None = None,
    n_starts: int = 16,
    zeta_max: float = 5.0,
    batch_size: int | None = None,
    key=None,
    compute_eulerian: bool = False,
    *,
    solver: str = 'auto',
    strategy: str = "tanh",
    skip_type_i_optimization: bool = True,
    warm_start: str = "cold",
    neighbor_fraction: float = 1.0 / 16.0,
    starts: str = "axis+gaussian",
) -> ECGridResult:
    """Verify energy conditions across an entire grid.

    Flatten-vmap-reshape pattern: classification and eigenvalue checks
    are vmapped; optimization uses lax.map (or vmap) for memory safety.

    Parameters
    ----------
    T_field : Float[Array, "... 4 4"]
        Stress-energy tensor field, shape ``(*grid_shape, 4, 4)``.
    g_field : Float[Array, "... 4 4"]
        Metric tensor field, shape ``(*grid_shape, 4, 4)``.
    g_inv_field : Float[Array, "... 4 4"] or None
        Inverse metric field. Computed if not provided.
    n_starts : int
        Multi-start count for optimization.
    zeta_max : float
        Maximum rapidity.
    batch_size : int or None
        If set, use ``lax.map`` with this batch size for memory-safe
        processing. If None, use ``jax.vmap``.
    key : PRNGKey or None
        Random key for optimization.
    compute_eulerian : bool
        If True, also compute Eulerian-frame EC at each point and take
        the worse margin. Default False.
    solver : {'standard', 'generalized', 'auto'}, keyword-only
        Eigenvalue backend forwarded to :func:`classify_with_solver`.
        ``'auto'`` (default) uses standard eig with generalized fallback
        on ill-conditioned pencils.
    skip_type_i_optimization : bool
        If True (default), skip BFGS on Type-I points where algebraic
        eigenvalue margins are exact. Optimizer diagnostics remain NaN
        on skipped points.
    warm_start, neighbor_fraction, starts
        Forwarded to :func:`optimize_point`. ``warm_start='spatial_neighbor'``
        seeds each point with the previous WEC observer (C-order flatten).

    Returns
    -------
    ECGridResult
        Per-point fields reshaped to ``(*grid_shape, ...)``, plus
        summary statistics for each condition.

    Note
    ----
    Not ``jit``-able: type-census and imag-eigenvalue summary stats use
    ``int``/``float`` on traced values. Inner eigenvalue and optimization
    paths are still vmapped / JIT-compiled.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    grid_shape = T_field.shape[:-2]
    # Pure-host product of static shape dims; avoids a device round-trip.
    n_points = int(np.prod(grid_shape))

    # Flatten to (N, 4, 4)
    flat_T = T_field.reshape(n_points, 4, 4)
    flat_g = g_field.reshape(n_points, 4, 4)

    if g_inv_field is not None:
        flat_g_inv = g_inv_field.reshape(n_points, 4, 4)
    else:
        flat_g_inv = jax.vmap(jnp.linalg.inv)(flat_g)

    flat_T_mixed = jax.vmap(jnp.matmul)(flat_g_inv, flat_T)
    cls_results = _classify_grid_batch(
        flat_T_mixed, flat_g, flat_T, solver=solver,
    )

    he_types = cls_results.he_type
    eigenvalues = cls_results.eigenvalues
    rho_all = cls_results.rho
    pressures_all = cls_results.pressures

    nec_eig, wec_eig, sec_eig, dec_eig = jax.vmap(check_all)(rho_all, pressures_all)

    subkeys = jax.random.split(key, n_points)
    opt_results = _run_grid_optimization(
        flat_T, flat_g, subkeys,
        n_starts=n_starts,
        zeta_max=zeta_max,
        strategy=strategy,
        batch_size=batch_size,
        skip_type_i_optimization=skip_type_i_optimization,
        he_types=he_types,
        warm_start=warm_start,
        neighbor_fraction=neighbor_fraction,
        starts=starts,
    )

    (nec_opt, wec_opt, sec_opt, dec_opt, worst_obs, worst_par,
     nec_conv, wec_conv, sec_conv, dec_conv,
     nec_nsteps, wec_nsteps, sec_nsteps, dec_nsteps) = opt_results

    is_type_i = (he_types == 1.0)

    n_non_type_i = int(jnp.sum(~is_type_i))
    if n_non_type_i > 0:
        import warnings
        warnings.warn(
            f"{n_non_type_i} of {n_points} grid points are non-Type-I. "
            f"DEC future-directedness is only guaranteed for Type I.",
            stacklevel=2,
        )

    def merge_margins(eig_m, opt_m):
        return jnp.where(is_type_i, eig_m, opt_m)

    nec_margins = merge_margins(nec_eig, nec_opt)
    wec_margins = merge_margins(wec_eig, wec_opt)
    sec_margins = merge_margins(sec_eig, sec_opt)
    # Type-I DEC: min(algebraic proxy, optimizer) when the optimizer ran.
    dec_opt_finite = jnp.isfinite(dec_opt)
    dec_margins_type_i = jnp.where(
        dec_opt_finite, jnp.minimum(dec_eig, dec_opt), dec_eig
    )
    dec_margins = jnp.where(is_type_i, dec_margins_type_i, dec_opt)
    dec_margins = jnp.minimum(wec_margins, dec_margins)

    if compute_eulerian:
        eulerian_results = jax.vmap(_eulerian_ec_point)(flat_T, flat_g, flat_g_inv)
        nec_margins = jnp.minimum(nec_margins, eulerian_results["nec"])
        wec_margins = jnp.minimum(wec_margins, eulerian_results["wec"])
        sec_margins = jnp.minimum(sec_margins, eulerian_results["sec"])
        dec_margins = jnp.minimum(dec_margins, eulerian_results["dec"])

    nec_summary = _compute_summary(nec_margins)
    wec_summary = _compute_summary(wec_margins)
    sec_summary = _compute_summary(sec_margins)
    dec_summary = _compute_summary(dec_margins)

    n_type_i = int(jnp.sum(he_types == 1.0))
    n_type_ii = int(jnp.sum(he_types == 2.0))
    n_type_iii = int(jnp.sum(he_types == 3.0))
    n_type_iv = int(jnp.sum(he_types == 4.0))
    # Near-vacuum is a subset of n_type_i; reported separately.
    n_vacuum = int(jnp.sum(cls_results.is_vacuum))
    # nanmax: a NaN-sanitized eigenvalue must not poison the grid-wide
    # imaginary-part diagnostic (it is a summary, not a certified margin).
    max_imag = float(jnp.nanmax(jnp.abs(cls_results.eigenvalues_imag)))

    def reshape(arr):
        return arr.reshape(*grid_shape, *arr.shape[1:])

    return ECGridResult(
        he_types=reshape(he_types),
        eigenvalues=reshape(eigenvalues),
        rho=reshape(rho_all),
        pressures=reshape(pressures_all),
        nec_margins=reshape(nec_margins),
        wec_margins=reshape(wec_margins),
        sec_margins=reshape(sec_margins),
        dec_margins=reshape(dec_margins),
        worst_observers=reshape(worst_obs),
        worst_params=reshape(worst_par),
        nec_summary=nec_summary,
        wec_summary=wec_summary,
        sec_summary=sec_summary,
        dec_summary=dec_summary,
        nec_opt_margins=reshape(nec_opt),
        wec_opt_margins=reshape(wec_opt),
        sec_opt_margins=reshape(sec_opt),
        dec_opt_margins=reshape(dec_opt),
        n_type_i=n_type_i,
        n_type_ii=n_type_ii,
        n_type_iii=n_type_iii,
        n_type_iv=n_type_iv,
        n_vacuum=n_vacuum,
        max_imag_eigenvalue=max_imag,
        nec_opt_converged=reshape(nec_conv),
        wec_opt_converged=reshape(wec_conv),
        sec_opt_converged=reshape(sec_conv),
        dec_opt_converged=reshape(dec_conv),
        nec_opt_n_steps=reshape(nec_nsteps),
        wec_opt_n_steps=reshape(wec_nsteps),
        sec_opt_n_steps=reshape(sec_nsteps),
        dec_opt_n_steps=reshape(dec_nsteps),
    )


def _eulerian_ec_point(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    g_inv: Float[Array, "4 4"],
) -> dict[str, Float[Array, ""]]:
    """Eulerian-frame EC margins at a single point (vmap-safe)."""
    # Floor -g^{00} so closed-timelike-curve noise does not emit NaN.
    alpha = 1.0 / jnp.sqrt(jnp.maximum(-g_inv[0, 0], 1e-30))
    n_up = jnp.zeros(4)
    n_up = n_up.at[0].set(1.0 / alpha)
    n_up = n_up.at[1].set(-g_inv[0, 1] * alpha)
    n_up = n_up.at[2].set(-g_inv[0, 2] * alpha)
    n_up = n_up.at[3].set(-g_inv[0, 3] * alpha)

    wec_margin = jnp.einsum("a,ab,b->", n_up, T_ab, n_up)

    tetrad = compute_orthonormal_tetrad(g_ab)
    spatial = tetrad[1:4]
    k_all = jnp.concatenate(
        [tetrad[0][None, :] + spatial, tetrad[0][None, :] - spatial], axis=0
    )
    nec_vals = jnp.einsum("ia,ab,ib->i", k_all, T_ab, k_all)
    nec_margin = jnp.min(nec_vals)

    T_trace = jnp.einsum("ab,ab->", g_inv, T_ab)
    sec_tensor = T_ab - 0.5 * T_trace * g_ab
    sec_margin = jnp.einsum("a,ab,b->", n_up, sec_tensor, n_up)

    T_mixed = g_inv @ T_ab
    j = -jnp.einsum("ac,c->a", T_mixed, n_up)
    j_norm_sq = jnp.einsum("a,ab,b->", j, g_ab, j)
    dec_flux_margin = -j_norm_sq

    n_down = jnp.einsum("ab,b->a", g_ab, n_up)
    dec_future_margin = -jnp.einsum("a,a->", j, n_down)

    dec_margin = jnp.minimum(wec_margin, jnp.minimum(dec_flux_margin, dec_future_margin))

    return {"nec": nec_margin, "wec": wec_margin, "sec": sec_margin, "dec": dec_margin}


def compute_eulerian_ec(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    g_inv: Float[Array, "4 4"] | None = None,
) -> dict[str, Float[Array, ""]]:
    """Energy-condition margins for the Eulerian observer only.

    The Eulerian observer is the unit normal to constant-time spatial
    hypersurfaces, ``n^a = (1/alpha, -beta^i/alpha)``. Useful as a clean
    single-frame baseline against the observer-robust margins.

    Parameters
    ----------
    T_ab, g_ab : Float[Array, "4 4"]
        Covariant stress-energy and metric tensors at a single point.
    g_inv : Float[Array, "4 4"] or None
        Inverse metric. Computed if omitted.

    Returns
    -------
    dict
        Maps ``"nec"``, ``"wec"``, ``"sec"``, ``"dec"`` to signed margins.
    """
    if g_inv is None:
        g_inv = jnp.linalg.inv(g_ab)
    return _eulerian_ec_point(T_ab, g_ab, g_inv)


def anec_integrand(
    T_ab: Float[Array, "4 4"],
    k: Float[Array, "4"],
) -> Float[Array, ""]:
    """Pointwise ANEC integrand ``T_{ab} k^a k^b`` at a single point.

    For the full line integral along a null geodesic, see
    :func:`warpax.averaged.anec`.
    """
    return jnp.einsum("a,ab,b->", k, T_ab, k)

