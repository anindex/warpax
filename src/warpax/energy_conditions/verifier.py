"""Two-tier energy condition verification orchestrator (pure JAX).

Ties together Hawking-Ellis classification, eigenvalue checks, and
Optimistix BFGS optimization into a complete grid-level pipeline.

Strategy
--------
1. Classify T^a_b via Hawking-Ellis at every grid point (vmapped).
2. For Type I points: fast eigenvalue algebraic checks.
3. For ALL points (including Type I): optimization over observer space
   to find worst-case margins.  For Type I, the final margin is the
   worse (smaller) of eigenvalue and optimization results.
4. Eulerian-frame EC results are computed SEPARATELY (not baked into
   optimization) via ``compute_eulerian_ec``, enabling clean comparison
   for the paper.

The per-point function ``verify_point`` uses Python control flow
(``if he_type == 1``) and is therefore NOT directly vmappable.
The grid version ``verify_grid`` handles vectorization by splitting
eigenvalue and optimization paths and merging results with ``jnp.where``.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from .classification import classify_hawking_ellis
from .eigenvalue_checks import check_all
from .optimization import (
    OptimizationResult,
    optimize_dec,
    optimize_nec,
    optimize_point,
    optimize_sec,
    optimize_wec,
)
from .observer import compute_orthonormal_tetrad, null_from_angles, timelike_from_rapidity
from .types import ECGridResult, ECPointResult, ECSummary


# ---------------------------------------------------------------------------
# Public API: per-point verification
# ---------------------------------------------------------------------------


def verify_point(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    g_inv: Float[Array, "4 4"] | None = None,
    n_starts: int = 16,
    zeta_max: float = 5.0,
    key=None,
) -> ECPointResult:
    """Verify all energy conditions at a single spacetime point.

    Two-tier strategy:
    - Type I: eigenvalue margins (fast) + optimization (validation).
      Final margin = min(eigenvalue, optimization) for each condition.
    - Non-Type-I: optimization only.

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

    Returns
    -------
    ECPointResult
        Complete per-point energy condition result.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    if g_inv is None:
        g_inv = jnp.linalg.inv(g_ab)

    # Classify
    T_mixed = g_inv @ T_ab
    cls = classify_hawking_ellis(T_mixed, g_ab)
    he_type = cls.he_type
    rho = cls.rho
    pressures = cls.pressures

    # Optimisation for ALL types
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

    # Track worst observer from WEC (most physically meaningful for violations)
    worst_observer = opt_results["wec"].worst_observer
    worst_params = opt_results["wec"].worst_params

    # For Type I: algebraic eigenvalue margins are exact (authoritative for
    # violation detection).  Optimizer provides worst-observer parameters
    # and zeta_max-capped severity diagnostics.
    he_type_int = int(he_type)
    if he_type_int == 1:
        nec_eig, wec_eig, sec_eig, dec_eig = check_all(rho, pressures)
        nec_margin = nec_eig
        wec_margin = wec_eig
        sec_margin = sec_eig
        dec_margin = dec_eig
    else:
        nec_margin = nec_opt
        wec_margin = wec_opt
        sec_margin = sec_opt
        dec_margin = dec_opt

    # DEC requires both causal flux AND non-negative energy density (WEC).
    # The optimizer checks flux only; incorporate the WEC margin.
    dec_margin = jnp.minimum(wec_margin, dec_margin)

    # Select worst observer from WEC optimizer (timelike condition with
    # the most physically meaningful observer interpretation).
    worst_observer = opt_results["wec"].worst_observer
    worst_params = opt_results["wec"].worst_params

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
    )


# ---------------------------------------------------------------------------
# Public API: grid verification
# ---------------------------------------------------------------------------


def _compute_summary(margins: Float[Array, "N"], atol: float = 1e-10) -> ECSummary:
    """Compute per-condition summary statistics over flattened margins."""
    n = margins.shape[0]
    violated = margins < -atol
    frac = jnp.sum(violated.astype(jnp.float64)) / n
    # max violation magnitude among violated points (0 if none)
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
        processing.  If None, use ``jax.vmap``.
    key : PRNGKey or None
        Random key for optimization.
    compute_eulerian : bool
        If True, also compute Eulerian-frame EC at each point and take
        the worse margin. Default False.

    Returns
    -------
    ECGridResult
        Per-point fields reshaped to ``(*grid_shape, ...)``, plus
        summary statistics for each condition.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    grid_shape = T_field.shape[:-2]
    n_points = int(jnp.prod(jnp.array(grid_shape)))

    # Flatten to (N, 4, 4)
    flat_T = T_field.reshape(n_points, 4, 4)
    flat_g = g_field.reshape(n_points, 4, 4)

    if g_inv_field is not None:
        flat_g_inv = g_inv_field.reshape(n_points, 4, 4)
    else:
        flat_g_inv = jax.vmap(jnp.linalg.inv)(flat_g)

    # Step 1: Classify all points (vmapped)
    flat_T_mixed = jax.vmap(jnp.matmul)(flat_g_inv, flat_T)
    cls_results = jax.vmap(classify_hawking_ellis)(flat_T_mixed, flat_g)

    he_types = cls_results.he_type      # (N,)
    eigenvalues = cls_results.eigenvalues  # (N, 4)
    rho_all = cls_results.rho            # (N,)
    pressures_all = cls_results.pressures  # (N, 3)

    # Step 2: Eigenvalue margins for ALL points (Type I values are real,
    # non-Type-I have NaN rho/pressures results will be NaN for those)
    nec_eig, wec_eig, sec_eig, dec_eig = jax.vmap(check_all)(rho_all, pressures_all)

    # Step 3: Optimisation for ALL points
    def optimize_single(args):
        T_i, g_i, subkey = args
        opt = optimize_point(
            T_i, g_i,
            conditions=("nec", "wec", "sec", "dec"),
            n_starts=n_starts,
            zeta_max=zeta_max,
            key=subkey,
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

    subkeys = jax.random.split(key, n_points)

    if batch_size is not None:
        # lax.map with batch_size for memory-safe sequential-ish processing
        opt_results = lax.map(
            lambda args: optimize_single(args),
            (flat_T, flat_g, subkeys),
            batch_size=batch_size,
        )
    else:
        opt_results = jax.vmap(optimize_single)((flat_T, flat_g, subkeys))

    (nec_opt, wec_opt, sec_opt, dec_opt, worst_obs, worst_par,
     nec_conv, wec_conv, sec_conv, dec_conv,
     nec_nsteps, wec_nsteps, sec_nsteps, dec_nsteps) = opt_results

    # Step 4: Merge eigenvalue and optimization results
    # For Type I: algebraic margins are exact (authoritative for detection).
    # For non-Type-I: use optimization only.
    is_type_i = (he_types == 1.0)

    # Diagnostic: warn if non-Type-I points exist (DEC future-directedness caveat)
    n_non_type_i = int(jnp.sum(~is_type_i))
    if n_non_type_i > 0:
        import warnings
        warnings.warn(
            f"{n_non_type_i} of {n_points} grid points are non-Type-I. "
            f"DEC future-directedness is only guaranteed for Type I.",
            stacklevel=2,
        )

    def merge_margins(eig_m, opt_m):
        # Type I: use algebraic (exact); non-Type-I: use optimizer
        return jnp.where(is_type_i, eig_m, opt_m)

    nec_margins = merge_margins(nec_eig, nec_opt)
    wec_margins = merge_margins(wec_eig, wec_opt)
    sec_margins = merge_margins(sec_eig, sec_opt)
    dec_margins = merge_margins(dec_eig, dec_opt)

    # DEC requires both causal flux AND non-negative energy density (WEC).
    # The optimizer checks flux only; incorporate the WEC margin.
    dec_margins = jnp.minimum(wec_margins, dec_margins)

    # Step 5: Optionally compute Eulerian EC and take worse margin
    if compute_eulerian:
        eulerian_results = jax.vmap(_eulerian_ec_point)(flat_T, flat_g, flat_g_inv)
        nec_margins = jnp.minimum(nec_margins, eulerian_results["nec"])
        wec_margins = jnp.minimum(wec_margins, eulerian_results["wec"])
        sec_margins = jnp.minimum(sec_margins, eulerian_results["sec"])
        dec_margins = jnp.minimum(dec_margins, eulerian_results["dec"])

    # Step 6: Summary statistics
    nec_summary = _compute_summary(nec_margins)
    wec_summary = _compute_summary(wec_margins)
    sec_summary = _compute_summary(sec_margins)
    dec_summary = _compute_summary(dec_margins)

    # Step 6b: Classification statistics
    n_type_i = int(jnp.sum(he_types == 1.0))
    n_type_ii = int(jnp.sum(he_types == 2.0))
    n_type_iii = int(jnp.sum(he_types == 3.0))
    n_type_iv = int(jnp.sum(he_types == 4.0))
    max_imag = float(jnp.max(jnp.abs(cls_results.eigenvalues_imag)))

    # Step 7: Reshape to grid
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


# ---------------------------------------------------------------------------
# Public API: Eulerian-frame EC (separate from optimization)
# ---------------------------------------------------------------------------


def _eulerian_ec_point(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    g_inv: Float[Array, "4 4"],
) -> dict[str, Float[Array, ""]]:
    """Internal: Eulerian EC margins at a single point (vmappable)."""
    # ADM normal: n^a = (1/alpha, -beta^i/alpha)
    # where alpha = 1/sqrt(-g^{00}) and beta^i = g^{0i} * alpha^2
    # => n^0 = 1/alpha, n^i = -g^{0i} * alpha
    alpha = 1.0 / jnp.sqrt(-g_inv[0, 0])
    n_up = jnp.zeros(4)
    n_up = n_up.at[0].set(1.0 / alpha)
    n_up = n_up.at[1].set(-g_inv[0, 1] * alpha)
    n_up = n_up.at[2].set(-g_inv[0, 2] * alpha)
    n_up = n_up.at[3].set(-g_inv[0, 3] * alpha)

    # WEC: T_{ab} n^a n^b (energy density seen by Eulerian observer)
    wec_margin = jnp.einsum("a,ab,b->", n_up, T_ab, n_up)

    # NEC: T_{ab} k^a k^b for null vectors aligned with Eulerian frame
    # Use orthonormal tetrad from the metric
    tetrad = compute_orthonormal_tetrad(g_ab)
    # Evaluate T_{ab} k^a k^b for null vectors in 6 principal directions
    # k = e_0 +/- e_i for i=1,2,3
    nec_vals = []
    for i in range(1, 4):
        k_plus = tetrad[0] + tetrad[i]
        k_minus = tetrad[0] - tetrad[i]
        nec_vals.append(jnp.einsum("a,ab,b->", k_plus, T_ab, k_plus))
        nec_vals.append(jnp.einsum("a,ab,b->", k_minus, T_ab, k_minus))
    nec_margin = jnp.min(jnp.array(nec_vals))

    # SEC: (T_{ab} - 0.5 T g_{ab}) n^a n^b
    T_trace = jnp.einsum("ab,ab->", g_inv, T_ab)
    sec_tensor = T_ab - 0.5 * T_trace * g_ab
    sec_margin = jnp.einsum("a,ab,b->", n_up, sec_tensor, n_up)

    # DEC: flux j^a = -T^a_b n^b must be causal AND future-directed
    T_mixed = g_inv @ T_ab
    j = -jnp.einsum("ac,c->a", T_mixed, n_up)
    j_norm_sq = jnp.einsum("a,ab,b->", j, g_ab, j)
    dec_flux_margin = -j_norm_sq  # positive when flux is timelike/null (causal)

    # Future-directedness: j . n < 0 means future-directed (n is future-pointing)
    n_down = jnp.einsum("ab,b->a", g_ab, n_up)
    dec_future_margin = -jnp.einsum("a,a->", j, n_down)

    # DEC requires causal flux AND future-directed AND non-negative energy (WEC).
    dec_margin = jnp.minimum(wec_margin, jnp.minimum(dec_flux_margin, dec_future_margin))

    return {"nec": nec_margin, "wec": wec_margin, "sec": sec_margin, "dec": dec_margin}


def compute_eulerian_ec(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    g_inv: Float[Array, "4 4"] | None = None,
) -> dict[str, Float[Array, ""]]:
    """Compute energy condition margins for the Eulerian observer ONLY.

    This is the "Eulerian-frame EC result computed on request" for clean
    comparison against observer-robust margins.  The Eulerian observer is
    the unit normal to constant-time spatial hypersurfaces:
    ``n^a = (1/alpha, -beta^i/alpha)``.

    Parameters
    ----------
    T_ab : Float[Array, "4 4"]
        Covariant stress-energy tensor at a single point.
    g_ab : Float[Array, "4 4"]
        Covariant metric at the same point.
    g_inv : Float[Array, "4 4"] or None
        Inverse metric.  Computed if not provided.

    Returns
    -------
    dict
        Maps condition name (``"wec"``, ``"nec"``, ``"sec"``, ``"dec"``)
        to signed margin scalar.
    """
    if g_inv is None:
        g_inv = jnp.linalg.inv(g_ab)
    return _eulerian_ec_point(T_ab, g_ab, g_inv)


# ---------------------------------------------------------------------------
# Public API: ANEC
# ---------------------------------------------------------------------------


def anec_integrand(
    T_ab: Float[Array, "4 4"],
    k: Float[Array, "4"],
) -> Float[Array, ""]:
    """Pointwise ANEC integrand: T_{ab} k^a k^b.

    Computes the integrand for the Averaged Null Energy Condition.
    The full ANEC integral along a null geodesic requires geodesic
    data (not yet implemented).

    Parameters
    ----------
    T_ab : Float[Array, "4 4"]
        Covariant stress-energy tensor at a single point.
    k : Float[Array, "4"]
        Null 4-vector (tangent to a null geodesic).

    Returns
    -------
    Float[Array, ""]
        Scalar T_{ab} k^a k^b.
    """
    return jnp.einsum("a,ab,b->", k, T_ab, k)


def anec_integral(T_field, geodesic):
    """ANEC line integral along a null geodesic.

    Raises
    ------
    NotImplementedError
        ANEC line integral requires geodesic data (not yet implemented).
    """
    raise NotImplementedError(
        "ANEC line integral requires geodesic data (not yet implemented)"
    )
