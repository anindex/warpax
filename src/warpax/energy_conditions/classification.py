"""Branchless Hawking-Ellis classification of stress-energy tensors (JAX).

Classifies ``T^a_b`` by its eigenvalue structure into four types:

- **Type I:** one timelike eigenvector and three spacelike eigenvectors;
  diagonalizable with real eigenvalues ``{-rho, p1, p2, p3}``. Covers
  perfect fluids, electromagnetism, and most physically reasonable matter.
- **Type II:** one null eigenvector (non-diagonalizable), degenerate eigenvalue.
- **Type III:** single eigenvalue of multiplicity four with a null eigenvector.
- **Type IV:** complex eigenvalue pair.

All control flow uses ``jnp.where`` rather than Python ``if`` on traced
values, so the classifier is JIT- and vmap-safe.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from warpax.energy_conditions.types import ClassificationResult

_VALID_SOLVERS = frozenset({"standard", "generalized", "auto"})

# Standard ``jnp.linalg.eig(T^a_b)`` mis-classifies near-degenerate pencils
# (e.g. WarpShell at v_s=0.5 returns spurious Type IV with |lambda| ~ 1e42).
# When eigenvalues blow up or carry large imaginary parts, fall back to the
# generalized pencil solver.
_AUTO_MAX_EIGENVALUE = 1e25
_AUTO_IMAG_RTOL = 0.05


def _standard_solver_unreliable_scalar(
    he_type: float,
    eigenvalues,
    eigenvalues_imag,
    *,
    max_eigenvalue: float = _AUTO_MAX_EIGENVALUE,
    imag_rtol: float = _AUTO_IMAG_RTOL,
) -> bool:
    """True when the standard eigen solver result should be discarded."""
    import numpy as np

    ev = np.asarray(eigenvalues)
    ev_im = np.asarray(eigenvalues_imag)
    if float(he_type) == 4.0:
        return True
    if float(np.max(np.abs(ev))) > max_eigenvalue:
        return True
    scale = max(float(np.max(np.abs(ev))), 1.0)
    return float(np.max(np.abs(ev_im))) > imag_rtol * scale


def _standard_solver_unreliable_mask(
    he_types: Int[Array, "N"],
    eigenvalues: Float[Array, "N 4"],
    eigenvalues_imag: Float[Array, "N 4"],
    *,
    max_eigenvalue: float = _AUTO_MAX_EIGENVALUE,
    imag_rtol: float = _AUTO_IMAG_RTOL,
) -> Float[Array, "N"]:
    """Boolean mask (float 0/1) for grid points needing generalized fallback."""
    max_abs = jnp.max(jnp.abs(eigenvalues), axis=-1)
    max_imag = jnp.max(jnp.abs(eigenvalues_imag), axis=-1)
    scale = jnp.maximum(max_abs, 1.0)
    is_type_iv = he_types == 4.0
    huge_eig = max_abs > max_eigenvalue
    large_imag = max_imag > imag_rtol * scale
    return (is_type_iv | huge_eig | large_imag).astype(jnp.float64)


def _is_unreliable_single(
    he_type: Int[Array, ""],
    eigenvalues: Float[Array, "4"],
    eigenvalues_imag: Float[Array, "4"],
    *,
    max_eigenvalue: float = _AUTO_MAX_EIGENVALUE,
    imag_rtol: float = _AUTO_IMAG_RTOL,
) -> Bool[Array, ""]:
    """Traced predicate: True when the standard result needs the pencil fallback."""
    max_abs = jnp.max(jnp.abs(eigenvalues))
    max_imag = jnp.max(jnp.abs(eigenvalues_imag))
    scale = jnp.maximum(max_abs, 1.0)
    return (
        (he_type == 4.0)
        | (max_abs > max_eigenvalue)
        | (max_imag > imag_rtol * scale)
    )


def classify_with_solver(
    T_mixed: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    T_ab: Float[Array, "4 4"] | None,
    *,
    solver: str = "auto",
    tol: float = 1e-10,
    imag_rtol: float = 3e-3,
) -> ClassificationResult:
    """Classify with optional auto-fallback to the generalized pencil solver.

    With ``solver='auto'`` the pencil solver is invoked through
    :func:`jax.lax.cond` so the routine remains JIT- and vmap-safe; the
    fallback predicate is computed from traced eigenvalues using
    :func:`_is_unreliable_single`.
    """
    if solver not in _VALID_SOLVERS:
        raise ValueError(
            f"solver must be one of {_VALID_SOLVERS}; got {solver!r}"
        )
    if solver == "standard":
        return classify_hawking_ellis(
            T_mixed, g_ab, solver="standard", tol=tol, imag_rtol=imag_rtol,
        )
    if solver == "generalized":
        if T_ab is None:
            T_ab = g_ab @ T_mixed
        return classify_hawking_ellis(
            T_mixed, g_ab, solver="generalized", T_ab=T_ab,
            tol=tol, imag_rtol=imag_rtol,
        )

    cls_std = classify_hawking_ellis(
        T_mixed, g_ab, solver="standard", tol=tol, imag_rtol=imag_rtol,
    )
    T_ab_local = g_ab @ T_mixed if T_ab is None else T_ab

    needs_fallback = _is_unreliable_single(
        cls_std.he_type, cls_std.eigenvalues, cls_std.eigenvalues_imag,
        imag_rtol=imag_rtol,
    )

    def _gen_branch(_):
        return classify_hawking_ellis(
            T_mixed, g_ab, solver="generalized", T_ab=T_ab_local,
            tol=tol, imag_rtol=imag_rtol,
        )

    def _std_branch(_):
        return cls_std

    return jax.lax.cond(needs_fallback, _gen_branch, _std_branch, operand=None)


def classify_hawking_ellis(
    T_mixed: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    *,
    solver: str = 'standard',
    T_ab: Float[Array, "4 4"] | None = None,
    tol: float = 1e-10,
    imag_rtol: float = 3e-3,
) -> ClassificationResult:
    """Classify the Hawking-Ellis type of *T^a_b* at a single spacetime point.

    Parameters
    ----------
    T_mixed : Float[Array, "4 4"]
        Mixed stress-energy tensor T^a_{\\;b}, shape (4, 4).
    g_ab : Float[Array, "4 4"]
        Covariant metric tensor g_{ab}, shape (4, 4).
    solver : {'standard', 'generalized'}, keyword-only
        Eigenvalue backend.

        - ``'standard'`` (default): ``jnp.linalg.eig(T_mixed)``;
          scipy-free pure-JAX path.
        - ``'generalized'``: ``scipy.linalg.eig(T_ab, g_ab)`` via
          :func:`jax.pure_callback`; solves the pencil ``(T - lam g)v = 0``
          directly. Stabilizes Jordan-defective classification at
          near-degenerate eigenstructure (e.g. WarpShell v_s=0.5 idx=8
          near ``|lam| ~ 10^42``). Carries host-callback overhead (~5-10x
          slower per grid; see :mod:`._gen_eig_callback` Notes).
        See :func:`classify_with_solver` for ``solver='auto'`` fallback.
    T_ab : Float[Array, "4 4"] or None, keyword-only
        Covariant stress-energy tensor T_{ab}, required when
        ``solver='generalized'`` for numerical stability. If ``None`` and
        ``solver='generalized'``, the function reconstructs ``T_ab =
        g_ab @ T_mixed`` (mathematically equivalent but introduces an
        additional matmul rounding step). Ignored when ``solver='standard'``.
    tol : float
        Tolerance for classifying eigenvalues as real / degenerate.
    imag_rtol : float
        Relative tolerance for split degenerate eigenvalue pairs.
        Eigenvalues with |Im| < imag_rtol * max|Re| are treated as real.

    Returns
    -------
    ClassificationResult
        NamedTuple with ``he_type`` (1-4), eigenvalues, eigenvectors,
        and (for Type I) ``rho`` / ``pressures``.

    Raises
    ------
    ValueError
        If ``solver`` is not one of ``{'standard', 'generalized'}``, or
        if ``solver='generalized'`` is requested without scipy installed
        (``pip install warpax[solver]`` to enable the path).
    """
    if solver not in ('standard', 'generalized'):
        raise ValueError(
            f"solver must be 'standard' or 'generalized'; got {solver!r}"
        )

    # cuSolver's geev crashes on NaN GPU inputs (CPU is graceful). Zero
    # out NaN entries; the near-vacuum guard below absorbs the result.
    T_safe = jnp.where(jnp.isnan(T_mixed), 0.0, T_mixed)

    if solver == 'standard':
        eigenvalues, eigenvectors = jnp.linalg.eig(T_safe)
    else:
        if T_ab is None:
            T_ab = g_ab @ T_safe
        T_ab_safe = jnp.where(jnp.isnan(T_ab), 0.0, T_ab)
        g_ab_safe = jnp.where(jnp.isnan(g_ab), 0.0, g_ab)
        try:
            from warpax.energy_conditions._gen_eig_callback import _gen_eig_pencil
        except ImportError as e:
            raise ValueError(
                "scipy is required for solver='generalized'; "
                "install warpax[solver]"
            ) from e
        eigenvalues, eigenvectors = _gen_eig_pencil(T_ab_safe, g_ab_safe)

    evals_real = eigenvalues.real
    evecs_real = eigenvectors.real
    evals_imag = eigenvalues.imag

    # max(|Re lambda|, 1) prevents division by zero in vacuum and makes
    # the imaginary-part and degeneracy checks scale-relative.
    scale = jnp.maximum(jnp.max(jnp.abs(evals_real)), 1.0)

    # Two complementary "real spectrum" tests, combined with logical OR:
    #   (a) absolute: |Im| < tol * scale -- catches uniformly tiny |Im|.
    #   (b) relative: |Im| < imag_rtol * max|Re| (unclamped) -- catches
    #       split-degenerate pairs at large ||T||, e.g. WarpShell where
    #       |Re| ~ 1e11 and |Im| ~ 1e8 (relative 0.1 %). The unclamped
    #       scale prevents reclassifying small-eigenvalue points where
    #       |Im| genuinely exceeds |Re|.
    imag_parts = jnp.abs(evals_imag)
    unclamped_scale = jnp.maximum(jnp.max(jnp.abs(evals_real)), jnp.sqrt(tol))
    all_real = jnp.all(imag_parts < tol * scale) | jnp.all(
        imag_parts < imag_rtol * unclamped_scale
    )

    # Near-vacuum bypass: eigenvectors are numerical noise, causal
    # character is unreliable. Force Type I (margins ~0). Kept tight
    # (``tol``, not ``sqrt(tol)``) so larger eigenvalues fall through.
    near_vacuum = jnp.max(jnp.abs(evals_real)) < tol

    # Causal character g_{ab} v^a v^b per eigenvector, with a relative
    # sign threshold (floored at 1.0 to keep Minkowski behavior).
    causal = jnp.einsum("ab,ak,bk->k", g_ab, evecs_real, evecs_real)
    g_quad_scale = jnp.maximum(jnp.max(jnp.abs(causal)), 1.0)
    relative_g_quad = causal / g_quad_scale

    n_timelike = jnp.sum(relative_g_quad < -tol)
    n_null = jnp.sum(jnp.abs(relative_g_quad) <= tol)

    sorted_evals = jnp.sort(evals_real)
    gaps = jnp.abs(jnp.diff(sorted_evals))
    n_unique = 1 + jnp.sum(gaps > tol * scale)

    is_type_iv = ~all_real & ~near_vacuum
    is_type_i = (all_real & (n_timelike >= 1) & (n_null == 0)) | near_vacuum
    is_type_iii = all_real & ~near_vacuum & (n_null >= 1) & (n_unique == 1)

    he_type = jnp.where(
        is_type_iv,
        4,
        jnp.where(is_type_i, 1, jnp.where(is_type_iii, 3, 2)),
    )

    # Type I extraction: ``rho = -eigenvalue(timelike)``, pressures = the
    # spacelike eigenvalues sorted. Pick the most-timelike eigenvector by its
    # NORMALIZED causal character ``relative_g_quad`` (in roughly [-1, 1]),
    # with a tiny scale-free bias toward the most-negative eigenvalue to break
    # ties when two causal characters are degenerate. The bias is normalized by
    # ``scale`` so it stays ~1e-12 regardless of ||T||: the previous
    # ``1e-15 * evals_real`` form grew to ~1e-4 at ||T|| ~ 1e11 (e.g. WarpShell),
    # large enough to mis-select the timelike eigenvector on near-degenerate
    # points and flip the sign of ``rho``.
    timelike_idx = jnp.argmin(relative_g_quad + 1e-12 * (evals_real / scale))

    indices = jnp.arange(4)
    is_spacelike = indices != timelike_idx
    masked_evals = jnp.where(is_spacelike, evals_real, jnp.inf)
    sorted_masked = jnp.sort(masked_evals)
    pressures_raw = sorted_masked[:3]

    rho_raw = -evals_real[timelike_idx]

    nan = jnp.array(jnp.nan, dtype=evals_real.dtype)
    rho = jnp.where(he_type == 1, rho_raw, nan)
    pressures = jnp.where(he_type == 1, pressures_raw, jnp.full(3, nan))

    return ClassificationResult(
        he_type=he_type,
        eigenvalues=evals_real,
        eigenvectors=evecs_real,
        rho=rho,
        pressures=pressures,
        eigenvalues_imag=evals_imag,
        is_vacuum=near_vacuum.astype(evals_real.dtype),
    )


def classify_mixed_tensor(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    g_inv: Float[Array, "4 4"],
    *,
    solver: str = 'standard',
    tol: float = 1e-10,
) -> ClassificationResult:
    """Convenience wrapper: raise the FIRST index of T then classify.

    Computes ``T^a_{\\;b} = g^{ac} T_{cb}`` (contracting the second slot
    of ``g^{-1}`` with the first slot of ``T``) and delegates to
    :func:`classify_hawking_ellis`. The convention matters when
    ``T_{ab}`` is non-symmetric: this routine always raises the first
    index, matching the eigenvalue equation
    ``T^a_{\\;b} v^b = lambda v^a`` consumed downstream.

    Parameters
    ----------
    T_ab : Float[Array, "4 4"]
        Covariant stress-energy tensor T_{ab}, shape (4, 4).
    g_ab : Float[Array, "4 4"]
        Covariant metric g_{ab}, shape (4, 4).
    g_inv : Float[Array, "4 4"]
        Contravariant (inverse) metric g^{ab}, shape (4, 4).
    solver : {'standard', 'generalized'}, keyword-only
        Eigenvalue backend. When ``'generalized'`` the caller's
        ``T_ab`` is forwarded directly to :func:`classify_hawking_ellis`
        so the pencil solver runs on the original covariant tensor.
    tol : float
        Classification tolerance.
    """
    T_mixed = jnp.einsum("ac,cb->ab", g_inv, T_ab)
    return classify_hawking_ellis(
        T_mixed, g_ab,
        solver=solver,
        T_ab=T_ab if solver == 'generalized' else None,
        tol=tol,
    )
