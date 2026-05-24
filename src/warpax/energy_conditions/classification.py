"""Branchless Hawking-Ellis classification of stress-energy tensors (JAX).

Classifies T^a_b by its eigenvalue structure into four types:

- **Type I:** One timelike eigenvector, three spacelike. Diagonalizable
  with real eigenvalues {-rho, p1, p2, p3}. Covers perfect fluids, EM, and
  most physically reasonable matter.
- **Type II:** One null eigenvector (non-diagonalizable), degenerate eigenvalue.
- **Type III:** Single eigenvalue with multiplicity 4 AND a null eigenvector.
- **Type IV:** Complex eigenvalue pair.

All operations use ``jnp`` no Python ``if/else`` over traced values --
making the classifier fully JIT-compilable and vmappable.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from warpax.energy_conditions.types import ClassificationResult

_VALID_SOLVERS = frozenset({"standard", "generalized", "auto"})
# Standard ``jnp.linalg.eig(T^a_b)`` mis-classifies near-degenerate pencils
# (WarpShell v_s=0.5: spurious Type IV at |λ| ~ 10^42).
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
    """Return True when the standard eigen solver result should be discarded."""
    import numpy as np

    ev = np.asarray(eigenvalues)
    ev_im = np.asarray(eigenvalues_imag)
    if float(he_type) == 4.0:
        return True
    if float(np.max(np.abs(ev))) > max_eigenvalue:
        return True
    scale = max(float(np.max(np.abs(ev))), 1.0)
    if float(np.max(np.abs(ev_im))) > imag_rtol * scale:
        return True
    return False


def _standard_solver_unreliable_mask(
    he_types: Float[Array, "N"],
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
    he_type: Float[Array, ""],
    eigenvalues: Float[Array, "4"],
    eigenvalues_imag: Float[Array, "4"],
    *,
    max_eigenvalue: float = _AUTO_MAX_EIGENVALUE,
    imag_rtol: float = _AUTO_IMAG_RTOL,
) -> Float[Array, ""]:
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

    # Sanitize NaN inputs: cuSolver's geev crashes on NaN matrices (GPU),
    # while CPU returns NaN eigenvalues gracefully. Replace NaN entries with
    # zero before decomposition; the near_vacuum guard below handles these
    # points correctly (all-zero eigenvalues -> Type I with zero margins).
    T_safe = jnp.where(jnp.isnan(T_mixed), 0.0, T_mixed)

    if solver == 'standard':
        eigenvalues, eigenvectors = jnp.linalg.eig(T_safe)
    else:  # solver == 'generalized'
        if T_ab is None:
            T_ab = g_ab @ T_safe  # fallback; less stable than caller-provided T_ab
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

    # Scale factor: max(|Re λ|, 1) prevents division-by-zero for vacuum
    # and makes both the imaginary-part and degeneracy checks relative.
    scale = jnp.maximum(jnp.max(jnp.abs(evals_real)), 1.0)

    # Imaginary-part check: two complementary criteria.
    #
    # (1) Absolute: |Im λ| < tol * scale. Catches eigenvalues with
    # tiny imaginary parts at any scale (e.g. |Im| ~ 1e-8, |Re| ~ 1e3).
    #
    # (2) Relative: |Im λ| < imag_rtol * max|Re λ| (unclamped). Catches
    # split degenerate eigenvalue pairs where jnp.linalg.eig returns
    # {λ ± εi} with |ε|/|λ| << 1 even though |ε| >> tol * scale.
    # This occurs at large ||T|| (e.g. WarpShell with |λ| ~ 1e11,
    # |Im| ~ 2e8: ratio 0.2%). The unclamped scale avoids the
    # floor at 1.0, ensuring small-eigenvalue points with |Im| > |Re|
    # are NOT reclassified (their algebraic margins are unreliable).
    #
    # An eigenvalue spectrum is "effectively real" if EITHER holds.
    imag_parts = jnp.abs(evals_imag)
    # imag_rtol is now a parameter (default 3e-3, 0.3% relative tolerance)
    unclamped_scale = jnp.maximum(jnp.max(jnp.abs(evals_real)), jnp.sqrt(tol))
    all_real = jnp.all(imag_parts < tol * scale) | jnp.all(
        imag_parts < imag_rtol * unclamped_scale
    )

    # Near-vacuum: all real eigenvalue magnitudes negligible.
    # Eigenvectors are pure numerical noise, so causal character is
    # unreliable. Bypass directly to Type I (margins are ~0).
    # Kept tight (tol, not sqrt(tol)) because at near-vacuum points
    # with larger eigenvalues, the algebraic margins from approximate
    # eigenvalues can differ significantly from the Eulerian result.
    near_vacuum = jnp.max(jnp.abs(evals_real)) < tol

    # Causal character ``g_{ab} v^a v^b`` for each eigenvector. Use a
    # RELATIVE sign threshold (normalized by ``max|v^T g v|``): an
    # absolute ``tol`` over-tightens the timelike test at large
    # ``|g_{ab}| |v|^2`` (e.g. WarpShell ~ 10^{11}). ``g_quad_scale``
    # floors at 1.0 so Minkowski-scale behavior is unchanged.
    causal = jnp.einsum("ab,ak,bk->k", g_ab, evecs_real, evecs_real)
    g_quad_scale = jnp.maximum(jnp.max(jnp.abs(causal)), 1.0)
    relative_g_quad = causal / g_quad_scale

    n_timelike = jnp.sum(relative_g_quad < -tol)
    n_null = jnp.sum(jnp.abs(relative_g_quad) <= tol)

    # Degeneracy: relative tolerance ``|lam_i - lam_j| < tol * scale``.
    sorted_evals = jnp.sort(evals_real)
    gaps = jnp.abs(jnp.diff(sorted_evals))
    n_unique = 1 + jnp.sum(gaps > tol * scale)

    # Branchless type assignment via nested ``jnp.where``. Near-vacuum
    # points bypass all checks (eigenvalues / eigenvectors are pure
    # numerical noise) so spurious Type IV is suppressed.
    is_type_iv = ~all_real & ~near_vacuum
    is_type_i = (all_real & (n_timelike >= 1) & (n_null == 0)) | near_vacuum
    is_type_iii = all_real & ~near_vacuum & (n_null >= 1) & (n_unique == 1)

    he_type = jnp.where(
        is_type_iv,
        4,
        jnp.where(is_type_i, 1, jnp.where(is_type_iii, 3, 2)),
    )

    # Type I extraction: ``rho = -eigenvalue(timelike)``, pressures = rest.
    # When ``causal[i] ~ causal[j]`` (near-degenerate eigenvectors), bias
    # the selection towards the most-negative eigenvalue with a 1e-15
    # tiebreaker so Type-I rho is stable under f64 noise.
    timelike_idx = jnp.argmin(causal + 1e-15 * evals_real)

    indices = jnp.arange(4)
    is_spacelike = indices != timelike_idx
    # Replace timelike eigenvalue with +inf so it sorts last; take the
    # first three of the sorted result for the spacelike pressures.
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
