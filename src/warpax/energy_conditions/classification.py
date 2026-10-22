"""Branchless Hawking-Ellis classification of stress-energy tensors (JAX).

Classifies ``T^a_b`` by its eigenvalue structure into four types:

- **Type I:** one timelike eigenvector and three spacelike eigenvectors;
  diagonalizable with real eigenvalues ``{-rho, p1, p2, p3}``. Covers
  perfect fluids, electromagnetism, and most physically reasonable matter.
- **Type II:** one null eigenvector (non-diagonalizable), degenerate eigenvalue;
  Segre ``[2,1,1]``, a ``J_2`` block.
- **Type III:** Segre ``[3,1]``, a ``J_3`` block with a null eigenvector. The
  repeated eigenvalue need *not* have multiplicity four: generic Type III is
  ``J_3(lambda) (+) [p]`` with ``p != lambda``.
- **Type IV:** complex eigenvalue pair.

All control flow uses ``jnp.where`` rather than Python ``if`` on traced
values, so the classifier is JIT- and vmap-safe.

Resolution limit, read this before trusting a Type II or Type III label
--------------------------------------------------------------------------
A defective Jordan block of size ``m`` is not a continuous function of the matrix
entries: perturbing it by ``delta`` moves its eigenvalues by ``O(delta^(1/m))``.
In float64, rounding alone supplies ``delta ~ eps``, so a ``J_2`` splits by
``eps^(1/2) ~ 1.5e-8`` and a ``J_3`` by ``eps^(1/3) ~ 6e-6``, and the split is
generically into a *complex* pair. No choice of ``tol`` repairs this: at
``tol = 1e-10`` every generic Type III is returned as Type IV, because its
eigenvalues do come back complex at the ``1e-6`` level.

Two consequences, both deliberate:

1. The Type II and Type III labels from this module are **diagnostics carrying a
   measurable error rate**, not certificates. Re-verify with
   :mod:`.classification_mpmath`, with one qualification: its 50-digit arithmetic
   resolves the *stored float64 matrix*, not the tensor before rounding. A ``J_m``
   block that was rounded into float64 is already split by ``O(eps^(1/m))``
   (about ``6e-6`` for ``J_3``) at any working precision, so the mpmath path
   flags such points ``uncertain`` (``input_rounding_split``) alongside its
   ``cond_V`` (Bauer-Fike) flag for ill-conditioned eigenvector matrices.
2. The energy conditions do **not** depend on any of it. :mod:`.slemma` decides
   NEC/WEC/SEC/DEC by a single linear matrix inequality that never forms an
   eigendecomposition and never consults the type, and it returns the right answer
   at exactly the points this classifier cannot resolve. The type is reported
   because it is physically informative, not because the certification needs it.

Because Type III and Type IV violate every standard energy condition
(Martin-Moruno & Visser 2017), the LMI also checks these labels: a point labeled
III or IV whose LMI margin certifies satisfaction is a classification error, and
counting them measures the rate rather than assuming it away.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from warpax.energy_conditions.types import ClassificationResult

_VALID_SOLVERS = frozenset({"standard", "generalized", "auto"})

_AUTO_MAX_EIGENVALUE = 1e25
_AUTO_IMAG_RTOL = 0.05


def _standard_solver_unreliable_mask(
    he_types: Float[Array, "N"],
    eigenvalues: Float[Array, "N 4"],
    eigenvalues_imag: Float[Array, "N 4"],
    *,
    max_eigenvalue: float = _AUTO_MAX_EIGENVALUE,
    imag_rtol: float = _AUTO_IMAG_RTOL,
) -> Float[Array, "N"]:
    """Boolean mask (float 0/1) for grid points needing generalized fallback.

    Standard ``jnp.linalg.eig(T^a_b)`` mis-classifies near-degenerate pencils
    (WarpShell at ``v_s=0.5`` returns a spurious Type IV with ``|lambda| ~ 1e42``),
    so every Type IV point, and any point whose eigenvalues exceed
    ``max_eigenvalue`` or carry imaginary parts above ``imag_rtol`` times the scale,
    is routed to the generalized pencil solver.
    """
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
) -> Bool[Array, ""]:
    """Traced predicate: True when the standard result needs the pencil fallback."""
    max_abs = jnp.max(jnp.abs(eigenvalues))
    max_imag = jnp.max(jnp.abs(eigenvalues_imag))
    scale = jnp.maximum(max_abs, 1.0)
    return (he_type == 4.0) | (max_abs > max_eigenvalue) | (max_imag > imag_rtol * scale)


_EVEC_COND_MAX = 1e5


def ill_conditioned_eigenbasis(evecs, cond_max: float = _EVEC_COND_MAX):
    """True where the eigenvector matrix is too ill-conditioned to read.

    Large boosts make a Type-I eigenbasis nearly singular, and the float64
    eigenvalues then carry errors of order ``eps * ||T|| * cond``. Such points
    are routed away from the eigenvalue slacks. The cutoff is a conditioning
    policy, not a rigorous error bound.
    """
    sv = jnp.linalg.svd(evecs, compute_uv=False)
    return (sv[..., 0] / jnp.maximum(sv[..., -1], 1e-300)) > cond_max


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
        raise ValueError(f"solver must be one of {_VALID_SOLVERS}; got {solver!r}")
    if solver == "standard":
        return classify_hawking_ellis(
            T_mixed,
            g_ab,
            solver="standard",
            tol=tol,
            imag_rtol=imag_rtol,
        )
    if solver == "generalized":
        if T_ab is None:
            T_ab = g_ab @ T_mixed
        return classify_hawking_ellis(
            T_mixed,
            g_ab,
            solver="generalized",
            T_ab=T_ab,
            tol=tol,
            imag_rtol=imag_rtol,
        )

    cls_std = classify_hawking_ellis(
        T_mixed,
        g_ab,
        solver="standard",
        tol=tol,
        imag_rtol=imag_rtol,
    )
    T_ab_local = g_ab @ T_mixed if T_ab is None else T_ab

    needs_fallback = _is_unreliable_single(
        cls_std.he_type,
        cls_std.eigenvalues,
        cls_std.eigenvalues_imag,
        imag_rtol=imag_rtol,
    )

    def _gen_branch(_):
        return classify_hawking_ellis(
            T_mixed,
            g_ab,
            solver="generalized",
            T_ab=T_ab_local,
            tol=tol,
            imag_rtol=imag_rtol,
        )

    def _std_branch(_):
        return cls_std

    return jax.lax.cond(needs_fallback, _gen_branch, _std_branch, operand=None)


def classify_hawking_ellis(
    T_mixed: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    *,
    solver: str = "standard",
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
        Tolerance for classifying eigenvalues as real / degenerate. **This is the
        real-spectrum threshold**: the spectrum counts as real when
        ``|Im lambda| < tol * scale`` with ``scale = max(max|Re lambda|,
        max|T^a_b|)``. Anything that sweeps a tolerance to test label stability
        must sweep this one.
    imag_rtol : float
        Dispatch threshold for ``solver='auto'`` only, it decides when the
        standard eigensolver is distrusted and the generalized pencil is tried
        instead. It does **not** enter the real/complex decision, so a caller
        tuning it to move labels gets silence, and a sweep over it reports
        stability it never tested.

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

    Notes
    -----
    Non-finite entries are zeroed before the eigensolve (cuSolver crashes on
    them) and the point returns ``he_type = NaN``. ``scale`` is
    ``max(max|Re lambda|, max|T|)`` floored at ``1e-300``, so every tolerance is
    covariant under ``T -> cT``; a floor at ``1.0`` would make them absolute
    below unit norm. Vacuum requires an exactly zero tensor because negative
    null dust is nilpotent (zero spectrum) yet Type II. The repeated eigenvalue
    is the mean of the largest cluster of gaps below ``tol * scale``; the
    Jordan chain length is the defect ``multiplicity - dim ker(T - lam I)``
    with the rank cut at the defective splitting scale ``sqrt(tol)``, which
    separates ``J_2 (+) [lam] (+) [q]`` (Type II) from ``J_3`` (Type III). A
    defective block splits its eigenvectors into nearly parallel directions
    whose causal character is ``O(split)``, so when the eigenvectors collapse a
    null test of ``4 * cluster_gap / scale`` is admitted; it is dimensionless
    and cluster-local, so it cannot change under a stress rescaling. The
    timelike eigenvector for ``rho`` is chosen by normalized causal character
    with a scale-free tie break.
    """
    if solver not in ("standard", "generalized"):
        raise ValueError(f"solver must be 'standard' or 'generalized'; got {solver!r}")

    has_nan = jnp.any(~jnp.isfinite(T_mixed)) | jnp.any(~jnp.isfinite(g_ab))
    T_safe = jnp.where(jnp.isfinite(T_mixed), T_mixed, 0.0)

    if solver == "standard":
        eigenvalues, eigenvectors = jnp.linalg.eig(T_safe)
    else:
        if T_ab is None:
            T_ab = g_ab @ T_safe
        T_ab_safe = jnp.where(jnp.isfinite(T_ab), T_ab, 0.0)
        g_ab_safe = jnp.where(jnp.isfinite(g_ab), g_ab, 0.0)
        try:
            from warpax.energy_conditions._gen_eig_callback import _gen_eig_pencil
        except ImportError as e:
            raise ValueError(
                "scipy is required for solver='generalized'; install warpax[solver]"
            ) from e
        eigenvalues, eigenvectors = _gen_eig_pencil(T_ab_safe, g_ab_safe)

    evals_real = eigenvalues.real
    evecs_real = eigenvectors.real
    evals_imag = eigenvalues.imag

    scale = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(evals_real)), jnp.max(jnp.abs(T_safe))), 1e-300)

    imag_parts = jnp.abs(evals_imag)
    all_real = jnp.all(imag_parts < tol * scale)

    near_vacuum = (jnp.max(jnp.abs(eigenvalues)) < tol) & (jnp.max(jnp.abs(T_safe)) == 0.0)

    causal = jnp.einsum("ab,ak,bk->k", g_ab, evecs_real, evecs_real)
    g_quad_scale = jnp.max(jnp.abs(causal))
    relative_g_quad = jnp.where(g_quad_scale > 0.0, causal / g_quad_scale, causal)

    n_timelike = jnp.sum(relative_g_quad < -tol)
    n_null = jnp.sum(jnp.abs(relative_g_quad) <= tol)

    sorted_evals = jnp.sort(evals_real)
    gaps = jnp.abs(jnp.diff(sorted_evals))

    close = gaps <= tol * scale
    longest_run = jnp.where(
        close[0] & close[1] & close[2],
        3,
        jnp.where(
            (close[0] & close[1]) | (close[1] & close[2]),
            2,
            jnp.where(close[0] | close[1] | close[2], 1, 0),
        ),
    )
    max_multiplicity = 1 + longest_run
    run_start = jnp.where(
        close[0] & close[1] & close[2],
        0,
        jnp.where(
            close[0] & close[1],
            0,
            jnp.where(close[1] & close[2], 1, jnp.where(close[0], 0, jnp.where(close[1], 1, 2))),
        ),
    )
    idx = jnp.arange(4)
    in_run = (idx >= run_start) & (idx < run_start + max_multiplicity)
    lam_bar = jnp.sum(jnp.where(in_run, sorted_evals, 0.0)) / max_multiplicity

    nilpotent = T_safe - lam_bar * jnp.eye(4, dtype=evals_real.dtype)
    svals = jnp.linalg.svd(nilpotent, compute_uv=False)
    rank_tol = jnp.maximum(jnp.sqrt(tol), 1e-7) * scale
    kernel_dim = jnp.sum(svals <= rank_tol)
    defect = max_multiplicity - kernel_dim

    unit_evecs = evecs_real / jnp.maximum(
        jnp.linalg.norm(evecs_real, axis=0, keepdims=True), 1e-300
    )
    gram_offdiag = jnp.abs(unit_evecs.T @ unit_evecs) - jnp.eye(4)
    evecs_collapsed = jnp.max(gram_offdiag) > 1.0 - 1e-6
    cluster_gap = jnp.max(jnp.where(close, gaps, 0.0))
    split_null_tol = jnp.maximum(tol, 4.0 * cluster_gap / scale)
    n_null_split = jnp.sum(jnp.abs(relative_g_quad) <= split_null_tol)

    is_type_iv = ~all_real & ~near_vacuum
    is_type_iii = (
        all_real
        & ~near_vacuum
        & (max_multiplicity >= 3)
        & (defect >= 2)
        & ((n_null >= 1) | (evecs_collapsed & (n_null_split >= 1)))
    )
    is_type_i = (all_real & (n_timelike >= 1) & (n_null == 0) & ~is_type_iii) | near_vacuum

    he_type = jnp.where(
        is_type_iv,
        4,
        jnp.where(is_type_i, 1, jnp.where(is_type_iii, 3, 2)),
    )
    he_type = jnp.where(has_nan, jnp.nan, he_type.astype(evals_real.dtype))

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
        is_vacuum=jnp.where(has_nan, nan, near_vacuum.astype(evals_real.dtype)),
    )


def classify_mixed_tensor(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    g_inv: Float[Array, "4 4"],
    *,
    solver: str = "standard",
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
        T_mixed,
        g_ab,
        solver=solver,
        T_ab=T_ab if solver == "generalized" else None,
        tol=tol,
    )
