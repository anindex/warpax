"""Branchless Hawking-Ellis classification of stress-energy tensors (JAX).

Classifies T^a_b by its eigenvalue structure into four types:

- **Type I:**  One timelike eigenvector, three spacelike.  Diagonalizable
  with real eigenvalues {-rho, p1, p2, p3}.  Covers perfect fluids, EM, and
  most physically reasonable matter.
- **Type II:**  One null eigenvector (non-diagonalizable), degenerate eigenvalue.
- **Type III:** Single eigenvalue with multiplicity 4 AND a null eigenvector.
- **Type IV:** Complex eigenvalue pair.

All operations use ``jnp`` no Python ``if/else`` over traced values --
making the classifier fully JIT-compilable and vmappable.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from warpax.energy_conditions.types import ClassificationResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_hawking_ellis(
    T_mixed: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
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
    """
    # Sanitize NaN inputs: cuSolver's geev crashes on NaN matrices (GPU),
    # while CPU returns NaN eigenvalues gracefully.  Replace NaN entries with
    # zero before decomposition; the near_vacuum guard below handles these
    # points correctly (all-zero eigenvalues -> Type I with zero margins).
    T_safe = jnp.where(jnp.isnan(T_mixed), 0.0, T_mixed)
    eigenvalues, eigenvectors = jnp.linalg.eig(T_safe)

    evals_real = eigenvalues.real
    evecs_real = eigenvectors.real
    evals_imag = eigenvalues.imag

    # Scale factor: max(|Re λ|, 1) prevents division-by-zero for vacuum
    # and makes both the imaginary-part and degeneracy checks relative.
    scale = jnp.maximum(jnp.max(jnp.abs(evals_real)), 1.0)

    # Imaginary-part check: two complementary criteria.
    #
    # (1) Absolute: |Im λ| < tol * scale.  Catches eigenvalues with
    #     tiny imaginary parts at any scale (e.g. |Im| ~ 1e-8, |Re| ~ 1e3).
    #
    # (2) Relative: |Im λ| < imag_rtol * max|Re λ| (unclamped).  Catches
    #     split degenerate eigenvalue pairs where jnp.linalg.eig returns
    #     {λ ± εi} with |ε|/|λ| << 1 even though |ε| >> tol * scale.
    #     This occurs at large ||T|| (e.g. WarpShell with |λ| ~ 1e11,
    #     |Im| ~ 2e8: ratio 0.2%).  The unclamped scale avoids the
    #     floor at 1.0, ensuring small-eigenvalue points with |Im| > |Re|
    #     are NOT reclassified (their algebraic margins are unreliable).
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
    # unreliable.  Bypass directly to Type I (margins are ~0).
    # Kept tight (tol, not sqrt(tol)) because at near-vacuum points
    # with larger eigenvalues, the algebraic margins from approximate
    # eigenvalues can differ significantly from the Eulerian result.
    near_vacuum = jnp.max(jnp.abs(evals_real)) < tol

    # ------------------------------------------------------------------
    # Causal character of each eigenvector: g_{ab} v^a v^b
    # Note: jnp.linalg.eig returns δ-orthogonal (Euclidean) eigenvectors,
    # not the g-orthogonal Hawking-Ellis eigenvectors.  The causal
    # character can be unreliable at points with large metric
    # off-diagonal components (shift vector).
    # ------------------------------------------------------------------
    causal = jnp.einsum("ab,ak,bk->k", g_ab, evecs_real, evecs_real)

    n_timelike = jnp.sum(causal < -tol)
    n_null = jnp.sum(jnp.abs(causal) <= tol)

    # ------------------------------------------------------------------
    # Degeneracy: relative tolerance  |lam_i - lam_j| < tol * scale
    # ------------------------------------------------------------------
    sorted_evals = jnp.sort(evals_real)
    gaps = jnp.abs(jnp.diff(sorted_evals))
    n_unique = 1 + jnp.sum(gaps > tol * scale)

    # ------------------------------------------------------------------
    # Branchless type assignment via nested jnp.where
    # ------------------------------------------------------------------
    # Near-vacuum points bypass all checks; their eigenvalues and
    # eigenvectors are pure numerical noise, so Type IV from spurious
    # imaginary parts is suppressed.
    is_type_iv = ~all_real & ~near_vacuum
    is_type_i = (all_real & (n_timelike >= 1) & (n_null == 0)) | near_vacuum
    is_type_iii = all_real & ~near_vacuum & (n_null >= 1) & (n_unique == 1)

    he_type = jnp.where(
        is_type_iv,
        4,
        jnp.where(is_type_i, 1, jnp.where(is_type_iii, 3, 2)),
    )

    # ------------------------------------------------------------------
    # Type I extraction: rho = -eigenvalue(timelike), pressures = rest
    # For non-Type-I the fields are set to NaN.
    # ------------------------------------------------------------------
    timelike_idx = jnp.argmin(causal)

    # Mask for the three spacelike eigenvectors (indices != timelike_idx)
    indices = jnp.arange(4)
    is_spacelike = indices != timelike_idx
    # Gather spacelike eigenvalues via boolean indexing (static size via where)
    # We need exactly 3 values.  Use jnp.where to zero out the timelike entry
    # then sort and take the last 3 (the zeroed entry will sort to the front
    # only if all others are positive, which is not guaranteed).  Instead,
    # replace timelike eigenvalue with +inf so it sorts last, then take first 3.
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
    )


def classify_mixed_tensor(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    g_inv: Float[Array, "4 4"],
    tol: float = 1e-10,
) -> ClassificationResult:
    """Convenience wrapper: raise the first index then classify.

    Computes ``T^a_{\\;b} = g^{ac} T_{cb}`` and delegates to
    :func:`classify_hawking_ellis`.

    Parameters
    ----------
    T_ab : Float[Array, "4 4"]
        Covariant stress-energy tensor T_{ab}, shape (4, 4).
    g_ab : Float[Array, "4 4"]
        Covariant metric g_{ab}, shape (4, 4).
    g_inv : Float[Array, "4 4"]
        Contravariant (inverse) metric g^{ab}, shape (4, 4).
    tol : float
        Classification tolerance.

    Returns
    -------
    ClassificationResult
    """
    T_mixed = g_inv @ T_ab
    return classify_hawking_ellis(T_mixed, g_ab, tol=tol)
