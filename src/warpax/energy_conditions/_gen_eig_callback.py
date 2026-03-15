"""Host-callback wrapper for the generalized eigenvalue problem (T − λg)v = 0.

The Hawking-Ellis classifier requires non-Hermitian generalized eigenvalues
of the pencil ``(T_ab, g_ab)`` for the ``solver='generalized'`` path.
JAX 0.10 has no native generalized non-Hermitian eigensolver, and Cholesky-
whiten is empirically infeasible because the Lorentzian metric ``g`` is
indefinite (``np.linalg.cholesky(diag(-1,1,1,1))`` raises
``LinAlgError: Matrix is not positive definite``).

This module bridges to LAPACK's ``zggev`` / ``dggev`` (the QZ algorithm)
via :func:`scipy.linalg.eig` wrapped in :func:`jax.pure_callback`. The
callback is composable under :func:`jax.jit` and :func:`jax.vmap` provided
``vmap_method='sequential'`` is passed (mandatory under JAX 0.10).

scipy is imported lazily by the caller (``classification.py`` only imports
this module when ``solver='generalized'``), so the default
``solver='standard'`` path retains zero scipy dependency.

References
----------
- Paper commitment to solving the generalized pencil (T - lambda*g)v = 0.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as _sla
from jaxtyping import Array, Complex, Float


def _scipy_gen_eig_host(
    T_ab: np.ndarray, g_ab: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Host-side generalized eigensolver (LAPACK zggev via scipy.linalg.eig).

    Solves ``T_ab · v = lambda · g_ab · v`` for the four (complex) eigenvalues
    and right eigenvectors. Uses LAPACK ``dggev`` / ``zggev`` (the QZ
    algorithm), which handles indefinite ``g`` natively (no Cholesky
    whitening required).

    Parameters
    ----------
    T_ab : np.ndarray, shape (4, 4)
        Covariant stress-energy tensor.
    g_ab : np.ndarray, shape (4, 4)
        Covariant metric (Lorentzian signature, indefinite).

    Returns
    -------
    eigvals : np.ndarray, complex128, shape (4,)
        The four (complex) generalized eigenvalues.
    eigvecs : np.ndarray, complex128, shape (4, 4)
        The four right eigenvectors as columns; g-orthogonal for distinct
        eigenvalues.

    Notes
    -----
    ``check_finite=False``: NaN sanitisation happens upstream in
    ``classify_hawking_ellis``, so skipping the LAPACK NaN scan is safe and
    saves ~5% of the call cost.
    """
    eigvals, eigvecs = _sla.eig(T_ab, g_ab, check_finite=False)
    return eigvals.astype(np.complex128), eigvecs.astype(np.complex128)


def _gen_eig_pencil(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
) -> tuple[Complex[Array, "4"], Complex[Array, "4 4"]]:
    """JAX-callable generalized eigensolver: ``(T − λg)v = 0``.

    Wraps :func:`_scipy_gen_eig_host` in :func:`jax.pure_callback` with
    ``vmap_method='sequential'`` so it is composable under :func:`jax.jit`
    and :func:`jax.vmap`.

    Parameters
    ----------
    T_ab : Float[Array, "4 4"]
        Covariant stress-energy tensor (NaN-sanitised by caller).
    g_ab : Float[Array, "4 4"]
        Covariant metric (NaN-sanitised by caller; indefinite signature OK).

    Returns
    -------
    eigvals : Complex[Array, "4"]
        Generalized eigenvalues, complex128.
    eigvecs : Complex[Array, "4 4"]
        Right eigenvectors as columns, complex128.

    Notes
    -----
    - ``vmap_method='sequential'`` is mandatory under JAX 0.10. Without it,
      ``jax.vmap`` over this function raises ``NotImplementedError``.
    - Grid-level cost: ~30-40 µs per 4×4 callback on CPU; vmap is
      sequential (LAPACK call is single-threaded). Expect ~5-10x
      slowdown vs ``jnp.linalg.eig`` on a 32^3 grid.
    - Float64 enforced globally via ``warpax/__init__.py``; the
      ``ShapeDtypeStruct(..., jnp.complex128)`` requires x64 mode.
    """
    out_shape = (
        jax.ShapeDtypeStruct((4,), jnp.complex128),
        jax.ShapeDtypeStruct((4, 4), jnp.complex128),
    )
    return jax.pure_callback(
        _scipy_gen_eig_host,
        out_shape,
        T_ab,
        g_ab,
        vmap_method='sequential',
    )
