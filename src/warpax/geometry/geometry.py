"""Pointwise curvature computation chain via JAX autodiff.

Computes the full chain: metric -> Christoffel -> Riemann -> Ricci -> Einstein -> stress-energy
at a single spacetime point using exact forward-mode automatic differentiation (jax.jacfwd),
eliminating all finite-difference error.

Index conventions:
    - Christoffel: Gamma^lam_{mu nu} as (4,4,4) array [upper, lower, lower]
    - Riemann: R^lam_{mu nu rho} as (4,4,4,4) array [upper, lower, lower, lower]
    - Ricci: R_{mu nu} as (4,4) array [lower, lower]
    - Einstein: G_{mu nu} as (4,4) array [lower, lower]
    - Stress-energy: T_{mu nu} as (4,4) array [lower, lower]
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class CurvatureResult(NamedTuple):
    """All tensors from the curvature computation chain at a single point."""
    metric: Float[Array, "4 4"]
    metric_inv: Float[Array, "4 4"]
    christoffel: Float[Array, "4 4 4"]
    riemann: Float[Array, "4 4 4 4"]
    ricci: Float[Array, "4 4"]
    ricci_scalar: Float[Array, ""]
    einstein: Float[Array, "4 4"]
    stress_energy: Float[Array, "4 4"]


def christoffel_symbols(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "4"],
) -> Float[Array, "4 4 4"]:
    """Christoffel symbols of the second kind at a single spacetime point.

    Gamma^lam_{mu nu} = 1/2 g^{lam sigma} (d_mu g_{nu sigma} + d_nu g_{mu sigma} - d_sigma g_{mu nu})

    Uses jax.jacfwd(metric_fn) for exact partial derivatives.

    Args:
        metric_fn: Callable mapping coords (4,) -> metric tensor (4,4).
        coords: Spacetime coordinates (t, x, y, z) as shape (4,) array.

    Returns:
        Christoffel symbols of shape (4, 4, 4) with index convention
        [upper, lower, lower] = Gamma^lam_{mu nu}.
    """
    g = metric_fn(coords)
    g_inv = jnp.linalg.inv(g)
    # dg[a, b, c] = d g_{ab} / d x^c  (derivative index is LAST per JAX jacfwd convention)
    dg = jax.jacfwd(metric_fn)(coords)

    # Gamma^lam_{mu nu} = 0.5 * g^{lam sig} * (d_mu g_{nu sig} + d_nu g_{mu sig} - d_sig g_{mu nu})
    # In dg indexing: d_mu g_{nu sig} = dg[nu, sig, mu]
    term1 = jnp.einsum('ls,nsm->lmn', g_inv, dg)  # g^{ls} d_m g_{ns}
    term2 = jnp.einsum('ls,msn->lmn', g_inv, dg)  # g^{ls} d_n g_{ms}
    term3 = jnp.einsum('ls,mns->lmn', g_inv, dg)  # g^{ls} d_s g_{mn}

    return 0.5 * (term1 + term2 - term3)


def riemann_tensor(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "4"],
) -> Float[Array, "4 4 4 4"]:
    """Riemann curvature tensor at a single spacetime point.

    R^lam_{mu nu rho} = d_nu Gamma^lam_{mu rho} - d_rho Gamma^lam_{mu nu}
                        + Gamma^lam_{sig nu} Gamma^sig_{mu rho}
                        - Gamma^lam_{sig rho} Gamma^sig_{mu nu}

    Uses nested jax.jacfwd (forward-over-forward) for exact second derivatives.

    Args:
        metric_fn: Callable mapping coords (4,) -> metric tensor (4,4).
        coords: Spacetime coordinates as shape (4,) array.

    Returns:
        Riemann tensor of shape (4, 4, 4, 4) with index convention
        R^lam_{mu nu rho}.
    """
    def gamma_at(x):
        return christoffel_symbols(metric_fn, x)

    gamma = gamma_at(coords)
    # dgamma[l, m, n, s] = d Gamma^l_{mn} / d x^s  (derivative index LAST)
    dgamma = jax.jacfwd(gamma_at)(coords)

    # d_n Gamma^l_{mr} = dgamma[l, m, r, n] -> swap axes 2,3 of dgamma
    # d_r Gamma^l_{mn} = dgamma[l, m, n, r] -> dgamma itself
    deriv_term = jnp.swapaxes(dgamma, 2, 3) - dgamma

    # Quadratic terms
    quad_pos = jnp.einsum('lsn,smr->lmnr', gamma, gamma)  # Gamma^l_{sn} Gamma^s_{mr}
    quad_neg = jnp.einsum('lsr,smn->lmnr', gamma, gamma)  # Gamma^l_{sr} Gamma^s_{mn}

    return deriv_term + quad_pos - quad_neg


def ricci_tensor(riemann: Float[Array, "4 4 4 4"]) -> Float[Array, "4 4"]:
    """Ricci tensor via contraction of Riemann tensor.

    R_{mu rho} = R^lam_{mu lam rho} (trace on first and third indices).

    Args:
        riemann: Riemann tensor of shape (4, 4, 4, 4).

    Returns:
        Ricci tensor of shape (4, 4).
    """
    return jnp.einsum('lmlr->mr', riemann)


def ricci_scalar(
    g_inv: Float[Array, "4 4"],
    ricci: Float[Array, "4 4"],
) -> Float[Array, ""]:
    """Ricci scalar via contraction with inverse metric.

    R = g^{mu nu} R_{mu nu}.

    Args:
        g_inv: Inverse metric tensor of shape (4, 4).
        ricci: Ricci tensor of shape (4, 4).

    Returns:
        Ricci scalar (scalar array).
    """
    return jnp.einsum('mn,mn->', g_inv, ricci)


def einstein_tensor(
    ricci: Float[Array, "4 4"],
    scalar: Float[Array, ""],
    g: Float[Array, "4 4"],
) -> Float[Array, "4 4"]:
    """Einstein tensor.

    G_{mu nu} = R_{mu nu} - 1/2 R g_{mu nu}.

    Args:
        ricci: Ricci tensor of shape (4, 4).
        scalar: Ricci scalar.
        g: Metric tensor of shape (4, 4).

    Returns:
        Einstein tensor of shape (4, 4).
    """
    return ricci - 0.5 * scalar * g


def stress_energy_tensor(einstein: Float[Array, "4 4"]) -> Float[Array, "4 4"]:
    """Stress-energy tensor from Einstein field equations (geometric units G=c=1).

    T_{mu nu} = G_{mu nu} / (8 pi).

    Args:
        einstein: Einstein tensor of shape (4, 4).

    Returns:
        Stress-energy tensor of shape (4, 4).
    """
    return einstein / (8.0 * jnp.pi)


def compute_curvature_chain(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "4"],
) -> CurvatureResult:
    """Compute the full curvature chain at a single spacetime point.

    Evaluates: metric -> inverse metric -> Christoffel -> Riemann -> Ricci
    -> Ricci scalar -> Einstein -> stress-energy.

    All derivatives are computed via jax.jacfwd (exact autodiff).
    JIT-compatible.

    Args:
        metric_fn: Callable mapping coords (4,) -> metric tensor (4,4).
        coords: Spacetime coordinates as shape (4,) array.

    Returns:
        CurvatureResult NamedTuple with all 8 tensors.
    """
    g = metric_fn(coords)
    g_inv = jnp.linalg.inv(g)
    gamma = christoffel_symbols(metric_fn, coords)
    R_tensor = riemann_tensor(metric_fn, coords)
    Ric = ricci_tensor(R_tensor)
    R_scalar = ricci_scalar(g_inv, Ric)
    G = einstein_tensor(Ric, R_scalar, g)
    T = stress_energy_tensor(G)

    return CurvatureResult(
        metric=g,
        metric_inv=g_inv,
        christoffel=gamma,
        riemann=R_tensor,
        ricci=Ric,
        ricci_scalar=R_scalar,
        einstein=G,
        stress_energy=T,
    )
