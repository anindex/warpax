r"""Closed-form worst-case observer for Type-I stress-energy.

For Type-I matter with eigenframe ``T^a_b = diag(-rho, p_1, p_2, p_3)``, the
energy density measured by an observer obtained from the rest frame by a boost of
rapidity ``zeta`` along principal axis ``i`` is

    rho_obs(zeta) = rho cosh^2(zeta) + p_i sinh^2(zeta)
                  = rho + (rho + p_i) sinh^2(zeta).

Hence:

- If ``rho + p_i >= 0`` for every ``i`` (NEC holds), the minimum over observers is
  ``rho`` at ``zeta = 0`` (the rest/Eulerian frame already sees the least energy).
- If ``rho + p_{i*} < 0`` for some axis ``i* = argmin_i (rho + p_i)``, then
  ``rho_obs -> -infinity`` as ``zeta -> infinity`` along ``e_{i*}``: the WEC/NEC
  worst observer is UNBOUNDED in rapidity, the violation magnitude is unbounded,
  and any observer boosted past the threshold

    sinh^2(zeta_th) = rho / |rho + p_{i*}|   (when rho > 0)

  measures negative energy density.

This makes precise the referee's point (CQG-115130.R2 pts 2,6): the "worst-case
non-Eulerian observer" is not a numerical discovery of a rapidity-capped
optimizer but a closed-form consequence of the eigenstructure of ``T^a_b``. The
worst spatial boost direction is the principal eigenvector ``e_{i*}`` of the
most-violating principal pressure, and the threshold rapidity is the closed form
above. This module returns exactly those quantities; it is validated against the
BFGS optimizer in :mod:`.optimization` (see ``tests/test_worst_observer_analytic``).

Everything here is pure-JAX, vmappable, and -- like
:mod:`.frame_free` -- never constructs the Eulerian normal, so it is valid at all
warp velocities including ``v_s >= 1``.

Scope: the closed form gives the worst DIRECTION, the THRESHOLD rapidity, and the
asymptotic SIGN. It does not return a finite "worst energy density" because, when
``rho + p_{i*} < 0``, that extremum is ``-infinity`` (the divergence the
``zeta_max``-capped optimizer can only approximate). DEC is handled at the
eigenvalue-bound level (``rho >= |p_i|``); its worst axis is ``argmax_i |p_i|``.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

_VALID = frozenset({"nec", "wec", "sec", "dec"})


def boosted_energy_density(
    rho: Float[Array, ""],
    p_axis: Float[Array, ""],
    zeta: Float[Array, ""],
) -> Float[Array, ""]:
    """``rho_obs(zeta) = rho + (rho + p_axis) sinh^2(zeta)`` (boost along one axis)."""
    return rho + (rho + p_axis) * jnp.sinh(zeta) ** 2


def worst_observer_typeI(
    eigenvalues: Float[Array, "4"],
    eigenvectors: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    *,
    condition: str = "wec",
    atol: float = 1e-12,
) -> dict:
    r"""Closed-form worst observer for a single Type-I point.

    Parameters
    ----------
    eigenvalues : Float[Array, "4"]
        Real eigenvalues of ``T^a_b`` (as returned by the classifier).
    eigenvectors : Float[Array, "4 4"]
        Eigenvectors as COLUMNS (``eigenvectors[:, k]`` is the k-th).
    g_ab : Float[Array, "4 4"]
        Covariant metric, used to (a) identify the timelike eigenvector and
        (b) normalise the boost frame. No Eulerian normal is used, so this is
        valid at ``v_s >= 1``.
    condition : {"nec", "wec", "sec", "dec"}
        Which condition's worst axis to return. NEC/WEC/SEC use the
        most-negative ``rho + p_i`` axis; DEC uses the largest ``|p_i|`` axis.
    atol : float
        Slack threshold below which a margin counts as violated.

    Returns
    -------
    dict of scalars/vectors
        ``rho``, ``worst_axis`` (0-2 among spacelike), ``p_star``,
        ``margin`` (``rho+p_star`` for NEC/WEC/SEC; ``rho-|p_star|`` for DEC),
        ``zeta_th`` (threshold rapidity; 0 if already violated in rest frame;
        ``inf`` if never violated), ``asymptotic_sign`` (sign of the energy
        density as ``zeta -> inf`` along the worst axis),
        ``boost_direction`` (unit spacelike 4-vector ``e_{i*}``),
        ``rest_frame`` (unit timelike 4-vector ``e_0``),
        ``worst_observer`` (``cosh(z) e_0 + sinh(z) e_{i*}`` at a finite
        reference ``z = min(zeta_th, 8)``).
    """
    if condition not in _VALID:
        raise ValueError(f"condition must be in {_VALID}; got {condition!r}")

    # Causal character of each eigenvector: g_{ab} v^a v^b. Timelike < 0.
    causal = jnp.einsum("ab,ak,bk->k", g_ab, eigenvectors, eigenvectors)
    timelike_idx = jnp.argmin(causal)
    rho = -eigenvalues[timelike_idx]

    idx = jnp.arange(4)
    is_spacelike = idx != timelike_idx

    if condition == "dec":
        # Worst DEC axis: largest |p_i| (eigenvalue bound rho >= |p_i|).
        score = jnp.where(is_spacelike, jnp.abs(eigenvalues), -jnp.inf)
        worst_axis = jnp.argmax(score)
        p_star = eigenvalues[worst_axis]
        margin = rho - jnp.abs(p_star)
    else:
        # Worst NEC/WEC/SEC axis: most-negative (rho + p_i).
        slack = jnp.where(is_spacelike, rho + eigenvalues, jnp.inf)
        worst_axis = jnp.argmin(slack)
        p_star = eigenvalues[worst_axis]
        margin = rho + p_star

    # Threshold rapidity (WEC/NEC interpretation rho_obs(zeta) = 0).
    # rho_obs = rho + (rho + p_star) sinh^2(zeta); set to zero.
    rho_plus_p = rho + p_star
    violated = rho_plus_p < -atol
    # sinh^2 zeta_th = -rho / (rho + p_star) = rho / |rho + p_star|  (rho > 0)
    ratio = rho / jnp.maximum(-rho_plus_p, atol)
    zeta_th = jnp.where(
        violated & (rho > 0.0),
        jnp.arcsinh(jnp.sqrt(jnp.maximum(ratio, 0.0))),
        jnp.where(violated, 0.0, jnp.inf),  # rho<=0: already violated at rest
    )
    asymptotic_sign = jnp.sign(rho_plus_p)

    # Normalised boost frame (no Eulerian normal): e_0 timelike, e_i* spacelike.
    v0 = eigenvectors[:, timelike_idx]
    vi = eigenvectors[:, worst_axis]
    n0 = jnp.sqrt(jnp.maximum(-causal[timelike_idx], atol))
    ni = jnp.sqrt(jnp.maximum(causal[worst_axis], atol))
    e0 = v0 / n0
    ei = vi / ni

    z_ref = jnp.minimum(jnp.where(jnp.isinf(zeta_th), 8.0, zeta_th), 8.0)
    worst_u = jnp.cosh(z_ref) * e0 + jnp.sinh(z_ref) * ei

    return {
        "rho": rho,
        "worst_axis": worst_axis,
        "p_star": p_star,
        "margin": margin,
        "zeta_th": zeta_th,
        "asymptotic_sign": asymptotic_sign,
        "boost_direction": ei,
        "rest_frame": e0,
        "worst_observer": worst_u,
    }
