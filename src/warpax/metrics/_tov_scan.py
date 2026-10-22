"""Inward RK4 integration of the TOV pressure equation on a uniform radial grid."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _tov_rhs(
    r: Float[Array, ""],
    p: Float[Array, ""],
    rho: Float[Array, ""],
    m: Float[Array, ""],
) -> Float[Array, ""]:
    r_safe = jnp.maximum(jnp.abs(r), 1e-30)
    denom = r_safe * (r_safe - 2.0 * m)
    denom_safe = jnp.where(
        jnp.abs(denom) < 1e-30,
        jnp.where(denom >= 0.0, 1e-30, -1e-30),
        denom,
    )
    return -(rho + p) * (m + 4.0 * jnp.pi * r_safe**3 * p) / denom_safe


def integrate_tov_inward(
    r_desc: Float[Array, "N"],
    rho_desc: Float[Array, "N"],
    m_desc: Float[Array, "N"],
    *,
    rho_mid: Float[Array, "M"] | None = None,
    m_mid: Float[Array, "M"] | None = None,
) -> Float[Array, "N"]:
    """``p_r`` on ``r_desc``, from ``p_r = 0`` at the outer boundary ``r_desc[0]``.

    The grid descends and is uniform. ``rho_mid`` and ``m_mid`` are the density
    and mass at the RK4 midpoints; when omitted they are linearly interpolated
    from adjacent samples. The scheme is fourth order with exact midpoints and
    second order with interpolated ones. Pressure is floored at zero, so an
    inward step never drives it negative. If any sample is trapped
    (``r <= 2 m``) the whole output is NaN.
    """
    h = r_desc[1] - r_desc[0]
    if rho_mid is None:
        rho_mid = 0.5 * (rho_desc[:-1] + rho_desc[1:])
    if m_mid is None:
        m_mid = 0.5 * (m_desc[:-1] + m_desc[1:])
    ok = jnp.all(r_desc > 2.0 * m_desc)

    def step(p, inputs):
        r_a, rho_a, m_a, r_b, rho_b, m_b, rho_m, m_m = inputs
        r_mid = r_a + 0.5 * h
        k1 = _tov_rhs(r_a, p, rho_a, m_a)
        k2 = _tov_rhs(r_mid, p + 0.5 * h * k1, rho_m, m_m)
        k3 = _tov_rhs(r_mid, p + 0.5 * h * k2, rho_m, m_m)
        k4 = _tov_rhs(r_b, p + h * k3, rho_b, m_b)
        p_next = jnp.maximum(p + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0)
        return p_next, p_next

    _, p_scan = jax.lax.scan(
        step,
        jnp.float64(0.0),
        (
            r_desc[:-1],
            rho_desc[:-1],
            m_desc[:-1],
            r_desc[1:],
            rho_desc[1:],
            m_desc[1:],
            rho_mid,
            m_mid,
        ),
    )
    p = jnp.concatenate([jnp.array([0.0]), p_scan])
    return jnp.where(ok, p, jnp.nan)
