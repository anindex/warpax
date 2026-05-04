"""Tests for TOV residuals."""
import jax
import jax.numpy as jnp
from warpax.tov import tov_residual

jax.config.update("jax_enable_x64", True)


def test_tov_residual_isotropic_perfect_fluid():
    """Isotropic fluid (p_r = p_t) with zero derivative should have zero residual."""
    r = jnp.array(1.0)
    rho = lambda rr: jnp.array(1.0)
    p_r = lambda rr: jnp.array(0.5)
    p_t = lambda rr: jnp.array(0.5)
    Phi_prime = jnp.array(0.0)
    result = tov_residual(r, rho, p_r, p_t, Phi_prime)
    assert jnp.isclose(result, 0.0, atol=1e-14)


def test_tov_residual_anisotropic():
    """Anisotropic fluid with balancing terms should have zero residual."""
    r = jnp.array(2.0)
    rho = lambda rr: jnp.array(1.0)
    p_r = lambda rr: jnp.array(0.5)
    p_t = lambda rr: jnp.array(0.5 + 0.1)  # p_t > p_r
    # Choose Phi_prime to balance: Phi' = 2(p_t - p_r) / (r * (rho + p_r))
    Phi_prime = jnp.array(2.0 * 0.1 / (2.0 * (1.0 + 0.5)))
    result = tov_residual(r, rho, p_r, p_t, Phi_prime)
    assert jnp.isclose(result, 0.0, atol=1e-6)


def test_tov_residual_with_gravity():
    """Fluid with non-zero Phi_prime should show non-zero residual if not in equilibrium."""
    r = jnp.array(1.0)
    rho = lambda rr: jnp.array(1.0)
    p_r = lambda rr: jnp.array(0.5)
    p_t = lambda rr: jnp.array(0.5)
    Phi_prime = jnp.array(1.0)
    result = tov_residual(r, rho, p_r, p_t, Phi_prime)
    # For isotropic fluid with constant p_r, dp_r/dr = 0, so residual = (rho + p_r) * Phi'
    expected = (1.0 + 0.5) * 1.0
    assert jnp.isclose(result, expected, atol=1e-6)
