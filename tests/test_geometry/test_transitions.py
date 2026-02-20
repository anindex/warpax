"""Tests for shared smoothstep transition functions."""

import jax
import jax.numpy as jnp
import pytest

from warpax.geometry.transitions import smoothstep, smoothstep_c1, smoothstep_c2


class TestSmoothstepTransitions:
    """Tests for C1/C2 smoothstep functions."""

    # ------------------------------------------------------------------
    # Boundary values
    # ------------------------------------------------------------------

    def test_smoothstep_c1_boundary_values(self):
        """C1 cubic: f(0)=0, f(1)=1, f(0.5)=0.5."""
        assert jnp.isclose(smoothstep_c1(jnp.array(0.0)), 0.0, atol=1e-15)
        assert jnp.isclose(smoothstep_c1(jnp.array(1.0)), 1.0, atol=1e-15)
        assert jnp.isclose(smoothstep_c1(jnp.array(0.5)), 0.5, atol=1e-15)

    def test_smoothstep_c2_boundary_values(self):
        """C2 quintic: f(0)=0, f(1)=1, f(0.5)=0.5."""
        assert jnp.isclose(smoothstep_c2(jnp.array(0.0)), 0.0, atol=1e-15)
        assert jnp.isclose(smoothstep_c2(jnp.array(1.0)), 1.0, atol=1e-15)
        assert jnp.isclose(smoothstep_c2(jnp.array(0.5)), 0.5, atol=1e-15)

    # ------------------------------------------------------------------
    # First derivative at endpoints
    # ------------------------------------------------------------------

    def test_smoothstep_c1_first_derivative_zero(self):
        """C1 cubic: f'(0)=0, f'(1)=0."""
        grad_fn = jax.grad(lambda t: smoothstep_c1(t))
        # Evaluate at points just inside [0, 1] to avoid clipping boundary
        # effects. The analytical derivative of 3t^2 - 2t^3 is 6t - 6t^2.
        assert jnp.isclose(grad_fn(jnp.array(0.0)), 0.0, atol=1e-10)
        assert jnp.isclose(grad_fn(jnp.array(1.0)), 0.0, atol=1e-10)

    def test_smoothstep_c2_first_derivative_zero(self):
        """C2 quintic: f'(0)=0, f'(1)=0."""
        grad_fn = jax.grad(lambda t: smoothstep_c2(t))
        assert jnp.isclose(grad_fn(jnp.array(0.0)), 0.0, atol=1e-10)
        assert jnp.isclose(grad_fn(jnp.array(1.0)), 0.0, atol=1e-10)

    # ------------------------------------------------------------------
    # Second derivative at endpoints (THE key C2 property)
    # ------------------------------------------------------------------

    def test_smoothstep_c2_second_derivative_zero(self):
        """C2 quintic: f''(0)=0, f''(1)=0.

        This is THE defining property of C2 smoothness. The second
        derivative 120t^3 - 180t^2 + 60t vanishes at both endpoints.
        """
        grad2_fn = jax.grad(jax.grad(lambda t: smoothstep_c2(t)))
        assert jnp.isclose(grad2_fn(jnp.array(0.0)), 0.0, atol=1e-10), (
            f"C2 f''(0) = {grad2_fn(jnp.array(0.0))}, expected 0.0"
        )
        assert jnp.isclose(grad2_fn(jnp.array(1.0)), 0.0, atol=1e-10), (
            f"C2 f''(1) = {grad2_fn(jnp.array(1.0))}, expected 0.0"
        )

    def test_smoothstep_c1_second_derivative_nonzero(self):
        """C1 cubic: f''(eps) is large (NOT near zero confirms C1 is not C2).

        The second derivative of 3t^2 - 2t^3 is 6 - 12t, which gives
        f''(0)=6 analytically. At the exact clip boundary, JAX autodiff
        returns a modified value due to the clip gradient convention, so
        we evaluate at a point just inside [0, 1].
        """
        grad2_fn = jax.grad(jax.grad(lambda t: smoothstep_c1(t)))
        # Use a point just inside the domain to avoid clip boundary effects
        f2_near_0 = grad2_fn(jnp.array(0.01))
        # f''(0.01) = 6 - 12*0.01 = 5.88
        assert jnp.isclose(f2_near_0, 5.88, atol=0.01), (
            f"C1 f''(0.01) = {f2_near_0}, expected ~5.88"
        )
        assert not jnp.isclose(f2_near_0, 0.0, atol=1.0), (
            "C1 f''(0.01) should NOT be near zero"
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def test_smoothstep_dispatch(self):
        """smoothstep(t, order=N) matches the corresponding function."""
        t = jnp.linspace(0.0, 1.0, 50)
        assert jnp.allclose(smoothstep(t, order=1), smoothstep_c1(t), atol=1e-15)
        assert jnp.allclose(smoothstep(t, order=2), smoothstep_c2(t), atol=1e-15)

    def test_smoothstep_invalid_order(self):
        """smoothstep(t, order=3) raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported smoothstep order"):
            smoothstep(jnp.array(0.5), order=3)

    # ------------------------------------------------------------------
    # Clipping and monotonicity
    # ------------------------------------------------------------------

    def test_smoothstep_clipping(self):
        """Values outside [0,1] are clipped: f(-0.5)=0, f(1.5)=1."""
        for fn in [smoothstep_c1, smoothstep_c2]:
            assert jnp.isclose(fn(jnp.array(-0.5)), 0.0, atol=1e-15), (
                f"{fn.__name__}(-0.5) should be 0.0"
            )
            assert jnp.isclose(fn(jnp.array(1.5)), 1.0, atol=1e-15), (
                f"{fn.__name__}(1.5) should be 1.0"
            )

    def test_smoothstep_monotonic(self):
        """Both C1 and C2 are monotonically non-decreasing on [0, 1]."""
        t = jnp.linspace(0.0, 1.0, 100)
        for fn in [smoothstep_c1, smoothstep_c2]:
            vals = fn(t)
            diffs = jnp.diff(vals)
            assert jnp.all(diffs >= -1e-15), (
                f"{fn.__name__} is not monotonic: min diff = {jnp.min(diffs)}"
            )
