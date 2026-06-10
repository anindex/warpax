"""Darmois-Israel junction conditions and extended shell tests."""

from __future__ import annotations
from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric, SchwarzschildMetric
from warpax.junction import DarmoisResult, darmois
from warpax.junction import surface_stress_energy
from warpax.junction.darmois import _induced_and_extrinsic, _unit_normal
from warpax.metrics import WarpShellMetric
import jax
import jax.numpy as jnp
import numpy as np
import pytest



class TestDarmois:
    """Darmois junction checker regression tests.

    - Alcubierre (smooth): both discontinuities small when probes
      bracket the boundary closely on either side of the smooth region.
    - WarpShell (matter shell): discontinuities non-zero; we record the
      actual behavior for regression pinning.
    - JIT path: no errors on Alcubierre probe.
    - Determinism: same inputs produce same outputs bit-identically.
    """

    def test_alcubierre_is_smooth_far_outside_bubble(self):
        """Alcubierre is C^infty; at x=3 (well outside bubble wall),
        first-form discontinuity should be ~ 0 (Minkowski-like)."""
        boundary_fn = lambda c: c[1] - 3.0
        # Probes close to boundary (both in Minkowski-like region)
        inner = jnp.array([0.0, 2.95, 0.0, 0.0])
        outer = jnp.array([0.0, 3.05, 0.0, 0.0])
        result = darmois(
            AlcubierreMetric(),
            boundary_fn,
            probe_coords_inside=inner,
            probe_coords_outside=outer,
        )
        assert isinstance(result, DarmoisResult)
        # Far outside bubble => Minkowski; discontinuity near machine epsilon.
        assert float(result.first_form_discontinuity) < 1e-6
        assert float(result.second_form_discontinuity) < 1e-6
        # NamedTuple unpack order matches the named attributes.
        ff, sf, phys = result
        assert float(ff) == float(result.first_form_discontinuity)
        assert float(sf) == float(result.second_form_discontinuity)
        assert bool(phys) == bool(result.physical)

    def test_warpshell_matter_shell_boundary(self):
        """WarpShell at its matter-shell boundary ``R_1`` - golden snapshot
        of the actual discontinuity values (measured, regression pin).
        """
        metric = WarpShellMetric()
        R_1 = metric.R_1
        boundary_fn = lambda c: jnp.sqrt(c[1] ** 2 + c[2] ** 2 + c[3] ** 2) - R_1
        inner = jnp.array([0.0, 0.95 * R_1, 0.0, 0.0])
        outer = jnp.array([0.0, 1.05 * R_1, 0.0, 0.0])
        result = darmois(
            metric,
            boundary_fn,
            probe_coords_inside=inner,
            probe_coords_outside=outer,
        )
        np.testing.assert_allclose(
            float(result.first_form_discontinuity), 0.11036642755306081, rtol=1e-6
        )
        np.testing.assert_allclose(
            float(result.second_form_discontinuity), 0.278833872139782, rtol=1e-6
        )
        # Genuine matter shell: the jump exceeds tolerance, so not physical.
        assert bool(result.physical) is False

        # S_ab is nonzero here, so symmetry is actually informative.
        S_ab = surface_stress_energy(metric, boundary_fn, inner, outer)
        assert float(jnp.abs(S_ab).max()) > 1e-3
        assert jnp.allclose(S_ab, S_ab.T, atol=1e-14)

    def test_alcubierre_smooth_at_default_boundary(self):
        """Default probes (0.9, 1.1 - inside / outside unit-radius wall)
        return a finite, reproducible DarmoisResult.

        The Alcubierre wall has strong but smooth curvature at r~1; the
        discontinuity is then finite-difference-limited rather than a
        genuine shell discontinuity.
        """
        boundary_fn = lambda c: c[1] - 1.0
        result = darmois(AlcubierreMetric(), boundary_fn)
        # Regression pin: finite values.
        assert jnp.isfinite(result.first_form_discontinuity)
        assert jnp.isfinite(result.second_form_discontinuity)

    def test_determinism(self):
        """`darmois` is deterministic on repeated invocation."""
        boundary_fn = lambda c: c[1] - 2.0
        inner = jnp.array([0.0, 1.95, 0.0, 0.0])
        outer = jnp.array([0.0, 2.05, 0.0, 0.0])
        r1 = darmois(AlcubierreMetric(), boundary_fn, inner, outer)
        r2 = darmois(AlcubierreMetric(), boundary_fn, inner, outer)
        assert float(r1.first_form_discontinuity) == float(r2.first_form_discontinuity)
        assert float(r1.second_form_discontinuity) == float(r2.second_form_discontinuity)
        assert bool(r1.physical) == bool(r2.physical)

jax.config.update("jax_enable_x64", True)


def test_surface_stress_energy_vacuum():
    """Vacuum region should have zero surface stress-energy."""
    metric = MinkowskiMetric()
    boundary_fn = lambda coords: coords[1] - 5.0  # x = 5 boundary
    inside = jnp.array([0.0, 4.99, 0.0, 0.0], dtype=jnp.float64)
    outside = jnp.array([0.0, 5.01, 0.0, 0.0], dtype=jnp.float64)
    S_ab = surface_stress_energy(metric, boundary_fn, inside, outside)
    assert S_ab.shape == (4, 4)
    assert jnp.allclose(S_ab, 0.0, atol=1e-8)


def _surface_density_analytical(M: float, R: float) -> float:
    """Israel surface density for an interior-Minkowski / exterior-Schwarzschild shell."""
    return -(1.0 / (4.0 * np.pi * R)) * (np.sqrt(1.0 - 2.0 * M / R) - 1.0)


def test_minkowski_planar_boundary_zero_jump():
    """Same metric on both sides of a planar boundary: ``[h] = [K] = 0`` exactly."""
    metric = MinkowskiMetric()
    boundary_fn = lambda c: c[1]  # planar boundary at x = 0
    inside = jnp.array([0.0, -1.0, 0.0, 0.0])
    outside = jnp.array([0.0, 1.0, 0.0, 0.0])

    result = darmois(metric, boundary_fn, inside, outside)
    assert float(result.first_form_discontinuity) < 1e-10
    assert float(result.second_form_discontinuity) < 1e-10

    S_ab = surface_stress_energy(metric, boundary_fn, inside, outside)
    assert np.allclose(np.asarray(S_ab), 0.0, atol=1e-10)


@pytest.mark.parametrize("M, R", [(0.1, 5.0), (0.2, 10.0)])
def test_schwarzschild_shell_surface_density(M, R):
    """Israel surface density from a two-sided jump matches the analytical value.

    Coverage for the v0.4 junction fixes: covariant ``nabla n`` in the
    extrinsic curvature, ``epsilon = sign(g(n,n))`` on the induced
    metric, and ``h^{ab}``-trace in the Israel jump.
    """
    metric_in = MinkowskiMetric()
    metric_out = SchwarzschildMetric(M=M)
    boundary_fn = lambda c: jnp.sqrt(c[1] ** 2 + c[2] ** 2 + c[3] ** 2) - R
    inside = jnp.array([0.0, 0.99 * R, 0.0, 0.0])
    outside = jnp.array([0.0, 1.01 * R, 0.0, 0.0])

    _h_in, K_in, eps_in = _induced_and_extrinsic(metric_in, boundary_fn, inside)
    _h_out, K_out, eps_out = _induced_and_extrinsic(metric_out, boundary_fn, outside)

    n_in_cov, _, _ = _unit_normal(metric_in, boundary_fn, inside)
    n_out_cov, _, _ = _unit_normal(metric_out, boundary_fn, outside)
    h_in = metric_in(inside) - eps_in * jnp.outer(n_in_cov, n_in_cov)
    h_out = metric_out(outside) - eps_out * jnp.outer(n_out_cov, n_out_cov)
    h_avg = 0.5 * (h_in + h_out)

    delta_K = K_out - K_in
    h_inv_avg = jnp.linalg.pinv(h_avg)
    delta_K_trace = jnp.einsum("ab,ab->", h_inv_avg, delta_K)
    epsilon = 0.5 * (eps_in + eps_out)
    S_ab = -(epsilon / (8.0 * jnp.pi)) * (delta_K - delta_K_trace * h_avg)

    sigma_predicted = _surface_density_analytical(M, R)
    sigma_measured = float(S_ab[0, 0])

    # Probe-based finite difference + per-side autodiff carries ~5% error on
    # this scale; tighten the bound if the integrator gains another digit.
    assert abs(sigma_measured - sigma_predicted) < 0.1 * abs(sigma_predicted) + 1e-6
