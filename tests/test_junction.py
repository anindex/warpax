"""Darmois-Israel junction conditions and extended shell tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric, SchwarzschildMetric
from warpax.geometry.metric import MetricSpecification
from warpax.junction import DarmoisResult, darmois, surface_stress_energy
from warpax.junction.darmois import _induced_and_extrinsic, _unit_normal
from warpax.metrics import WarpShellMetric


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
        inner = jnp.array([0.0, 2.95, 0.0, 0.0])
        outer = jnp.array([0.0, 3.05, 0.0, 0.0])
        result = darmois(
            AlcubierreMetric(),
            boundary_fn,
            probe_coords_inside=inner,
            probe_coords_outside=outer,
        )
        assert isinstance(result, DarmoisResult)
        assert float(result.first_form_discontinuity) < 1e-6
        assert float(result.second_form_discontinuity) < 1e-6
        ff, sf, phys = result
        assert float(ff) == float(result.first_form_discontinuity)
        assert float(sf) == float(result.second_form_discontinuity)
        assert bool(phys) == bool(result.physical)

    def test_warpshell_matter_shell_boundary(self):
        """WarpShell at ``R_1``: Darmois pins at 5% probe separation, and the
        Israel stress vanishes with the separation.

        Each side is extrapolated to Sigma. Differencing the raw probes
        instead read 0.11036642755306081 and 0.278833872139782, which carried
        the metric's own gradient across the gap rather than the jump.
        WarpShellMetric is C2 at ``R_1``, so the extrapolated ``[[K]]`` is
        ``O(s^2)`` and halving the separation (5e-4 to 2.5e-4 of ``R_1``)
        must cut ``max|S|`` by at least 3 (measured 2.49e-6 to 6.27e-7).
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
            float(result.first_form_discontinuity), 0.10526042346782394, rtol=1e-6
        )
        np.testing.assert_allclose(
            float(result.second_form_discontinuity), 0.07215228609795868, rtol=1e-6
        )
        assert bool(result.physical) is False

        def max_S(d):
            i = jnp.array([0.0, (1.0 - d) * R_1, 0.0, 0.0])
            o = jnp.array([0.0, (1.0 + d) * R_1, 0.0, 0.0])
            S_ab = surface_stress_energy(metric, boundary_fn, i, o)
            assert jnp.allclose(S_ab, S_ab.T, atol=1e-14)
            return float(jnp.abs(S_ab).max())

        coarse, fine = max_S(5e-4), max_S(2.5e-4)
        assert np.isfinite(coarse) and np.isfinite(fine)
        assert coarse > 3.0 * fine

    def test_alcubierre_smooth_at_default_boundary(self):
        """Default probes (0.9, 1.1 - inside / outside unit-radius wall)
        return a finite, reproducible DarmoisResult.

        The Alcubierre wall has strong but smooth curvature at r~1; the
        discontinuity is then finite-difference-limited rather than a
        true shell discontinuity.
        """
        boundary_fn = lambda c: c[1] - 1.0
        result = darmois(AlcubierreMetric(), boundary_fn)
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


def test_surface_stress_energy_vacuum():
    """Vacuum region should have zero surface stress-energy."""
    metric = MinkowskiMetric()
    boundary_fn = lambda coords: coords[1] - 5.0
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
    boundary_fn = lambda c: c[1]
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
    metric, and the exact ``h^{ab}``-trace in the Israel jump. Probe-based
    finite difference carries ~5% error on this scale.
    """
    metric_in = MinkowskiMetric()
    metric_out = SchwarzschildMetric(M=M)
    boundary_fn = lambda c: jnp.sqrt(c[1] ** 2 + c[2] ** 2 + c[3] ** 2) - R
    inside = jnp.array([0.0, 0.99 * R, 0.0, 0.0])
    outside = jnp.array([0.0, 1.01 * R, 0.0, 0.0])

    h_in, K_in, eps_in, hup_in = _induced_and_extrinsic(metric_in, boundary_fn, inside)
    h_out, K_out, eps_out, hup_out = _induced_and_extrinsic(metric_out, boundary_fn, outside)
    assert float(eps_in) == float(eps_out) == 1.0

    h_avg = 0.5 * (h_in + h_out)
    h_up = 0.5 * (hup_in + hup_out)
    delta_K = K_out - K_in
    delta_K_trace = jnp.einsum("ab,ab->", h_up, delta_K)
    S_ab = -(eps_in / (8.0 * jnp.pi)) * (delta_K - delta_K_trace * h_avg)

    sigma_predicted = _surface_density_analytical(M, R)
    sigma_measured = float(S_ab[0, 0])
    assert abs(sigma_measured - sigma_predicted) < 0.1 * abs(sigma_predicted) + 1e-6


def test_minkowski_sphere_surface_stress_vanishes():
    """Minkowski with ``f = r - 1`` has no shell, so ``S_ab`` must vanish.

    Raw probes at r = 0.9 and 1.1 gave ``S_00 = 0.01607626`` because
    ``K_tangent ~ 1/r`` was differenced on different spheres; one-sided
    limits at the same sphere give ``max|S| < 1e-5`` (measured 1.2e-7 at
    probes 0.99 and 1.01).
    """
    metric = MinkowskiMetric()
    boundary_fn = lambda c: jnp.sqrt(c[1] ** 2 + c[2] ** 2 + c[3] ** 2) - 1.0
    inside = jnp.array([0.0, 0.99, 0.0, 0.0])
    outside = jnp.array([0.0, 1.01, 0.0, 0.0])
    S_ab = surface_stress_energy(metric, boundary_fn, inside, outside)
    assert bool(jnp.isfinite(S_ab).all())
    assert float(jnp.abs(S_ab).max()) < 1e-5


def test_null_boundary_gives_nan():
    """``f = t - x`` in Minkowski has a null gradient, ``g(n, n) = 0``.

    No unit normal exists, so ``_unit_normal`` returns NaN ``epsilon`` and
    ``surface_stress_energy`` returns an all-NaN tensor instead of the
    spacelike-normal answer a ``1e-30`` floor would produce.
    """
    metric = MinkowskiMetric()
    boundary_fn = lambda c: c[0] - c[1]
    _, _, epsilon = _unit_normal(metric, boundary_fn, jnp.zeros(4))
    assert bool(jnp.isnan(epsilon))
    inside = jnp.array([0.0, 0.5, 0.0, 0.0])
    outside = jnp.array([0.0, -0.5, 0.0, 0.0])
    S_ab = surface_stress_energy(metric, boundary_fn, inside, outside)
    assert bool(jnp.isnan(S_ab).all())


def test_swapped_probes_give_nan():
    """Israel's normal points from inside to outside, so probes with
    ``boundary_fn(inside) > 0 > boundary_fn(outside)`` violate the
    documented orientation and ``surface_stress_energy`` returns NaN.
    """
    metric = MinkowskiMetric()
    boundary_fn = lambda c: jnp.sqrt(c[1] ** 2 + c[2] ** 2 + c[3] ** 2) - 1.0
    inside = jnp.array([0.0, 1.01, 0.0, 0.0])
    outside = jnp.array([0.0, 0.99, 0.0, 0.0])
    S_ab = surface_stress_energy(metric, boundary_fn, inside, outside)
    assert bool(jnp.isnan(S_ab).all())


class _NullPlaneMetric(MetricSpecification):
    """``g = [[-x^2, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]``: the plane ``x = 0`` is null."""

    def __call__(self, coords):
        x = coords[1]
        g = jnp.eye(4).at[0, 0].set(-(x**2)).at[1, 1].set(0.0)
        return g.at[0, 1].set(1.0).at[1, 0].set(1.0)

    def shape_function_value(self, coords):
        return jnp.array(0.0)

    def symbolic(self):
        raise NotImplementedError

    def name(self):
        return "NullPlane"


@pytest.mark.parametrize("eps", [0.01, 0.001])
def test_null_boundary_with_spacelike_probes_gives_nan(eps):
    """A boundary that is null only on Sigma itself returns an all-NaN ``S_ab``.

    For ``g = [[-x^2, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]``
    with ``f = x``, ``g^{xx} = x^2`` so the normal is null at ``x = 0`` and
    spacelike at every probe ``x = +-eps``. The metric is smooth with no
    shell, so extrapolating from the probes alone would report a large
    fictitious ``S_xx`` growing like ``1/eps^2``; the Sigma-point causal
    check rejects the input instead.
    """

    metric = _NullPlaneMetric()
    boundary_fn = lambda c: c[1]
    inside = jnp.array([0.0, -eps, 0.0, 0.0])
    outside = jnp.array([0.0, eps, 0.0, 0.0])
    _, _, eps_probe = _unit_normal(metric, boundary_fn, outside)
    assert float(eps_probe) == 1.0
    S_ab = surface_stress_energy(metric, boundary_fn, inside, outside)
    assert bool(jnp.isnan(S_ab).all())
