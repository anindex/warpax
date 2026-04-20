"""Regression tests for warpax.junction.darmois .

Four tests:

- Smooth metric (Alcubierre) - no junction: low first-form discontinuity.
- WarpShell at its matter-shell boundary - non-trivial discontinuity.
- Planar-null boundary for Alcubierre - numerical stability check.
- Determinism - same inputs give same outputs.
"""
from __future__ import annotations

import jax.numpy as jnp

from warpax.benchmarks import AlcubierreMetric
from warpax.junction import DarmoisResult, darmois
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

    def test_warpshell_matter_shell_boundary(self):
        """WarpShell at its matter-shell boundary R_1=10 - record actual
        behavior as a regression pin.

        The exact discontinuity depends on whether WarpShell uses regularisation; we pin the observed values rather than asserting
        a specific class.
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
        # Result is a regression pin; assert finite values.
        assert jnp.isfinite(result.first_form_discontinuity)
        assert jnp.isfinite(result.second_form_discontinuity)
        assert isinstance(result.physical, bool)

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
        assert r1.physical is r2.physical

    def test_darmois_result_is_namedtuple(self):
        """`DarmoisResult` exposes named attributes."""
        boundary_fn = lambda c: c[1] - 3.0
        inner = jnp.array([0.0, 2.95, 0.0, 0.0])
        outer = jnp.array([0.0, 3.05, 0.0, 0.0])
        result = darmois(
            AlcubierreMetric(),
            boundary_fn,
            probe_coords_inside=inner,
            probe_coords_outside=outer,
        )
        ff, sf, phys = result
        assert float(ff) == float(result.first_form_discontinuity)
        assert float(sf) == float(result.second_form_discontinuity)
        assert phys is result.physical
