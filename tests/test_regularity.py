"""Tests for the geometry.regularity module.

Validates C^k continuity diagnostics against analytical metrics with
known smoothness properties.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class TestRegularityDiagnostics:
    """Verify the metric regularity diagnostic module."""

    def test_minkowski_passes_c2(self):
        """Minkowski metric is trivially C^inf (all jumps near zero)."""
        from warpax.benchmarks import MinkowskiMetric
        from warpax.geometry import regularity_report

        report = regularity_report(MinkowskiMetric(), r_min=5.0, r_max=25.0)
        assert report.is_c2

        for name, diag in report.components.items():
            assert diag.c0_max_jump < 1.0, f"{name} C0 jump = {diag.c0_max_jump}"
            assert diag.c1_max_jump < 1.0, f"{name} C1 jump = {diag.c1_max_jump}"

    def test_schwarzschild_passes_c2(self):
        """Schwarzschild metric is C^inf away from the horizon."""
        from warpax.benchmarks import SchwarzschildMetric
        from warpax.geometry import regularity_report

        report = regularity_report(
            SchwarzschildMetric(M=1.0),
            r_min=5.0, r_max=25.0,
        )
        assert report.is_c2

    def test_c1_smoothstep_has_larger_c2_jumps(self):
        """C1 cubic smoothstep produces larger C^2 jumps than C2 quintic.

        This tests the module's ability to discriminate between smoothness
        levels. The C1 cubic has f''(0)=6, f''(1)=-6 at transition
        boundaries, so its C2 diagnostic should show larger jumps.
        """
        from warpax.geometry import metric_c2_diagnostic
        from warpax.metrics import WarpShellPhysical

        r_vals = jnp.linspace(5.0, 25.0, 200)

        m_c1 = WarpShellPhysical(
            v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0,
            transition_order=1,
        )
        m_c2 = WarpShellPhysical(
            v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0,
            transition_order=2,
        )

        diag_c1 = metric_c2_diagnostic(m_c1, r_vals, component=(0, 0))
        diag_c2 = metric_c2_diagnostic(m_c2, r_vals, component=(0, 0))

        assert diag_c1.c2_max_jump > diag_c2.c2_max_jump, \
            f"C1 jump ({diag_c1.c2_max_jump:.1f}) should exceed " \
            f"C2 jump ({diag_c2.c2_max_jump:.1f})"
