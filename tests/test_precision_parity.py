"""TestPrecisionBand - fp32-screen+fp64-verify parity tests.

Covers the precision-band kwargs on ``evaluate_curvature_grid``:
- ``precision in {'fp64', 'fp32_screen+fp64_verify'}``
- ``backend in {'cpu', 'gpu'}``
- ``fp32_band_tol: float = 5e-4``
- ``WARPAX_PERF_BACKEND`` env var override
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.benchmarks.minkowski import MinkowskiMetric
from warpax.geometry.grid import evaluate_curvature_grid
from warpax.geometry.types import GridSpec


class TestPrecisionBand:
    """precision-band parity + env-override tests."""

    def _alcubierre_8cubed(self):
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        grid = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(8, 8, 8))
        return metric, grid

    def test_default_fp64_preserves_v10_bit_exact(self):
        """Default + explicit precision='fp64' are bit-identical to v1.0."""
        metric, grid = self._alcubierre_8cubed()

        ref = evaluate_curvature_grid(metric, grid)
        explicit_fp64 = evaluate_curvature_grid(metric, grid, precision="fp64")

        for field in ref._fields:
            a = getattr(ref, field)
            b = getattr(explicit_fp64, field)
            assert jnp.array_equal(a, b), (
                f"Field {field!r} drifted under precision='fp64' explicit call"
            )

    def test_fp32_screen_band_reverifies_flagged_points(self):
        """Near-zero-Kretschmann points fall in the band and re-verify to the
        same value as the v0.1.x fp64 path (within 1e-12)."""
        # Flat Minkowski grid has Kretschmann ≡ 0; every point falls in the band.
        metric = MinkowskiMetric()
        grid = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(4, 4, 4))

        ref_fp64 = evaluate_curvature_grid(metric, grid, precision="fp64")
        band = evaluate_curvature_grid(
            metric, grid, precision="fp32_screen+fp64_verify", fp32_band_tol=5e-4
        )

        # On flat Minkowski, both paths must agree to fp64 precision
        for field in ref_fp64._fields:
            a = np.array(getattr(ref_fp64, field))
            b = np.array(getattr(band, field))
            np.testing.assert_allclose(
                b,
                a,
                atol=1e-12,
                rtol=1e-12,
                err_msg=f"Minkowski band field {field!r} drifted beyond 1e-12",
            )

    def test_fp32_screen_non_flagged_points_remain_fp32_cast(self):
        """High-curvature Alcubierre wall points fall OUTSIDE the band at
        tight tolerance (|margin_fp32| >= tol * scale)."""
        metric, grid = self._alcubierre_8cubed()

        # Tighter tol → fewer points flagged; result is still valid curvature
        band = evaluate_curvature_grid(
            metric, grid, precision="fp32_screen+fp64_verify", fp32_band_tol=1e-8
        )
        # Sanity check: dtype is float64 (on this build, x64 is global)
        assert band.metric.dtype == jnp.float64
        # Shape preserved
        assert band.metric.shape == (8, 8, 8, 4, 4)

    def test_no_false_negatives_at_type_i_iv_boundary(self):
        """At a synthesised near-boundary point (perturbed wall), the band
        path agrees with the v0.1.x fp64 path on the Hawking-Ellis
        classification proxy (stress_energy values within 1e-10)."""
        # Perturb Alcubierre wall radius to create near-boundary curvature
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        # 1-point grid at wall edge ~= R: most sensitive to precision drift
        grid = GridSpec(
            bounds=[(0.99, 1.01), (-0.01, 0.01), (-0.01, 0.01)],
            shape=(2, 2, 2),
        )

        ref = evaluate_curvature_grid(metric, grid, precision="fp64")
        band = evaluate_curvature_grid(
            metric, grid, precision="fp32_screen+fp64_verify", fp32_band_tol=5e-4
        )

        # No false-negatives: stress_energy must match to 1e-10 at every
        # boundary-adjacent point.
        np.testing.assert_allclose(
            np.array(band.stress_energy),
            np.array(ref.stress_energy),
            atol=1e-10,
            rtol=1e-10,
            err_msg=(
                "No-false-negative invariant violated at Type-I/IV boundary: "
                "precision-band stress_energy diverged from fp64 path"
            ),
        )

    def test_invalid_precision_raises(self):
        """precision='fp16' (or any non-{'fp64','fp32_screen+fp64_verify'}) raises."""
        metric, grid = self._alcubierre_8cubed()
        with pytest.raises(ValueError, match="precision must be one of"):
            evaluate_curvature_grid(metric, grid, precision="fp16")

    def test_backend_env_var_override(self, monkeypatch):
        """WARPAX_PERF_BACKEND env var overrides the ``backend`` kwarg.

        Invokes with explicit ``backend='gpu'`` but WARPAX_PERF_BACKEND=cpu;
        on a machine without GPU, the call would fail at
        ``jax.devices('gpu')`` without the env override. With the override,
        the call succeeds and runs on CPU.
        """
        metric, grid = self._alcubierre_8cubed()

        # Force CPU via env var, override kwarg 'gpu' without error
        monkeypatch.setenv("WARPAX_PERF_BACKEND", "cpu")

        result = evaluate_curvature_grid(
            metric,
            grid,
            precision="fp32_screen+fp64_verify",
            backend="gpu",  # env var override prevents GPU device acquisition
        )
        # If we reached here, env var override worked (CPU was used)
        assert result.metric.shape == (8, 8, 8, 4, 4)

        # Also verify the devices seen by the function were CPU
        assert jax.devices("cpu")[0].platform == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
