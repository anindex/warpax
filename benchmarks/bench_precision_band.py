"""asv benchmark - precision-band screening.

Compares full-fp64 evaluation vs fp32-screen+fp64-verify pipeline on
Alcubierre 64^3. Target ≥ 1.5x speedup on CPU per REQ (though on
this build the global float64 flag means the screen pass is fp64; the
benchmark establishes the API regression watchpoint for the follow-up
fp32-intermediates mode).

20% asv noise budget per . CPU-only per and
 default.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU canonical

import jax
import jax.numpy as jnp

import warpax  # noqa: F401 -- triggers jax_enable_x64
from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid


class PrecisionBandScreening:
    """asv benchmark - fp64 vs fp32_screen+fp64_verify on Alcubierre 64^3."""

    warmup_time = 3.0
    number = 1
    repeat = (2, 3, 60.0)
    timeout = 180.0

    def setup(self) -> None:
        self.metric = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        self.grid = GridSpec(bounds=[(-6.0, 6.0)] * 3, shape=(64, 64, 64))

        def _run_fp64(metric: AlcubierreMetric) -> jnp.ndarray:
            r = evaluate_curvature_grid(
                metric,
                self.grid,
                t=0.0,
                compute_invariants=False,
                precision="fp64",
            )
            return r.stress_energy

        def _run_band(metric: AlcubierreMetric) -> jnp.ndarray:
            r = evaluate_curvature_grid(
                metric,
                self.grid,
                t=0.0,
                compute_invariants=False,
                precision="fp32_screen+fp64_verify",
                backend="cpu",
                fp32_band_tol=5e-4,
            )
            return r.stress_energy

        self.fn_fp64 = jax.jit(_run_fp64)
        self.fn_band = jax.jit(_run_band)

        # JIT warmup - keep timed methods lean
        _ = self.fn_fp64(self.metric).block_until_ready()
        _ = self.fn_band(self.metric).block_until_ready()

    def time_full_fp64_64cubed(self) -> None:
        self.fn_fp64(self.metric).block_until_ready()

    def time_fp32_screen_fp64_verify_64cubed(self) -> None:
        self.fn_band(self.metric).block_until_ready()
