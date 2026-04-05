"""asv benchmark - auto_chunk_threshold memory-envelope.

Compares full-vmap vs chunked (``auto_chunk_threshold=25_000``) on the
Alcubierre 64³ curvature chain. On CPU, wall-clock overlaps within the
 20% noise budget - the chunked variant is primarily
intended as a memory envelope on large-grid GPU workloads (128³+ where
full-vmap would exhaust VRAM). Reported here as a regression watchpoint
rather than a wall-clock win.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU canonical

import jax
import jax.numpy as jnp

import warpax  # noqa: F401 -- triggers jax_enable_x64
from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid


class AutoChunkMemoryEnvelope:
    """asv benchmark - full vmap vs auto-chunked 25k on Alcubierre 64³."""

    warmup_time = 3.0
    number = 1
    repeat = (2, 3, 60.0)
    timeout = 180.0

    def setup(self) -> None:
        self.metric = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        self.grid = GridSpec(bounds=[(-6.0, 6.0)] * 3, shape=(64, 64, 64))

        def _run_full(metric: AlcubierreMetric) -> jnp.ndarray:
            r = evaluate_curvature_grid(
                metric, self.grid, t=0.0, compute_invariants=False
            )
            return r.stress_energy

        def _run_chunked(metric: AlcubierreMetric) -> jnp.ndarray:
            r = evaluate_curvature_grid(
                metric,
                self.grid,
                t=0.0,
                compute_invariants=False,
                auto_chunk_threshold=25_000,
            )
            return r.stress_energy

        self.fn_full = jax.jit(_run_full)
        self.fn_chunked = jax.jit(_run_chunked)

        # JIT warmup - keep timed methods lean
        _ = self.fn_full(self.metric).block_until_ready()
        _ = self.fn_chunked(self.metric).block_until_ready()

    def time_full_vmap_64cubed(self) -> None:
        self.fn_full(self.metric).block_until_ready()

    def time_chunked_25k_64cubed(self) -> None:
        self.fn_chunked(self.metric).block_until_ready()
