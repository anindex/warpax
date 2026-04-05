"""asv benchmark - persistent JIT cache cold-vs-warm wall-clock.

Compares the wall-clock of a curvature-chain evaluation when the JAX
compilation cache is (a) cold (fresh temp cache dir, no prior artifacts) vs
(b) warm (same temp cache dir pre-populated by one prior evaluation).

The benchmark respects 20% noise budget and uses the CPU
platform (canonical reproduction).

Note
----
This benchmark targets the acceptance claim ("≥ 30% cold→warm
reduction"). Ratios are reported through the asv time harness; analysis
lives in
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU canonical

import tempfile

import jax
import jax.numpy as jnp

import warpax  # noqa: F401 -- triggers jax_enable_x64 + _initialize_jit_cache
from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid


def _run_curvature(metric: AlcubierreMetric, grid: GridSpec) -> jnp.ndarray:
    """Pointwise curvature-chain invocation used by both cold and warm paths."""
    result = evaluate_curvature_grid(metric, grid, t=0.0, compute_invariants=False)
    return result.stress_energy


class JITCacheColdVsWarm:
    """Cold-vs-warm JIT cache wall-clock on Alcubierre 16^3.

    ``time_cold_import_and_eval`` points the JAX persistent cache at a fresh
    ``TemporaryDirectory`` for each repeat (asv isolates repeats). The first
    compile populates the cache; subsequent logical repeats share the dir.

    ``time_warm_import_and_eval`` points at a pre-populated cache dir built
    during ``setup`` so no compile happens during the timed body.
    """

    warmup_time = 1.0
    number = 1
    repeat = (2, 3, 30.0)
    timeout = 120.0

    def setup(self) -> None:
        self.metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        self.grid = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(16, 16, 16))

        # Two distinct temp dirs: cold (fresh per setup) and warm (pre-warmed).
        self._cold_tmp = tempfile.TemporaryDirectory(prefix="warpax-jit-cold-")
        self._warm_tmp = tempfile.TemporaryDirectory(prefix="warpax-jit-warm-")
        self._cold_dir = self._cold_tmp.name
        self._warm_dir = self._warm_tmp.name

        # Pre-populate the warm cache by running one compile+eval there.
        jax.config.update("jax_compilation_cache_dir", self._warm_dir)
        _run_curvature(self.metric, self.grid).block_until_ready()

    def teardown(self) -> None:
        # Clean up temp dirs between benchmark invocations.
        try:
            self._cold_tmp.cleanup()
            self._warm_tmp.cleanup()
        except Exception:
            pass

    def time_cold_import_and_eval(self) -> None:
        """Cold-cache path: fresh temp dir, first compile must fill the cache."""
        jax.config.update("jax_compilation_cache_dir", self._cold_dir)
        _run_curvature(self.metric, self.grid).block_until_ready()

    def time_warm_import_and_eval(self) -> None:
        """Warm-cache path: pre-populated dir, compile is short-circuited."""
        jax.config.update("jax_compilation_cache_dir", self._warm_dir)
        _run_curvature(self.metric, self.grid).block_until_ready()
