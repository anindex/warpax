"""asv benchmark - persistent JIT cache cold-vs-warm wall-clock.

Each timed call runs a child interpreter that imports warpax and evaluates
the curvature chain on an Alcubierre 16^3 grid, so equinox's in-memory
trace cache cannot short-circuit the measurement. ``time_cold`` points the
child at a fresh persistent cache directory; ``time_warm`` reuses a
directory populated by one child run in ``setup``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

_SCRIPT = """
import warpax
from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid

metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
grid = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(16, 16, 16))
evaluate_curvature_grid(metric, grid, t=0.0, compute_invariants=False).stress_energy.block_until_ready()
"""


def _run_child(cache_dir: str) -> None:
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "WARPAX_JIT_CACHE": "1",
        "JAX_COMPILATION_CACHE_DIR": cache_dir,
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    }
    subprocess.run([sys.executable, "-c", _SCRIPT], env=env, check=True)


class JITCacheColdVsWarm:
    """Cold-vs-warm persistent JIT cache wall-clock on Alcubierre 16^3."""

    warmup_time = 0.0
    number = 1
    repeat = (2, 3, 120.0)
    timeout = 600.0

    def setup(self) -> None:
        self._warm_tmp = tempfile.TemporaryDirectory(prefix="warpax-jit-warm-")
        _run_child(self._warm_tmp.name)

    def teardown(self) -> None:
        self._warm_tmp.cleanup()

    def time_cold(self) -> None:
        """Fresh cache directory: the child must compile and populate it."""
        with tempfile.TemporaryDirectory(prefix="warpax-jit-cold-") as cold:
            _run_child(cold)

    def time_warm(self) -> None:
        """Pre-populated cache directory: the child loads compiled executables."""
        _run_child(self._warm_tmp.name)
