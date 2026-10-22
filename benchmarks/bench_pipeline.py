"""asv benchmarks - end-to-end pipeline hot paths.

Covers the public ``certify`` entry point, the frame-free grid certifier,
the observer-search grid verifier, wall-clustered grid construction, and
a retrace guard for ``evaluate_curvature_grid`` called with a new ``t``
on every invocation.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

import warpax
from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions import certify_grid_frame_free, verify_grid
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.grids import wall_clustered

_BOUNDS = [(-3.0, 3.0)] * 3


def _fields(v_s: float, n: int):
    metric = AlcubierreMetric(v_s=v_s, R=1.0, sigma=8.0)
    grid = GridSpec(bounds=_BOUNDS, shape=(n, n, n))
    curv = evaluate_curvature_grid(metric, grid, t=0.0, compute_invariants=False)
    return curv.stress_energy, curv.metric, curv.metric_inv


class Certify16:
    """asv benchmark - ``warpax.certify`` on Alcubierre 16^3, v_s=1.5."""

    warmup_time = 1.0
    number = 1
    repeat = (2, 3, 120.0)
    timeout = 600.0

    def setup(self) -> None:
        self.metric = AlcubierreMetric(v_s=1.5, R=1.0, sigma=8.0)
        warpax.certify(self.metric, shape=(16, 16, 16))

    def time_certify(self) -> None:
        warpax.certify(self.metric, shape=(16, 16, 16))


class FrameFreeGrid16:
    """asv benchmark - ``certify_grid_frame_free`` on 16^3 fields at v_s=1.5."""

    warmup_time = 1.0
    number = 1
    repeat = (2, 3, 120.0)
    timeout = 600.0

    def setup(self) -> None:
        self.T, self.g, self.gi = _fields(1.5, 16)
        self.time_frame_free()

    def time_frame_free(self) -> None:
        certify_grid_frame_free(self.T, self.g, self.gi).nec_margins.block_until_ready()


class VerifyGrid8:
    """asv benchmark - ``verify_grid`` with defaults on 8^3 fields."""

    warmup_time = 1.0
    number = 1
    repeat = (2, 3, 120.0)
    timeout = 600.0

    def setup(self) -> None:
        self.T, self.g, self.gi = _fields(0.5, 8)
        self.key = jax.random.PRNGKey(0)
        self.time_verify_grid()

    def time_verify_grid(self) -> None:
        verify_grid(self.T, self.g, self.gi, key=self.key).nec_margins.block_until_ready()


class WallClustered64:
    """asv benchmark - ``wall_clustered`` 64^3 grid plus volume weights."""

    warmup_time = 0.5
    number = 1
    repeat = (2, 3, 60.0)
    timeout = 300.0

    def setup(self) -> None:
        self.metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        self.time_wall_clustered()

    def time_wall_clustered(self) -> None:
        wall_clustered(self.metric, _BOUNDS, (64, 64, 64)).volume_weights_array.block_until_ready()


class CurvatureRetrace8:
    """asv benchmark - ``evaluate_curvature_grid`` with a fresh ``t`` per call.

    Catches regressions that retrace or recompile when only the slice time
    changes.
    """

    warmup_time = 0.0
    number = 1
    repeat = (3, 5, 60.0)
    timeout = 300.0

    def setup(self) -> None:
        self.metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        self.grid = GridSpec(bounds=_BOUNDS, shape=(8, 8, 8))
        self.i = 0
        self.time_retrace()

    def time_retrace(self) -> None:
        self.i += 1
        evaluate_curvature_grid(
            self.metric, self.grid, t=0.01 * self.i, compute_invariants=False
        ).stress_energy.block_until_ready()
