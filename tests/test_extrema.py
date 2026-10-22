"""Continuous extremum polishing off a grid seed."""

from __future__ import annotations

import numpy as np
import pytest

from warpax.analysis.extrema import refine_extremum
from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions import certify_grid_frame_free
from warpax.geometry import evaluate_curvature_grid
from warpax.grids import wall_clustered


def test_polished_imaginary_peak_beats_the_grid_and_sits_on_the_wall():
    """Polishing raises the sampled max|Im lambda| and lands on the wall.

    The seed grid only has to bracket the peak; the polish is what is under test,
    so it is kept coarse.
    """
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    grid = wall_clustered(metric, [(-3.0, 3.0)] * 3, (32, 32, 32), a=2.0)
    curv = evaluate_curvature_grid(metric, grid, batch_size=4096)
    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)

    imag = np.max(np.abs(np.asarray(ff.eigenvalues_imag)), axis=-1)
    k = int(np.argmax(imag.ravel()))
    axes = [np.asarray(grid.axes[a]) for a in range(3)]
    i0, i1, i2 = np.unravel_index(k, imag.shape)
    seed = [axes[0][i0], axes[1][i1], axes[2][i2]]
    coarse = float(imag.ravel()[k])

    def field_max_imag(c):
        f = certify_grid_frame_free(c.stress_energy, c.metric, c.metric_inv)
        return np.max(np.abs(np.asarray(f.eigenvalues_imag)), axis=-1)

    res = refine_extremum(metric, seed, field_max_imag, mode="max", half_width=0.15, n=9, levels=7)

    assert res["value"] >= coarse - 1e-9, "polished max must not fall below sampled max"
    shape = res["shape_at_extremum"]
    assert shape is not None and 0.02 < shape < 0.995, (
        "Type-IV imaginary-eigenvalue peak must sit on the wall"
    )


def test_metric_failure_inside_refinement_propagates():
    """Only NotImplementedError (no shape function) is tolerated at the final
    host-side shape evaluation; any other failure must surface."""
    import jax

    class BrokenMetric(AlcubierreMetric):
        def shape_function_value(self, coords):
            if isinstance(coords, jax.core.Tracer):
                return super().shape_function_value(coords)
            raise RuntimeError("shape failed")

    metric = BrokenMetric(v_s=0.5, R=1.0, sigma=8.0)
    with pytest.raises(RuntimeError, match="shape failed"):
        refine_extremum(metric, [1.0, 0.0, 0.0], lambda c: np.zeros((3, 3, 3)), n=3, levels=1)
