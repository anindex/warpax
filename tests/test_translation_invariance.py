"""The NEC-violation pattern travels with the bubble, and nothing else moves.

Section 5 of the observer-robust paper reports this as a coordinate-invariance check
beside the statement that no convergence order is assumed. It had no script, no test
and no artifact behind it, which is exactly the provenance gap the rest of the pipeline
exists to close, so it lives here now: the numbers in the manuscript are the numbers
this test asserts.

What is checked is that advancing the Alcubierre bubble centre from ``x_s = 0`` to
``x_s = v_s t = 1``, and carrying the sampling window with it, translates the
deficit-weighted centroid of the NEC violation by exactly the displacement and moves it
transversely not at all. What is deliberately NOT checked is agreement of the minimum
margins: the window includes the bubble centre, where the spherical form of the metric
carries a removable ``1/r_s`` that the two grids resolve differently, and the minima
there differ by orders of magnitude. That is a property of sampling a coordinate
singularity, not of the geometry, and the manuscript no longer claims otherwise.
"""

import numpy as np
import pytest

from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.energy_conditions import verify_grid
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.grids import GridSpec

HALF = 2.5  # window half-width, in units of R_b
N = 41


def _violation_centroid(x_s: float) -> tuple[int, int, float, float]:
    """Sampled count, violating count, and the deficit-weighted centroid ``(x, y)``."""
    spec = GridSpec(
        bounds=((x_s - HALF, x_s + HALF), (-HALF, HALF), (0.0, 0.0)),
        shape=(N, N, 1),
    )
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0, x_s=x_s)
    curv = evaluate_curvature_grid(metric, spec, batch_size=256)
    ec = verify_grid(curv.stress_energy, curv.metric, n_starts=8, batch_size=64)

    nec = np.asarray(ec.nec_margins).reshape(-1)
    coords = np.asarray(build_coord_batch(spec, t=0.0)).reshape(-1, 4)
    violating = nec < 0
    weight = -nec[violating]  # weight by the deficit, not by the count
    return (
        int(nec.size),
        int(violating.sum()),
        float((weight * coords[violating, 1]).sum() / weight.sum()),
        float((weight * coords[violating, 2]).sum() / weight.sum()),
    )


@pytest.mark.slow
def test_violation_centroid_translates_with_the_bubble():
    sampled_0, viol_0, cx_0, cy_0 = _violation_centroid(0.0)
    sampled_1, viol_1, cx_1, cy_1 = _violation_centroid(1.0)

    # The window is the same size and finds the same violating set on both sides.
    assert sampled_0 == sampled_1 == N * N == 1681
    assert viol_0 == viol_1 == 1677

    # The centroid moves by the displacement, and by nothing else.
    assert cx_1 - cx_0 == pytest.approx(1.0, abs=1e-8)
    assert cy_1 - cy_0 == pytest.approx(0.0, abs=1e-8)


if __name__ == "__main__":  # pragma: no cover - manual run
    for x_s in (0.0, 1.0):
        print(x_s, _violation_centroid(x_s))
