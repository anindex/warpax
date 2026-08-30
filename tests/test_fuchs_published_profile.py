"""The Fuchs row must be the *published* Fuchs metric.

Two substitutions had crept into the implementation, either of which breaks
attribution to arXiv:2405.02709:

1. the shift transition used a quintic ``smoothstep`` between ``R_1 + R_b`` and
   ``R_2 - R_b`` instead of the paper's reciprocal-exponential sigmoid;
2. the radial interpolation clamped beyond the solved grid, so the exterior was
   not Schwarzschild, which is precisely what the manuscript's
   Santiago-Schuster-Visser escape argument rests on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from warpax.metrics import fuchs_default
from warpax.metrics.fuchs_construction import _fuchs_shift_transition

R_1, R_2, R_B = 10.0, 20.0, 1.0


def _S(r: float) -> float:
    return float(_fuchs_shift_transition(jnp.asarray(r), R_1, R_2, R_B))


def test_sigmoid_is_symmetric_about_the_shell_midpoint():
    assert _S(0.5 * (R_1 + R_2)) == pytest.approx(0.5, abs=1e-12)


def test_sigmoid_saturates_outside_the_shell():
    assert _S(R_1 - 1.0) == pytest.approx(1.0)
    assert _S(R_2 + 1.0) == pytest.approx(0.0)


def test_published_ten_ninety_crossings():
    """Closed-form crossings of the paper's sigmoid, to six decimals.

    The superseded quintic smoothstep put them elsewhere, so these values are
    what ties the reported row to the published construction.
    """
    r90 = brentq(lambda r: _S(r) - 0.9, R_1 + 1e-3, 15.0)
    r10 = brentq(lambda r: _S(r) - 0.1, 15.0, R_2 - 1e-3)
    assert r90 == pytest.approx(12.790029, abs=1e-5)
    assert r10 == pytest.approx(17.209971, abs=1e-5)
    assert (r10 - r90) == pytest.approx(4.419943, abs=1e-5)
    assert 0.5 * (r10 + r90) == pytest.approx(15.0, abs=1e-9)


@pytest.mark.parametrize("r", [5.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0])
def test_transition_gradient_is_finite_through_the_poles(r):
    """The sigmoid has poles at R_1 and R_2; the clamp must keep AD finite."""
    g = float(jax.grad(lambda x: _fuchs_shift_transition(x, R_1, R_2, R_B))(jnp.asarray(r)))
    assert np.isfinite(g)


def test_exterior_is_schwarzschild():
    """Beyond the solved grid the potentials must be the vacuum solution."""
    m = fuchs_default()
    M = m.total_mass
    r_edge = float(m._r_grid[-1])
    for r in (r_edge + 5.0, 2 * r_edge, 4 * r_edge):
        c = jnp.array([0.0, r, 0.0, 0.0])
        assert float(m.lapse(c)) == pytest.approx(np.sqrt(1 - 2 * M / r), rel=1e-10)
        assert float(m.spatial_metric(c)[0, 0]) == pytest.approx(1.0 / (1 - 2 * M / r), rel=1e-10)


def test_exterior_matches_interior_at_the_grid_edge():
    """No jump where the analytic continuation takes over."""
    m = fuchs_default()
    r_edge = float(m._r_grid[-1])
    inside = jnp.array([0.0, r_edge * (1 - 1e-6), 0.0, 0.0])
    outside = jnp.array([0.0, r_edge * (1 + 1e-6), 0.0, 0.0])
    assert float(m.lapse(inside)) == pytest.approx(float(m.lapse(outside)), rel=1e-6)
    assert float(m.spatial_metric(inside)[0, 0]) == pytest.approx(
        float(m.spatial_metric(outside)[0, 0]), rel=1e-6
    )


def test_default_kernel_is_the_published_boxcar():
    """The paper smooths with MATLAB ``smooth()``; that is the default here."""
    import inspect

    assert inspect.signature(fuchs_default).parameters["kernel_type"].default == "moving_average"
