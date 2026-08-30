"""Verdicts certified from the metric, and the closing-speed field."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from warpax.energy_conditions.enclosure import (
    alcubierre_metric,
    certify_nec_deficit,
    rodal_metric,
    shape_interval,
)
from warpax.energy_conditions.interval_lmi import certify_point_from_metric

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


class TestCertifyFromMetric:
    def test_wall_point_certified_violated(self):
        m = alcubierre_metric(0.5, 1.0, 8.0)
        r = certify_point_from_metric(m, (0.0, 1.0, 0.35, 0.0))
        assert r["nec"] == "violated"
        assert r["wec"] == "violated"
        assert r["nec_lower"] <= r["nec_upper"] < 0.0

    def test_bracket_encloses_the_pointwise_deficit(self):
        # The branch-and-bound run over a degenerate box must agree with the
        # pointwise certification: same objective, same arithmetic.
        m = alcubierre_metric(0.5, 1.0, 8.0)
        pt = (0.0, 1.0, 0.35, 0.0)
        r = certify_point_from_metric(m, pt)
        enc = certify_nec_deficit(
            m,
            shape_interval(1.0, 8.0),
            x_range=(pt[1], pt[1]),
            s_range=(pt[2], pt[2]),
            tol=1e-6,
            max_boxes=8,
        )
        assert r["nec_lower"] <= enc.upper + 1e-9
        assert enc.lower <= r["nec_upper"] + 1e-9

    def test_vacuum_exterior_is_refused_not_guessed(self):
        # q == 0 out there, so nothing is strictly certifiable either way.
        m = alcubierre_metric(0.5, 1.0, 8.0)
        r = certify_point_from_metric(m, (0.0, 12.0, 0.0, 0.0))
        assert r["nec"] == "inconclusive"
        assert abs(r["nec_lower"]) < 1e-12
        assert abs(r["nec_upper"]) < 1e-12

    def test_irrotational_wall_gives_a_well_formed_bracket(self):
        # Rodal's wall is not uniformly NEC-violating, so the verdict at any one
        # point is not fixed; what must hold is that the bracket is ordered and
        # the label is one the caller can act on.
        m = rodal_metric(0.5, 1.0, 8.0)
        r = certify_point_from_metric(m, (0.0, 1.0, 0.35, 0.0))
        assert r["nec"] in ("satisfied", "violated", "inconclusive")
        assert r["nec_lower"] <= r["nec_upper"]


class TestClosingSpeed:
    def test_v_star_is_zero_without_momentum(self):
        from run_closing_speed import closing_speeds

        a = np.array([1.0, 2.0, 0.5])
        d = np.zeros(3)
        assert np.allclose(closing_speeds(a, d), 0.0)

    def test_v_star_matches_the_closed_form(self):
        from run_closing_speed import closing_speeds

        a = np.array([2.0, -4.0])
        d = np.array([1.0, 1.0])
        assert np.allclose(closing_speeds(a, d), [1.0, 0.5])

    def test_delta_sign_flips_at_v_star(self):
        # Delta = v^2 (a^2 v^2 - 4 d^2) < 0 exactly below v_* = 2d/|a|.
        from run_closing_speed import closing_speeds

        a, d = np.array([3.0]), np.array([2.0])
        v_star = float(closing_speeds(a, d)[0])
        for v, want_negative in ((0.9 * v_star, True), (1.1 * v_star, False)):
            delta = v**2 * (a[0] ** 2 * v**2 - 4.0 * d[0] ** 2)
            assert bool(delta < 0.0) is want_negative

    def test_never_closing_when_a_vanishes(self):
        from run_closing_speed import closing_speeds

        a = np.array([1.0, 0.0])
        d = np.array([1.0, 1.0])
        out = closing_speeds(a, d)
        assert np.isfinite(out[0]) and not np.isfinite(out[1])


@pytest.mark.slow
def test_wall_certified_violated_exterior_refused():
    """Wall: certified violated. Exterior: refused, not guessed.

    Moved here from a module __main__ block that only ran via this test.
    """
    from warpax.energy_conditions.enclosure import van_den_broeck_metric

    m = alcubierre_metric(0.5, 1.0, 8.0)
    wall = certify_point_from_metric(m, (0.0, 1.0, 0.35, 0.0))
    for cond in ("nec", "wec", "sec", "dec"):
        assert wall[cond] == "violated", (cond, wall)
        assert wall[f"{cond}_lower"] <= wall[f"{cond}_upper"] < 0.0, (cond, wall)

    # The ball contains the sphere and sigma >= 0 is a restriction of sigma free,
    # so the weak bracket can only sit at or below the null one.
    pad = 1e-9 * max(1.0, abs(wall["nec_lower"]))
    assert wall["wec_lower"] <= wall["nec_lower"] + pad, wall
    assert wall["wec_upper"] <= wall["nec_upper"] + pad, wall

    # Van den Broeck on the axis satisfies the null, weak and dominant
    # conditions and violates only the strong one, which is invisible to a null
    # witness because Theta(k,k) = T(k,k) there. If the SEC witness ever reverts
    # to the sphere this reads "inconclusive".
    vdb = van_den_broeck_metric(0.5, 1.0, 8.0, R_tilde=1.0, alpha_vdb=0.5, sigma_B=8.0)
    axis = certify_point_from_metric(vdb, (0.0, 1.0, 0.0, 0.0))
    assert axis["sec"] == "violated", axis
    assert axis["sec_upper"] < 0.0, axis
    assert axis["nec"] == "satisfied", axis

    # The exterior is vacuum to 1e-24, so q == 0 there and nothing can be
    # certified strictly positive. "inconclusive" is correct, and the bracket
    # must show it is a saturation refusal rather than a failure. Alcubierre
    # only: a construction with a 1/r tail is genuinely decidable here.
    far = certify_point_from_metric(m, (0.0, 12.0, 0.0, 0.0))
    assert far["nec"] == "inconclusive", far
    assert abs(far["nec_lower"]) < 1e-12 and abs(far["nec_upper"]) < 1e-12, far
