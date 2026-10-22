"""SSV lower-bound saturation: fit recovery and subluminal filtering."""

from __future__ import annotations

import os

import numpy as np
import pytest

from ._helpers import SCRIPTS_DIR as _SCRIPTS
from ._helpers import load_script as _load_script


class TestFitBound:
    def test_recovers_coefficient_and_exponent(self):
        mod = _load_script("run_ssv_bound.py")
        vs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        deficits = 0.688 * vs**2
        fit = mod.fit_bound(vs, deficits)
        assert abs(fit["C"] - 0.688) < 1e-9
        assert abs(fit["q_free"] - 2.0) < 1e-6
        assert fit["r_squared_fixed"] > 0.9999
        assert fit["max_rel_dev"] < 1e-9

    def test_insufficient_points(self):
        mod = _load_script("run_ssv_bound.py")
        fit = mod.fit_bound(np.array([0.5]), np.array([0.2]))
        assert fit["C"] is None and fit["n"] == 1

    def test_subluminal_filter(self):
        """Only the v_s=0.5 violating, resolved, subluminal point survives.
        The Type-I node count travels with the point, because the reported
        deviation is meaningless without it: on a vortical wall the branch is
        a residual that contracts as the speed falls."""
        mod = _load_script("run_ssv_bound.py")
        rows = [
            {"metric": "R", "v_s": 0.5, "typeI_nec_min": -0.10, "n_type_i_wall": 10},
            {"metric": "R", "v_s": 1.5, "typeI_nec_min": -0.50, "n_type_i_wall": 10},
            {"metric": "R", "v_s": 0.3, "typeI_nec_min": 0.20, "n_type_i_wall": 10},
            {"metric": "R", "v_s": 0.2, "typeI_nec_min": -0.05, "n_type_i_wall": 0},
        ]
        vs, deficits, n_type_i = mod._subluminal_deficits(rows, "R")
        assert list(np.round(vs, 6)) == [0.5]
        assert list(np.round(deficits, 6)) == [0.10]
        assert list(n_type_i) == [10]

    def test_cached_rodal_coefficient(self):
        """For the fixed Rodal profile with zero momentum the quadratic
        coefficient is derived from the same sampled grid as the fit; the exact
        law gives the same ratio at every speed to machine precision, and the
        fit must record which sweep it read so a stale pairing is visible in
        the artifact rather than only in a file mtime."""
        import json

        path = os.path.join(_SCRIPTS, "..", "results", "ssv_bound.json")
        if not os.path.exists(path):
            pytest.skip("results/ssv_bound.json not present")
        data = json.load(open(path))
        fits = data["fits"]
        sweep_path = os.path.join(_SCRIPTS, "..", "results", "velocity_sweep.json")
        if not os.path.exists(sweep_path):
            pytest.skip("results/velocity_sweep.json not present")
        sweep = json.load(open(sweep_path))
        ratios = [
            abs(r["typeI_nec_min"]) / r["v_s"] ** 2
            for r in sweep["rows"]
            if r["metric"] == "Rodal" and r["v_s"] < 1.0 and r.get("typeI_nec_min") is not None
        ]
        assert ratios, "no subluminal Rodal rows in the sweep"
        assert max(ratios) - min(ratios) < 1e-12
        assert abs(fits["Rodal"]["C"] - ratios[0]) < 1e-9
        assert fits["Rodal"]["r_squared_fixed"] > 0.999
        assert fits["Rodal"]["max_rel_dev"] < 1e-9
        assert data["source"]["config"]["N"] == sweep["config"]["N"]
