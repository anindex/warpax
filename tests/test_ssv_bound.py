"""SSV lower-bound saturation (K12): fit recovery and subluminal filtering."""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load_script(name):
    # scripts import siblings (_json_io); keep this file standalone-runnable
    if _SCRIPTS not in sys.path:
        sys.path.insert(0, _SCRIPTS)
    path = os.path.join(_SCRIPTS, name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


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
        mod = _load_script("run_ssv_bound.py")
        rows = [
            {"metric": "R", "v_s": 0.5, "typeI_nec_min": -0.10, "n_type_i_wall": 10},
            {"metric": "R", "v_s": 1.5, "typeI_nec_min": -0.50, "n_type_i_wall": 10},
            {"metric": "R", "v_s": 0.3, "typeI_nec_min": 0.20, "n_type_i_wall": 10},
            {"metric": "R", "v_s": 0.2, "typeI_nec_min": -0.05, "n_type_i_wall": 0},
        ]
        vs, deficits = mod._subluminal_deficits(rows, "R")
        # Only the v_s=0.5 violating, resolved, subluminal point survives.
        assert list(np.round(vs, 6)) == [0.5]
        assert list(np.round(deficits, 6)) == [0.10]

    def test_cached_rodal_coefficient(self):
        import json

        path = os.path.join(_SCRIPTS, "..", "results", "ssv_bound.json")
        if not os.path.exists(path):
            pytest.skip("results/ssv_bound.json not present")
        fits = json.load(open(path))["fits"]
        assert abs(fits["Rodal"]["C"] - 0.688) < 0.01
        assert fits["Rodal"]["r_squared_fixed"] > 0.999
