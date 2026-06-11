"""Single source of truth mapping paper_numbers.tex macros to results/*.json.

Used by ``emit_paper_numbers.py``. Each auto-sourced macro carries a
callable that recomputes its value from the cached analysis outputs; macros with
multi-file / derived provenance are listed as manually maintained with their
source file for review (they are checked for existence, not recomputed).
"""
from __future__ import annotations

import json
import os

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def _load(rel):
    with open(os.path.join(RESULTS, rel)) as f:
        return json.load(f)


def _vort(metric, field="vorticity", v_s=0.5):
    d = _load("shift_vorticity.json")
    for row in d["raw"][metric]:
        if abs(row["v_s"] - v_s) < 1e-9:
            return row[field]
    raise KeyError(f"{metric} v_s={v_s} not in shift_vorticity.json")


def _rodal_nec_scaling():
    d = _load("exoticity_ranking.json")
    return d["scaling_laws"]["Rodal"]["A"]


def _kappa_vorticity():
    """Slope of the controlled pure-rotation f = kappa*omega law (R^2 ~ 1)."""
    return _load("vorticity_type_analytic.json")["controlled_family"]["kappa"]


def _cross(metric, key):
    """Per-metric cross-validation entry from the vorticity-mechanism run."""
    return _load("vorticity_type_analytic.json")["cross_metric"][metric][key]


def _vdb_transition_vs():
    """v_s where the VdB wall Type-I fraction first reaches 0.5 (linear interp)."""
    rows = [r for r in _load("velocity_sweep.json")["rows"]
            if r["metric"] == "Van den Broeck"]
    rows.sort(key=lambda r: r["v_s"])
    prev = None
    for r in rows:
        f = r.get("wall_frac_type_i", 0.0)
        if prev is not None and (prev[1] - 0.5) * (f - 0.5) <= 0.0 and prev[1] != f:
            # First crossing of 0.5 in either direction (VdB falls through it).
            (v0, f0), (v1, f1) = prev, (r["v_s"], f)
            return v0 + (0.5 - f0) * (v1 - v0) / (f1 - f0)
        prev = (r["v_s"], f)
    return float("nan")


# macro name -> (recompute callable | None, rounding, source description)
AUTO_SOURCED = {
    "rodalVortFrac": (lambda: _vort("Rodal"), 3, "shift_vorticity.json"),
    "natarioVortFrac": (lambda: _vort("Natário"), 3, "shift_vorticity.json"),
    "natarioExpFrac": (lambda: _vort("Natário", "expansion"), 3, "shift_vorticity.json"),
    "alcubierreVortFrac": (lambda: _vort("Alcubierre"), 3, "shift_vorticity.json"),
    "vdbVortFrac": (lambda: _vort("Van den Broeck"), 3, "shift_vorticity.json"),
    "rodalNECscaling": (_rodal_nec_scaling, 3, "exoticity_ranking.json (scaling fit)"),
    "vdbTransitionVS": (_vdb_transition_vs, 2, "velocity_sweep.json (50% Type-I crossover)"),
    "kappaVorticity": (_kappa_vorticity, 3,
                       "vorticity_type_analytic.json (controlled pure-rotation slope)"),
    "alcubierreImagRatio": (lambda: _cross("Alcubierre", "imag_ratio"), 1,
                            "vorticity_type_analytic.json (cross-metric Im/(kappa*omega))"),
    "natarioImagRatio": (lambda: _cross("Natário", "imag_ratio"), 1,
                         "vorticity_type_analytic.json (cross-metric Im/(kappa*omega))"),
    "vdbImagRatio": (lambda: _cross("Van den Broeck", "imag_ratio"), 1,
                     "vorticity_type_analytic.json (cross-metric Im/(kappa*omega))"),
    "alcubierreShearVortRatio": (lambda: _cross("Alcubierre", "shear_to_vorticity"), 1,
                                 "vorticity_type_analytic.json (sigma/omega at wall sample)"),
    "natarioShearVortRatio": (lambda: _cross("Natário", "shear_to_vorticity"), 1,
                              "vorticity_type_analytic.json (sigma/omega at wall sample)"),
    "vdbShearVortRatio": (lambda: _cross("Van den Broeck", "shear_to_vorticity"), 1,
                          "vorticity_type_analytic.json (sigma/omega at wall sample)"),
}

# Manually maintained (multi-file/derived provenance): verified by hand against
# the listed source; the audit checks the source file exists but does not
# recompute (the value is a rounded/native-params figure).
MANUAL = {
    "alcubierreNECmissVSfive": "wall_restricted_analysis.json (unconditional)",
    "alcubierreWECmissVSfive": "wall_restricted_analysis.json (unconditional)",
    "rodalNECmissVSfive": "wall_restricted_analysis.json (unconditional)",
    "rodalWECmissVSfive": "wall_restricted_analysis.json (unconditional)",
    "rodalDECmissVSfive": "wall_restricted_analysis.json (unconditional)",
}


def recompute_auto() -> dict[str, str]:
    """Recompute the auto-sourced macros, returned as formatted strings."""
    out = {}
    for name, (fn, nd, _src) in AUTO_SOURCED.items():
        val = float(fn())
        out[name] = f"{val:.{nd}f}"
    return out
