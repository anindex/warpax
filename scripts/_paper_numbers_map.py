"""Single source of truth mapping paper_numbers.tex macros to results/*.json.

Used by ``emit_paper_numbers.py``. Each auto-sourced macro carries a
callable that recomputes its value from the cached analysis outputs; macros with
multi-file / derived provenance are listed as manually maintained with their
source file for review (they are checked for existence, not recomputed).

AUTO_SOURCED maps macro name to (recompute callable, rounding, source
description). MANUAL macros have multi-file or derived provenance and are
verified by hand against the listed source; the check verifies the source file
exists but does not recompute, since the value is a rounded or native-parameter
figure.
"""

from __future__ import annotations

import os

from _json_io import load_json
from _paths import RESULTS_DIR


def _vort(metric, field="vorticity", v_s=0.5):
    d = load_json(os.path.join(RESULTS_DIR, "shift_vorticity.json"))
    for row in d["raw"][metric]:
        if abs(row["v_s"] - v_s) < 1e-9:
            return row[field]
    raise KeyError(f"{metric} v_s={v_s} not in shift_vorticity.json")


def _rodal_nec_scaling():
    d = load_json(os.path.join(RESULTS_DIR, "exoticity_ranking.json"))
    return d["scaling_laws"]["Rodal"]["A"]


def _kappa_vorticity():
    """Slope of the controlled pure-rotation f = kappa*omega law (R^2 ~ 1)."""
    return load_json(os.path.join(RESULTS_DIR, "vorticity_type_analytic.json"))[
        "controlled_family"
    ]["kappa"]


def _vdb_transition_vs():
    """v_s of the first 0.5 crossing, in either direction, of the VdB wall Type-I fraction.

    Linear interpolation between sweep points; VdB falls through 0.5.
    """
    rows = [
        r
        for r in load_json(os.path.join(RESULTS_DIR, "velocity_sweep.json"))["rows"]
        if r["metric"] == "Van den Broeck"
    ]
    rows.sort(key=lambda r: r["v_s"])
    prev = None
    for r in rows:
        f = r.get("wall_frac_type_i", 0.0)
        if prev is not None and (prev[1] - 0.5) * (f - 0.5) <= 0.0 and prev[1] != f:
            (v0, f0), (v1, f1) = prev, (r["v_s"], f)
            return v0 + (0.5 - f0) * (v1 - v0) / (f1 - f0)
        prev = (r["v_s"], f)
    return float("nan")


AUTO_SOURCED = {
    "rodalVortFrac": (lambda: _vort("Rodal"), 3, "shift_vorticity.json"),
    "natarioVortFrac": (lambda: _vort("Natário"), 3, "shift_vorticity.json"),
    "natarioExpFrac": (lambda: _vort("Natário", "expansion"), 3, "shift_vorticity.json"),
    "alcubierreVortFrac": (lambda: _vort("Alcubierre"), 3, "shift_vorticity.json"),
    "vdbVortFrac": (lambda: _vort("Van den Broeck"), 3, "shift_vorticity.json"),
    "rodalNECscaling": (_rodal_nec_scaling, 3, "exoticity_ranking.json (scaling fit)"),
    "vdbTransitionVS": (_vdb_transition_vs, 2, "velocity_sweep.json (50% Type-I crossover)"),
    "kappaVorticity": (
        _kappa_vorticity,
        3,
        "vorticity_type_analytic.json (controlled pure-rotation slope)",
    ),
}

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
