"""Convergence verdicts for the exoticity index and the per-metric selected finite-ray integral.

- Exoticity index (per metric): a composite spatial-grid diagnostic (it folds in
  a thresholded Type-IV fraction), refined on the canonical wall-resolved graded
  ladder (N = [80, 100, 120] -> 5.9 / 7.6 / 8.9 cells across the wall, every level
  clearing the four-cell criterion; see _benchmark_grid.py).
- selected finite-ray integral (per metric): a geodesic line integral, so its resolution knob is
  the symplectic integrator step density, starting at 16384, 32768, 65536
  and doubling further if needed, with fixed impact parameter and affine endpoints.

The exoticity index includes a thresholded fraction. We report its observed
grid spread without assuming a convergence order. Selected fixed rays retain
their full step ladders and the production finest-pair acceptance criterion. Matched family R_b = 1,
sigma = 8, v_s = 0.5 for all four drives.

Outputs: results/exoticity_anec_convergence.json, ../warpax_arxiv/tables/extra_convergence.tex
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from itertools import pairwise
from pathlib import Path

import jax
from _benchmark_grid import N_LADDER, benchmark_grid
from _json_io import dump_json

jax.config.update("jax_enable_x64", True)
import numpy as np

# Reuse the exoticity index and the impact fan (no re-implementation).
from run_exoticity_ranking import exoticity_index

from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions import (
    certify_grid_frame_free,
    type_fractions,
    typeI_min_margins,
)
from warpax.energy_conditions.filtering import shape_function_mask
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.grids import proper_volume_weights
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric


def _stability(values):
    """Observed grid spread of the composite containing a thresholded fraction."""
    vals = [v for v in values if v is not None and np.isfinite(v)]
    spread = float(max(vals) - min(vals)) if vals else float("nan")
    return {
        "values": list(values),
        "spread": spread,
        "stable": bool(spread <= 0.05 * (abs(np.mean(vals)) + 1e-12) or spread <= 0.02),
        "verdict": "stability",
    }


HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]
V_S = 0.5
# (class, kwargs, bounds, R). Matched family R_b=1, sigma=8 for all four drives.
FAMILY = {
    "Alcubierre": (AlcubierreMetric, {"R": 1.0, "sigma": 8.0}, [(-3.0, 3.0)] * 3, 1.0),
    "Natário": (NatarioMetric, {"R": 1.0, "sigma": 8.0}, [(-3.0, 3.0)] * 3, 1.0),
    "Van den Broeck": (
        VanDenBroeckMetric,
        {"R": 1.0, "sigma": 8.0, "R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0},
        [(-3.0, 3.0)] * 3,
        1.0,
    ),
    # Matched family (R=1, sigma=8), same as run_anec_symplectic / run_velocity_sweep
    # -> run_exoticity_ranking. At native R=100/sigma=0.03 the wall sits at r~100 and
    # the ANEC min is a shallow ~-3e-5 near b~210, not the published -0.0041 @ b~1.90;
    # matching the family reproduces both published tables (ANEC -0.0041, exoticity 0.014).
    "Rodal": (RodalMetric, {"R": 1.0, "sigma": 8.0}, [(-3.0, 3.0)] * 3, 1.0),
}
GRID_N = N_LADDER  # exoticity-index spatial resolutions (wall-resolved ladder)
# Brackets the production setting (run_anec_symplectic.py uses 32768 at SPAN0),
# so the ladder checks the resolution the paper actually reports. Now that
# num_save follows num_steps, it refines the quadrature as well as the path.
ANEC_STEPS = [16384, 32768, 65536]
# The impact-parameter fan IS run_anec_symplectic's, imported rather than
# restated: the two scripts report the same selected finite-ray integral, and when this one kept
# 30 points on [1e-3, 2.5] while that one moved to 50 on [1e-3, 5.0] the same
# quantity came out -0.1416 here and -0.1461 there.


def grid_axes_provenance():
    root = Path(HERE).resolve().parent
    sources = [
        *sorted((root / "src/warpax").rglob("*.py")),
        Path(__file__).resolve(),
        root / "scripts/run_exoticity_ranking.py",
        root / "scripts/_benchmark_grid.py",
        root / "scripts/_json_io.py",
    ]
    return {
        "sources_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "configuration": {
            "grid_N": GRID_N,
            "v_s": V_S,
            "family": {n: repr(_instantiate(n)) for n in ORDER},
            "jax_enable_x64": True,
            "jax_backend": jax.default_backend(),
        },
        "packages": {
            n: importlib.metadata.version(n) for n in ("jax", "jaxlib", "numpy", "equinox")
        },
    }


def load_grid_axes(path):
    data = json.loads(Path(path).read_text())
    if data.get("provenance") != grid_axes_provenance():
        raise ValueError("Stale spatial-axis checkpoint: source or configuration differs")
    if not data.get("complete"):
        raise ValueError("Spatial-axis checkpoint is incomplete")
    expected = {f"{N}:{name}" for N in GRID_N for name in ORDER}
    if set(data["axes"]) != expected:
        raise ValueError("Spatial-axis checkpoint has missing or unexpected rows")
    return data["axes"]


def _instantiate(name):
    cls, kw, _, _ = FAMILY[name]
    return cls(v_s=V_S, **kw)


# ------------------------------------------------------------- exoticity axes
def _grid_axes(name, N):
    """Wall NEC severity + wall Type-IV fraction on the N^3 wall-clustered grid."""
    metric = _instantiate(name)
    grid = benchmark_grid(metric, N)
    bs = 4096 if N < 100 else 2048
    curv = evaluate_curvature_grid(metric, grid, batch_size=bs)
    coords = build_coord_batch(grid, t=0.0)
    mask = shape_function_mask(metric, coords, grid.shape)
    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv, lmi_where=mask)
    vol_w = proper_volume_weights(grid.volume_weights_array, curv.metric)
    fracs = type_fractions(ff, mask=mask, volume_weights=vol_w)
    margins = typeI_min_margins(ff, mask=mask)
    nec = margins["nec_min"]
    nec_sev = abs(nec) if (nec is not None and np.isfinite(nec) and nec < 0) else 0.0
    return {"nec_severity": nec_sev, "type_iv_frac": float(fracs["frac_type_iv"])}


def _f(x, nd=3):
    return f"{x:.{nd}f}" if (x is not None and np.isfinite(x)) else "--"


def _verdict_str(c):
    word = "stable" if c.get("stable", False) else r"\emph{unstable}"
    if c.get("verdict") == "step_stability":
        return f"{word} ($\\delta={c['finest_change']:.1e}$)"
    if c.get("verdict") == "stability":
        return f"{word} ({_f(c.get('spread'), 4)})"
    return "--"


def selected_series(ray, name):
    records = ray["records"]
    steps = [r["steps_per_reference_span"] for r in records]
    assert steps[:3] == ANEC_STEPS, f"{name}: wrong initial step ladder"
    assert all(b == 2 * a for a, b in pairwise(steps)), name
    values = [r["line_integral"] for r in records]
    assert all(v is not None and np.isfinite(v) for v in values), name
    change = abs(values[-1] - values[-2])
    assert ray["step_stable"] and change <= 1e-8 + 1e-4 * abs(values[-1]), (
        f"{name}: selected ray not step-stable"
    )
    return {
        "values": values,
        "steps": steps,
        "num_steps": [r["num_steps"] for r in records],
        "spread": max(values) - min(values),
        "finest_change": change,
        "stable": True,
        "verdict": "step_stability",
    }


def write_table(results, out_path):
    lines = [
        "% Generated by scripts/run_exoticity_anec_convergence.py; do not edit.",
        "% Every selected fixed-ray level is shown as steps per reference span: integral.",
        "% The step verdict uses the finest pair: delta <= 1e-8 + 1e-4*abs(I_finest).",
        r"\begin{tabular}{@{}l ccc l l l@{}}",
        r"  \toprule",
        r"  & \multicolumn{4}{c}{Exoticity index ($N$)}"
        r" & \multicolumn{2}{c}{Selected finite ray} \\",
        r"  \cmidrule(lr){2-5}\cmidrule(lr){6-7}",
        f"  Metric & {GRID_N[0]} & {GRID_N[1]} & {GRID_N[2]} & verdict & "
        + r"step density: integral & verdict \\",
        r"  \midrule",
    ]
    for name in ORDER:
        ex = results[name]["exoticity"]
        an = results[name]["anec_min"]
        ev = ex["values"]
        ladder = (
            r"\shortstack[l]{"
            + r"\\".join(
                f"{n}: ${_f(v, 6)}$" for n, v in zip(an["steps"], an["values"], strict=True)
            )
            + "}"
        )
        lines.append(
            f"  {name} & {_f(ev[0])} & {_f(ev[1])} & {_f(ev[2])} & {_verdict_str(ex)} & "
            f"{ladder} & {_verdict_str(an)} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-axes-checkpoint", type=Path)
    args = parser.parse_args()
    cached_axes = load_grid_axes(args.grid_axes_checkpoint) if args.grid_axes_checkpoint else None
    print("=" * 72)
    print("EXTRA CONVERGENCE: exoticity index + selected finite-ray integral")
    print("=" * 72)

    # Reuse the convergence of the actual selected ray, with fixed b and span.
    with open(os.path.join(RESULTS_DIR, "anec", "retained_symplectic.json")) as fh:
        selected = json.load(fh)["metrics"]
    anec_series, anec_spans, anec_min_abs, anec_summaries = {}, {}, {}, {}
    for name in ORDER:
        ray = selected[name]["selected_ray_convergence"]["minimum_found"]
        anec_summaries[name] = selected_series(ray, name)
        anec_series[name] = anec_summaries[name]["values"]
        anec_spans[name] = {"affine_span": ray["span"], "b": ray["b"]}
        anec_min_abs[name] = abs(anec_series[name][-1])

    # ---- Exoticity index vs wall grid resolution
    print("\nExoticity index vs wall grid resolution:")
    raw_axes = {N: {} for N in GRID_N}
    for N in GRID_N:
        for name in ORDER:
            ga = (
                dict(cached_axes[f"{N}:{name}"]) if cached_axes is not None else _grid_axes(name, N)
            )
            ga["anec_min_abs"] = anec_min_abs[name]
            raw_axes[N][name] = ga
    exo_series = {name: [] for name in ORDER}
    for N in GRID_N:
        base = raw_axes[N]["Alcubierre"]
        for name in ORDER:
            idx = exoticity_index(raw_axes[N][name], base)["index"]
            exo_series[name].append(float(idx))
    for name in ORDER:
        print(f"  {name:16s} " + " ".join(f"{v:.4f}" for v in exo_series[name]))

    # ---- Report observed spreads
    results = {}
    for name in ORDER:
        results[name] = {
            "exoticity": _stability(exo_series[name]),
            "anec_min": anec_summaries[name],
        }
        e, a = results[name]["exoticity"], results[name]["anec_min"]
        print(f"  {name:16s} exoticity: {_verdict_str(e):20s}  ANEC: {_verdict_str(a)}")

    out = {
        "grid_axes_provenance": grid_axes_provenance(),
        "params": {
            "v_s": V_S,
            "grid_N": GRID_N,
            "anec_steps": {name: summary["steps"] for name, summary in anec_summaries.items()},
            "n_b": 1,
            "ray_selection": "minimum found in the production basin, held fixed across steps",
            "note": "exoticity resolution = spatial grid N (wall-clustered); "
            "ANEC resolution = symplectic integrator steps (geodesic, "
            "not a spatial grid). matched family R_b=1 sigma=8 for all "
            "four drives (Rodal included).",
        },
        "order": ORDER,
        "anec_affine_window": anec_spans,
        "anec_min_series": anec_series,
        "exoticity_series": exo_series,
        "results": results,
    }
    dump_json(out, os.path.join(RESULTS_DIR, "exoticity_anec_convergence.json"))
    print(f"\nWrote {os.path.join(RESULTS_DIR, 'exoticity_anec_convergence.json')}")
    write_table(results, os.path.join(TABLES_DIR, "extra_convergence.tex"))


if __name__ == "__main__":
    main()
