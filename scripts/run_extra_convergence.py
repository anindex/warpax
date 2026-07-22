"""Convergence verdicts for the exoticity index and the per-metric ANEC minimum.

run_diagnostic_convergence.py already certifies the per-point wall diagnostics
(Type-IV fraction, max|Im lambda|, min NEC/DEC margins). These two lacked one:

- Exoticity index (per metric): a composite spatial-grid diagnostic (it folds in
  a thresholded Type-IV fraction), refined on the canonical wall-resolved graded
  ladder (N = [80, 100, 120] -> 4.5 / 5.6 / 6.7 cells across the wall, every level
  clearing the four-cell criterion; see _benchmark_grid.py).
- ANEC minimum (per metric): a geodesic line integral, so its resolution knob is
  the symplectic integrator step count (2048, 4096, 8192), not grid N.

Neither is a valid Richardson target (the exoticity index folds a non-smooth
fraction; the ANEC minimum is an extremum over impact parameter), so both carry a
grid/step stability spread, not an assumed order. Matched family R_b = 1,
sigma = 8, v_s = 0.5 for all four drives (Rodal included, matching the published
anec_symplectic / exoticity_ranking tables).

Outputs: results/extra_convergence.json, ../warpax_arxiv/tables/extra_convergence.tex
"""
from __future__ import annotations

import os

from _json_io import dump_json
from _benchmark_grid import benchmark_grid, N_LADDER

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.averaged.anec import anec_rigorous
from warpax.benchmarks import AlcubierreMetric
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.energy_conditions import (
    certify_grid_frame_free,
    type_fractions,
    typeI_min_margins,
)
from warpax.energy_conditions.filtering import shape_function_mask

# Reuse the exoticity index (no re-implementation).
from run_exoticity_ranking import exoticity_index


def _stability(values):
    """Stability verdict for a (non-Richardson) series: exoticity index is a
    composite with a thresholded-fraction component and the ANEC minimum is an
    extremum over impact parameter, so neither is a valid Richardson target, so we
    report the spread across the ladder, not an assumed order."""
    vals = [v for v in values if v is not None and np.isfinite(v)]
    spread = float(max(vals) - min(vals)) if vals else float("nan")
    return {"values": list(values), "spread": spread,
            "stable": bool(spread <= 0.05 * (abs(np.mean(vals)) + 1e-12) or spread <= 0.02),
            "verdict": "stability"}

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
        [(-3.0, 3.0)] * 3, 1.0,
    ),
    # Matched family (R=1, sigma=8), same as run_anec_symplectic / run_velocity_sweep
    # -> run_exoticity_ranking. At native R=100/sigma=0.03 the wall sits at r~100 and
    # the ANEC min is a shallow ~-3e-5 near b~210, not the published -0.0041 @ b~1.90;
    # matching the family reproduces both published tables (ANEC -0.0041, exoticity 0.014).
    "Rodal": (RodalMetric, {"R": 1.0, "sigma": 8.0}, [(-3.0, 3.0)] * 3, 1.0),
}
GRID_N = N_LADDER             # exoticity-index spatial resolutions (wall-resolved ladder)
ANEC_STEPS = [2048, 4096, 8192]  # ANEC integrator resolutions
N_B = 30                       # impact-parameter fan (matches run_anec_symplectic;
                               # Natario's min at b~0.777 is sharp, needs dense b)


def _instantiate(name):
    cls, kw, _, _ = FAMILY[name]
    return cls(v_s=V_S, **kw)


# ---------------------------------------------------------------- ANEC minimum
def _anec_min(metric, R, num_steps):
    """Most-negative geodesic ANEC line integral over an axial impact fan."""
    x_start = -8.0 * R
    span = 16.0 * R
    b_scan = np.linspace(1.0e-3 * R, 2.5 * R, N_B)
    best = np.inf
    worst_wit = 0.0
    for b in b_scan:
        x0 = jnp.array([0.0, x_start, float(b), 0.0], dtype=jnp.float64)
        r = anec_rigorous(metric, x0, jnp.array([1.0, 0.0, 0.0]),
                          affine_bounds=(0.0, span), num_steps=num_steps,
                          order=4, null_tol=1e-6)
        li = float(r.symplectic.line_integral)
        worst_wit = max(worst_wit, float(r.symplectic.max_abs_g_kk))
        best = min(best, li)
    return best, worst_wit


# ------------------------------------------------------------- exoticity axes
def _grid_axes(name, N):
    """Wall NEC severity + wall Type-IV fraction on the N^3 wall-clustered grid."""
    metric = _instantiate(name)
    grid = benchmark_grid(metric, N)
    bs = 4096 if N < 100 else 2048
    curv = evaluate_curvature_grid(metric, grid, batch_size=bs)
    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)
    coords = build_coord_batch(grid, t=0.0)
    mask = shape_function_mask(metric, coords, grid.shape)
    fracs = type_fractions(ff, mask=mask, volume_weights=grid.volume_weights_array)
    margins = typeI_min_margins(ff, mask=mask)
    nec = margins["nec_min"]
    nec_sev = abs(nec) if (nec is not None and np.isfinite(nec) and nec < 0) else 0.0
    return {"nec_severity": nec_sev, "type_iv_frac": float(fracs["frac_type_iv"])}


def _f(x, nd=3):
    return f"{x:.{nd}f}" if (x is not None and np.isfinite(x)) else "--"


def _verdict_str(c):
    if c.get("verdict") == "stability":
        return f"stable ({_f(c.get('spread'), 4)})"
    return "--"


def write_table(results, out_path):
    lines = [
        "% Generated by scripts/run_extra_convergence.py; do not edit.",
        f"% Exoticity columns are wall-resolved grid N={GRID_N[0]}/{GRID_N[1]}/{GRID_N[2]};"
        " ANEC columns are integrator",
        "% steps 2048/4096/8192 (ANEC is a geodesic line integral, not a spatial grid).",
        r"\begin{tabular}{@{}l ccc l ccc l@{}}",
        r"  \toprule",
        r"  & \multicolumn{4}{c}{Exoticity index ($N$)}"
        r" & \multicolumn{4}{c}{ANEC min (steps)} \\",
        r"  \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        f"  Metric & {GRID_N[0]} & {GRID_N[1]} & {GRID_N[2]} & verdict "
        r"& 2048 & 4096 & 8192 & verdict \\",
        r"  \midrule",
    ]
    for name in ORDER:
        ex = results[name]["exoticity"]
        an = results[name]["anec_min"]
        ev, av = ex["values"], an["values"]
        lines.append(
            f"  {name} & {_f(ev[0])} & {_f(ev[1])} & {_f(ev[2])} & {_verdict_str(ex)} & "
            f"{_f(av[0], 4)} & {_f(av[1], 4)} & {_f(av[2], 4)} & {_verdict_str(an)} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main():
    print("=" * 72)
    print("EXTRA CONVERGENCE: exoticity index + ANEC minimum")
    print("=" * 72)

    # ---- ANEC minimum vs integrator steps (also supplies the exoticity ANEC axis)
    print("\nANEC minimum (geodesic) vs integrator steps:")
    anec_series = {}
    anec_min_abs = {}
    for name in ORDER:
        metric = _instantiate(name)
        R = FAMILY[name][3]
        vals, wits = [], []
        for ns in ANEC_STEPS:
            mn, wit = _anec_min(metric, R, ns)
            vals.append(mn)
            wits.append(wit)
        anec_series[name] = vals
        anec_min_abs[name] = abs(vals[-1])  # finest resolution
        print(f"  {name:16s} " + " ".join(f"{v:+.5f}" for v in vals) +
              f"   worst|g(k,k)|={max(wits):.1e}")

    # ---- Exoticity index vs wall grid resolution
    print("\nExoticity index vs wall grid resolution:")
    raw_axes = {N: {} for N in GRID_N}
    for N in GRID_N:
        for name in ORDER:
            ga = _grid_axes(name, N)
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

    # ---- Certify both
    results = {}
    for name in ORDER:
        results[name] = {
            "exoticity": _stability(exo_series[name]),
            "anec_min": _stability(anec_series[name]),
        }
        e, a = results[name]["exoticity"], results[name]["anec_min"]
        print(f"  {name:16s} exoticity: {_verdict_str(e):20s}  ANEC: {_verdict_str(a)}")

    out = {
        "params": {"v_s": V_S, "grid_N": GRID_N, "anec_steps": ANEC_STEPS,
                   "n_b": N_B,
                   "note": "exoticity resolution = spatial grid N (wall-clustered); "
                           "ANEC resolution = symplectic integrator steps (geodesic, "
                           "not a spatial grid). matched family R_b=1 sigma=8 for all "
                           "four drives (Rodal included)."},
        "order": ORDER,
        "anec_min_series": anec_series,
        "exoticity_series": exo_series,
        "results": results,
    }
    dump_json(out, os.path.join(RESULTS_DIR, "extra_convergence.json"))
    print(f"\nWrote {os.path.join(RESULTS_DIR, 'extra_convergence.json')}")
    write_table(results, os.path.join(TABLES_DIR, "extra_convergence.tex"))


if __name__ == "__main__":
    main()
