"""Matched-parameter, wall-resolved cross-metric benchmark.

Addresses the central referee request: a common-parameter, wall-resolved
comparison of all *resolved* warp metrics. Every metric is evaluated at the
SAME family parameters (R = 1, sigma = 8) on identical compact bounds, using
wall-clustered grids that place several cells across the bubble wall, and the
reported statistic is the volume-weighted, WALL-RESTRICTED conditional miss
rate (missed violations as a fraction of violated points within the wall,
f in [0.1, 0.9]). This removes the wall-volume-scaling confound of the
unconditional grid-volume fraction and is comparable across metrics.

For each metric and resolution we record:
  - wall-restricted NEC/WEC/SEC/DEC conditional miss rates (volume-weighted),
  - minimum NEC/DEC margins (smooth quantities, for Richardson extrapolation),
  - the number of resolved wall points.

Outputs
-------
- results/matched_benchmark.json                 : structured per-resolution data
- ../warpax_arxiv/tables/missed_wall_restricted.tex : headline table (finest N)
- ../warpax_arxiv/tables/convergence_per_metric.tex : per-metric convergence/stability

Usage
-----
    python scripts/run_matched_benchmark.py
    python scripts/run_matched_benchmark.py --resolutions 30 50 70 --n-starts 4
"""
from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace

from _json_io import dump_json

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpax.analysis import compare_eulerian_vs_robust
from warpax.analysis.convergence import f_miss_stability, richardson_extrapolation
from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions.filtering import (
    compute_wall_restricted_stats,
    shape_function_mask,
)
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.grids import wall_clustered
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "warpax_arxiv", "tables")

V_S = 0.5
BOUNDS = [(-3, 3)] * 3      # identical compact domain for every metric
F_LOW, F_HIGH = 0.1, 0.9    # active warp-wall region

# Matched family parameters (R = 1, sigma = 8) for every retained metric.
# Display name -> (constructor, extra kwargs beyond v_s/R/sigma).
METRICS: dict[str, tuple] = {
    "Alcubierre": (AlcubierreMetric, {}),
    "Natário": (NatarioMetric, {}),
    "Van den Broeck": (
        VanDenBroeckMetric,
        {"R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0},
    ),
    "Rodal": (RodalMetric, {}),
}
METRIC_ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]


def _instantiate(name: str):
    cls, extra = METRICS[name]
    return cls(v_s=V_S, R=1.0, sigma=8.0, **extra)


def run_single(name: str, metric, N: int, n_starts: int, batch_size: int) -> dict:
    """Wall-restricted, volume-weighted EC comparison on a clustered grid."""
    shape = (N, N, N)
    grid_spec = wall_clustered(metric, BOUNDS, shape, a=1.2)

    t0 = time.time()
    curv = evaluate_curvature_grid(metric, grid_spec, batch_size=256)
    comparison = compare_eulerian_vs_robust(
        curv.stress_energy, curv.metric, curv.metric_inv, shape,
        n_starts=n_starts, batch_size=batch_size,
    )
    elapsed = time.time() - t0

    ec = SimpleNamespace(
        he_types=comparison.he_types,
        nec_margins=comparison.robust_margins["nec"],
        wec_margins=comparison.robust_margins["wec"],
        sec_margins=comparison.robust_margins["sec"],
        dec_margins=comparison.robust_margins["dec"],
    )
    eulerian_margins = {c: comparison.eulerian_margins[c] for c in ("nec", "wec", "sec", "dec")}

    coords = build_coord_batch(grid_spec, t=0.0)
    mask = shape_function_mask(metric, coords, shape, f_low=F_LOW, f_high=F_HIGH)
    vol_w = grid_spec.volume_weights_array
    wall = compute_wall_restricted_stats(
        ec, mask, eulerian_margins=eulerian_margins, volume_weights=vol_w,
    )

    def _pct(x):
        return float(x) * 100.0 if x is not None else None

    he_flat = np.asarray(comparison.he_types).ravel()
    return {
        "metric": name,
        "N": N,
        "wall_n": int(wall.n_total),
        "full_type_i_pct": float(np.mean(he_flat == 1.0)) * 100.0,
        "miss": {
            "nec": _pct(wall.nec_miss_rate),
            "wec": _pct(wall.wec_miss_rate),
            "sec": _pct(wall.sec_miss_rate),
            "dec": _pct(wall.dec_miss_rate),
        },
        "nec_min": float(jnp.min(comparison.robust_margins["nec"])),
        "dec_min": float(jnp.min(comparison.robust_margins["dec"])),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------- table writers


def _fmt_pct(x: float | None) -> str:
    return f"{x:.1f}" if x is not None else "--"


def write_headline_table(panels: dict, resolutions: list[int], out_path: str) -> None:
    """Wall-restricted missed-by-Eulerian table at the finest resolution."""
    N = resolutions[-1]
    lines = [
        r"\begin{tabular}{@{}l c c cccc@{}}",
        r"  \toprule",
        r"  & Type~I & Wall & \multicolumn{4}{c}{Missed by Eulerian, wall-restricted (\%)} \\",
        r"  \cmidrule(lr){4-7}",
        r"  Metric & (\%) & pts & NEC & WEC & SEC & DEC \\",
        r"  \midrule",
    ]
    for name in METRIC_ORDER:
        rows = [r for r in panels[name] if r["N"] == N]
        if not rows:
            continue
        r = rows[0]
        m = r["miss"]
        lines.append(
            f"  {name} & {r['full_type_i_pct']:.1f} & {r['wall_n']} & "
            f"{_fmt_pct(m['nec'])} & {_fmt_pct(m['wec'])} & "
            f"{_fmt_pct(m['sec'])} & {_fmt_pct(m['dec'])} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def write_convergence_table(panels: dict, resolutions: list[int], out_path: str) -> None:
    """Per-metric resolution stability (wall DEC miss) + min-margin Richardson."""
    lines = [
        r"\begin{tabular}{@{}l ccc c c c@{}}",
        r"  \toprule",
        r"  & \multicolumn{3}{c}{Wall DEC miss (\%) at $N=$} & Max dev & "
        r"NEC-min & Stable \\",
        r"  \cmidrule(lr){2-4}",
        f"  Metric & {resolutions[0]} & {resolutions[1]} & {resolutions[2]} "
        r"& (pp) & order $p$ & \\",
        r"  \midrule",
    ]
    for name in METRIC_ORDER:
        rows = sorted(
            [r for r in panels[name] if r["N"] in resolutions], key=lambda r: r["N"]
        )
        if len(rows) < 3:
            continue
        dec_vals = [r["miss"]["dec"] for r in rows]
        nec_min = [r["nec_min"] for r in rows]
        if any(v is None for v in dec_vals):
            dec_cells = " & ".join("--" for _ in rows)
            stable_str, dev_str = "--", "--"
        else:
            stab = f_miss_stability(dec_vals, abs_tol_pp=0.5, rel_tol=0.05)
            dec_cells = " & ".join(f"{v:.1f}" for v in dec_vals)
            stable_str = r"\checkmark" if stab["stable"] else r"$\times$"
            dev_str = f"{stab['max_dev_pp']:.2f}"
        rich = richardson_extrapolation(nec_min, resolutions, expected_order=2)
        p_str = f"{rich['observed_order']:.1f}" + (r"$^\dagger$" if rich.get("fallback") else "")
        lines.append(
            f"  {name} & {dec_cells} & {dev_str} & {p_str} & {stable_str} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[30, 50, 70])
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    resolutions = sorted(args.resolutions)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print("Matched-parameter wall-resolved benchmark (R=1, sigma=8, clustered grids)")
    print(f"Resolutions: {resolutions}  |  bounds: {BOUNDS[0]}  |  v_s={V_S}")
    print("=" * 70)

    panels: dict[str, list[dict]] = {name: [] for name in METRIC_ORDER}
    for name in METRIC_ORDER:
        metric = _instantiate(name)
        for N in resolutions:
            print(f"\n--- {name} clustered {N}^3 ---")
            r = run_single(name, metric, N, args.n_starts, args.batch_size)
            m = r["miss"]
            print(
                f"    wall_n={r['wall_n']:5d}  TypeI={r['full_type_i_pct']:5.1f}%  "
                f"miss NEC/WEC/SEC/DEC = "
                f"{_fmt_pct(m['nec'])}/{_fmt_pct(m['wec'])}/"
                f"{_fmt_pct(m['sec'])}/{_fmt_pct(m['dec'])}  ({r['elapsed_s']:.0f}s)"
            )
            panels[name].append(r)

    out_json = os.path.join(RESULTS_DIR, "matched_benchmark.json")
    dump_json(
        {
            "metadata": {
                "v_s": V_S, "R": 1.0, "sigma": 8.0,
                "bounds": [list(b) for b in BOUNDS],
                "resolutions": resolutions, "n_starts": args.n_starts,
                "wall_bounds": [F_LOW, F_HIGH], "grid": "wall_clustered(a=1.2)",
                "statistic": "volume-weighted wall-restricted conditional miss rate",
            },
            "panels": panels,
        },
        out_json,
    )
    print(f"\nWrote {out_json}")

    write_headline_table(panels, resolutions, os.path.join(TABLES_DIR, "missed_wall_restricted.tex"))
    write_convergence_table(panels, resolutions, os.path.join(TABLES_DIR, "convergence_per_metric.tex"))

    print("\n" + "=" * 70)
    print("SUMMARY (wall-restricted DEC miss %, by resolution)")
    print("=" * 70)
    for name in METRIC_ORDER:
        cells = "  ".join(
            f"N={r['N']}:{_fmt_pct(r['miss']['dec'])}" for r in panels[name]
        )
        print(f"  {name:>16s}  {cells}")


if __name__ == "__main__":
    main()
