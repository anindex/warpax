"""Wall-clustered convergence study.

Runs Alcubierre (and optionally matched-parameter Rodal) at the
(R=1, sigma=8) tuple on *uniform* and *wall-clustered* grids across three
resolutions, recording wall-restricted Type-IV fractions and DEC violation
fractions. The study shows that:

(i)   under wall-clustered refinement the wall-restricted Type-IV fraction
      stabilizes, giving a resolution-stable statistic;
(ii)  on uniform grids the full-domain Type-I fraction drifts toward 100%
      with resolution while the wall-restricted fraction converges,
      confirming the full-domain number is a sampling artifact;
(iii) matched-parameter Rodal remains resolution-unstable even on the
      clustered grid, which is recorded as a negative result.

Outputs
-------
- results/clustered_convergence_alcubierre.json: per-resolution diagnostics.
- tables/clustered_convergence.tex: LaTeX convergence table.
- (optional) results/clustered_convergence_rodal_matched.json:
  matched-parameter Rodal feasibility check on the clustered grid.

Usage
-----
    PYTHONPATH=src python scripts/run_clustered_convergence.py
    PYTHONPATH=src python scripts/run_clustered_convergence.py --include-rodal-matched
"""
from __future__ import annotations

import argparse
import os
import time

from _json_io import dump_json

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from types import SimpleNamespace

from warpax.analysis import compare_eulerian_vs_robust
from warpax.benchmarks import AlcubierreMetric
from warpax.metrics import RodalMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.grids import wall_clustered
from warpax.energy_conditions.filtering import (
    compute_wall_restricted_stats,
    shape_function_mask,
)


F_LOW = 0.1
F_HIGH = 0.9


def run_single(
    name: str,
    metric,
    grid_spec: GridSpec,
    grid_label: str,
    n_starts: int,
    batch_size: int,
) -> dict:
    """Compare Eulerian vs robust EC and compute wall-restricted stats.

    Uses a single :func:`compare_eulerian_vs_robust` pass (one robust
    optimizer sweep) which yields the Eulerian margins, robust margins,
    and Hawking-Ellis types together. The wall-restricted conditional
    miss rates are then volume-weighted on the (non-uniform) clustered
    grid via ``grid_spec.volume_weights_array``, so the reported rates
    are unbiased toward the densely-sampled wall band.
    """
    t0 = time.time()
    curv = evaluate_curvature_grid(metric, grid_spec, batch_size=256)
    t_curv = time.time() - t0

    t0 = time.time()
    comparison = compare_eulerian_vs_robust(
        curv.stress_energy, curv.metric, curv.metric_inv, grid_spec.shape,
        n_starts=n_starts, batch_size=batch_size,
    )
    t_vg = time.time() - t0

    # Lightweight shim exposing exactly the fields compute_wall_restricted_stats
    # reads (he_types + robust margins), avoiding a redundant verify_grid pass.
    ec = SimpleNamespace(
        he_types=comparison.he_types,
        nec_margins=comparison.robust_margins["nec"],
        wec_margins=comparison.robust_margins["wec"],
        sec_margins=comparison.robust_margins["sec"],
        dec_margins=comparison.robust_margins["dec"],
    )
    eulerian_margins = {
        c: comparison.eulerian_margins[c] for c in ("nec", "wec", "sec", "dec")
    }

    coords = build_coord_batch(grid_spec, t=0.0)
    mask = shape_function_mask(metric, coords, grid_spec.shape, f_low=F_LOW, f_high=F_HIGH)
    # Proper per-cell volume weights on clustered grids; None (uniform) recovers
    # raw point-fraction behaviour exactly.
    vol_w = grid_spec.volume_weights_array
    wall = compute_wall_restricted_stats(
        ec, mask, eulerian_margins=eulerian_margins, volume_weights=vol_w,
    )

    def _rate(x):
        return float(x) if x is not None else None

    he_flat = np.asarray(comparison.he_types).ravel()
    n_grid = int(np.prod(grid_spec.shape))
    full_type_i = float(np.mean(he_flat == 1.0))
    full_type_iv = float(np.mean(he_flat == 4.0))

    return {
        "metric": name,
        "grid": grid_label,
        "shape": list(grid_spec.shape),
        "n_grid_total": n_grid,
        "wall_n_total": int(wall.n_total),
        "volume_weighted": bool(vol_w is not None),
        "wall_frac_type_i": float(wall.frac_type_i),
        "wall_frac_type_iv": float(wall.frac_type_iv),
        "wall_nec_miss_rate": _rate(wall.nec_miss_rate),
        "wall_wec_miss_rate": _rate(wall.wec_miss_rate),
        "wall_sec_miss_rate": _rate(wall.sec_miss_rate),
        "wall_dec_miss_rate": _rate(wall.dec_miss_rate),
        "full_frac_type_i": full_type_i,
        "full_frac_type_iv": full_type_iv,
        "n_vacuum": int(np.sum(he_flat == 0.0)),
        "nec_min_margin": float(jnp.min(comparison.robust_margins["nec"])),
        "dec_min_margin": float(jnp.min(comparison.robust_margins["dec"])),
        "t_curvature_s": t_curv,
        "t_verify_grid_s": t_vg,
    }


def alcubierre_panel(args) -> dict:
    """Alcubierre at (R=1, sigma=8) on uniform vs clustered grids."""
    metric_params = {"v_s": 0.5, "R": 1.0, "sigma": 8.0}
    bounds = [(-5, 5)] * 3
    metric = AlcubierreMetric(**metric_params)

    panel = {"metric": "alcubierre", "params": metric_params, "results": []}

    for N in args.resolutions:
        shape = (N, N, N)
        # Uniform grid
        gs_uniform = GridSpec(bounds=bounds, shape=shape)
        print(f"\n--- Alcubierre uniform {N}^3 ---")
        panel["results"].append(
            run_single("alcubierre", metric, gs_uniform, f"uniform_{N}",
                       n_starts=args.n_starts, batch_size=args.batch_size)
        )
        # Wall-clustered grid (a=1.2 empirical, wall radius inferred)
        gs_clustered = wall_clustered(metric, bounds, shape, a=1.2)
        print(f"\n--- Alcubierre wall-clustered {N}^3 (a=1.2) ---")
        panel["results"].append(
            run_single("alcubierre", metric, gs_clustered, f"clustered_{N}",
                       n_starts=args.n_starts, batch_size=args.batch_size)
        )

    return panel


def rodal_matched_panel(args) -> dict:
    """Rodal at the matched (R=1, sigma=8) tuple on the clustered grid.

    This configuration is resolution-unstable on a uniform grid. The
    clustered grid concentrates ~50% of points within a +/-0.5-unit band
    around the wall radius; if matched-parameter Rodal is still unstable
    here, that negative result is recorded rather than suppressed.
    """
    metric_params = {"v_s": 0.5, "R": 1.0, "sigma": 8.0}
    bounds = [(-3, 3)] * 3  # tighter domain for the matched-parameter run
    metric = RodalMetric(**metric_params)

    panel = {"metric": "rodal_matched", "params": metric_params, "results": []}

    for N in args.resolutions:
        shape = (N, N, N)
        gs_clustered = wall_clustered(metric, bounds, shape, a=1.2)
        print(f"\n--- Rodal matched clustered {N}^3 ---")
        try:
            res = run_single("rodal_matched", metric, gs_clustered,
                             f"clustered_{N}", n_starts=args.n_starts,
                             batch_size=args.batch_size)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            res = {
                "metric": "rodal_matched", "grid": f"clustered_{N}",
                "error": str(exc),
            }
        panel["results"].append(res)

    return panel


def write_convergence_table(panel: dict, out_path: str) -> None:
    """Emit a LaTeX convergence table from the Alcubierre clustered panel."""
    rows = [r for r in panel["results"] if r["metric"] == "alcubierre"]
    if not rows:
        return

    clustered = [r for r in rows if r["grid"].startswith("clustered_")]
    uniform   = [r for r in rows if r["grid"].startswith("uniform_")]

    def fmt_pct(x):
        return f"{100*x:.2f}" if x is not None and not (isinstance(x, float) and x != x) else "--"

    lines = [
        r"\begin{tabular}{@{}l c c c c c@{}}",
        r"    \toprule",
        r"    Grid & $N$ & Wall \% Type IV & Full \% Type I & Wall NEC miss & Wall DEC miss \\",
        r"    \midrule",
    ]
    for r in uniform:
        lines.append(
            f"    Uniform   & ${r['shape'][0]}^3$ & "
            f"{fmt_pct(r['wall_frac_type_iv'])} & "
            f"{fmt_pct(r['full_frac_type_i'])} & "
            f"{fmt_pct(r['wall_nec_miss_rate'])} & "
            f"{fmt_pct(r['wall_dec_miss_rate'])} \\\\"
        )
    lines.append(r"    \midrule")
    for r in clustered:
        lines.append(
            f"    Clustered & ${r['shape'][0]}^3$ & "
            f"{fmt_pct(r['wall_frac_type_iv'])} & "
            f"{fmt_pct(r['full_frac_type_i'])} & "
            f"{fmt_pct(r['wall_nec_miss_rate'])} & "
            f"{fmt_pct(r['wall_dec_miss_rate'])} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"\end{tabular}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=[25, 50])  # 100^3 too expensive at default
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--include-rodal-matched", action="store_true",
                        help="Also run matched-parameter Rodal feasibility check.")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    alc_panel = alcubierre_panel(args)
    out_alc = os.path.join(args.results_dir, "clustered_convergence_alcubierre.json")
    dump_json(alc_panel, out_alc)
    print(f"\nWrote {out_alc}")

    write_convergence_table(alc_panel, "../warpax_arxiv/tables/clustered_convergence.tex")

    if args.include_rodal_matched:
        rodal_panel = rodal_matched_panel(args)
        out_rodal = os.path.join(args.results_dir,
                                 "clustered_convergence_rodal_matched.json")
        dump_json(rodal_panel, out_rodal)
        print(f"Wrote {out_rodal}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary: Alcubierre clustered-vs-uniform")
    print("=" * 60)
    for r in alc_panel["results"]:
        print(
            f"  {r['grid']:<14s}  N_wall={r['wall_n_total']:5d}  "
            f"Wall%TypeIV={100*r['wall_frac_type_iv']:5.1f}  "
            f"Full%TypeI={100*r['full_frac_type_i']:5.1f}  "
            f"NaN_vac={r['n_vacuum']}"
        )


if __name__ == "__main__":
    main()
