"""Independent all-observer verification of warp-drive positive-energy claims.

At matched family parameters (R=1, sigma=8) on wall-clustered grids, and using
ONLY the frame-independent eigenstructure of T^a_b, we report, wall-restricted
and volume-weighted, for each metric:

  - Hawking-Ellis Type-I / Type-IV fractions (a Type-IV wall has no rest frame
    and no invariant energy density: the positive-energy question is ill-posed
    there);
  - the invariant peak NEC margin min(rho+p_i) over Type-I points;
  - the *single-frame miss*: fraction of all-observer violations the Eulerian
    frame does not see (Eulerian margin >= 0);
  - the integrated exotic-matter content E_- (invariant Type-I and Eulerian);
  - peak proper-energy-deficit reduction factors vs Alcubierre.

This is the live demonstration that single-frame, single-velocity positive-energy
claims (e.g. Rodal arXiv:2512.18008, verified Eulerian-only at v/c=1) require an
all-observer cross-check. The Eulerian baseline is timelike only at v_s < 1, so
the verification runs subluminally (the regime in which such claims are stated); the
companion velocity sweep extends the invariant quantities through v_s >= 1.

Outputs
-------
- results/invariant_verification.json
- ../warpax_arxiv/tables/invariant_benchmark.tex
"""
from __future__ import annotations

import argparse
import os

from _paper_metrics import instantiate
from _json_io import dump_json, write_table as write_tex_table
from _benchmark_grid import benchmark_grid, wall_cells, N_DEFAULT

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpax.analysis.invariant_verification import (
    integrated_exotic_content,
    peak_proper_energy_deficit,
    reduction_factors,
    single_frame_miss,
)
from warpax.analysis.extrema import refine_extremum
from warpax.energy_conditions.filtering import shape_function_mask
from warpax.energy_conditions.frame_free import certify_grid_frame_free, type_fractions
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.grids import proper_volume_weights

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

F_LOW, F_HIGH = 0.1, 0.9


def _field_nec_typeI(curv):
    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)
    he = np.asarray(ff.he_types)
    return np.where(he == 1.0, np.asarray(ff.nec_margins), np.inf)

ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]




def verify_metric(name, v_s, N):
    shape = (N, N, N)
    metric = instantiate(name, v_s)
    grid = benchmark_grid(metric, N)
    curv = evaluate_curvature_grid(metric, grid, batch_size=4096)
    T, g, gi = curv.stress_energy, curv.metric, curv.metric_inv

    coords = build_coord_batch(grid, t=0.0)
    mask = shape_function_mask(metric, coords, shape, f_low=F_LOW, f_high=F_HIGH)
    mask_flat = np.asarray(jnp.reshape(mask, (-1,))).astype(bool)
    vol_w = proper_volume_weights(grid.volume_weights_array, g)

    ff = certify_grid_frame_free(T, g, gi, solver="standard", lmi_where=mask)
    fr = type_fractions(ff, mask=mask, volume_weights=vol_w)
    miss = single_frame_miss(T, g, gi, mask=mask_flat, volume_weights=np.asarray(jnp.reshape(vol_w, (-1,))))
    exotic = integrated_exotic_content(T, g, gi, vol_w, mask=mask)
    peaks = peak_proper_energy_deficit(T, g, gi, mask=mask_flat)

    # Invariant peak NEC deficit min(rho+p_i) over Type-I wall points, polished to
    # its EXACT continuous value (the grid sample only bounds it from above; the
    # true minimum is resolution-independent, obtained by local refinement of the
    # exact tensor).
    nec_inv = np.asarray(ff.nec_margins).ravel()
    typeI_wall = mask_flat & (np.asarray(ff.he_types).ravel() == 1.0) & np.isfinite(nec_inv)
    nec_min_grid = float(np.min(nec_inv[typeI_wall])) if typeI_wall.any() else float("nan")
    nec_min = nec_min_grid
    nec_coord = None
    if typeI_wall.any():
        k = int(np.argmin(np.where(typeI_wall, nec_inv, np.inf)))
        axes = [np.asarray(grid.axes[a]) for a in range(3)]
        i0, i1, i2 = np.unravel_index(k, shape)
        nec_coord = [float(axes[0][i0]), float(axes[1][i1]), float(axes[2][i2])]
        pol = refine_extremum(metric, nec_coord, _field_nec_typeI, mode="min",
                              half_width=0.15, n=9, levels=7)
        if pol["value"] is not None and np.isfinite(pol["value"]):
            # guarantee the polished value is at least as deep as the grid sample
            nec_min = float(min(nec_min_grid, pol["value"]))

    wc, dxw = wall_cells(metric, N)

    def _pct(x):
        return x * 100.0 if x is not None else None

    return {
        "metric": name, "v_s": v_s, "N": N,
        "wall_cells": wc, "dx_wall": dxw,
        "frac_type_i": fr["frac_type_i"], "frac_type_iv": fr["frac_type_iv"],
        "wall_n": fr["n_selected"],
        "invariant_nec_min": nec_min,
        "invariant_nec_min_grid": nec_min_grid,
        "miss_wec_pct": _pct(miss["wec"]["miss_rate"]),
        "miss_nec_pct": _pct(miss["nec"]["miss_rate"]),
        "miss_dec_pct": _pct(miss["dec"]["miss_rate"]),
        "E_minus_inv": exotic["E_minus_inv"],
        "E_minus_eul": exotic["E_minus_eul"],
        "peak_deficit_inv": peaks["peak_deficit_inv"],
        "peak_deficit_eul": peaks["peak_deficit_eul"],
    }


def write_table(rows, out_path):
    def _f(x, nd=1):
        return f"{x:.{nd}f}" if (x is not None and np.isfinite(x)) else "--"

    lines = [
        r"\begin{tabular}{@{}l cc c ccc@{}}",
        r"  \toprule",
        r"  & Type~I & Type~IV & $\min(\rho+p_i)$ & "
        r"\multicolumn{3}{c}{Missed by Eulerian (\%)} \\",
        r"  \cmidrule(lr){5-7}",
        r"  Metric & (\%) & (\%) & (Type~I) & WEC & NEC & DEC \\",
        r"  \midrule",
    ]
    for name in ORDER:
        r = next((x for x in rows if x["metric"] == name), None)
        if r is None:
            continue
        lines.append(
            f"  {name} & {_f(r['frac_type_i']*100)} & {_f(r['frac_type_iv']*100)} & "
            f"{_f(r['invariant_nec_min'],3)} & "
            f"{_f(r['miss_wec_pct'])} & {_f(r['miss_nec_pct'])} & {_f(r['miss_dec_pct'])} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_tex_table(out_path, lines, script="scripts/run_invariant_verification.py", sources="results/invariant_verification.json")
    print(f"  Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v-s", type=float, default=0.5)
    p.add_argument("--N", type=int, default=N_DEFAULT)
    p.add_argument("--metrics", type=str, nargs="+", default=ORDER)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.N = 24
        args.metrics = ["Alcubierre", "Rodal"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print(f"INVARIANT ALL-OBSERVER VERIFICATION (R=1, sigma=8, v_s={args.v_s}, N={args.N})")
    print("=" * 70)
    rows = []
    for name in args.metrics:
        r = verify_metric(name, args.v_s, args.N)
        rows.append(r)
        print(f"  {name:>15s}  TypeI={r['frac_type_i']*100:5.1f}% TypeIV={r['frac_type_iv']*100:5.1f}%  "
              f"NECmin={r['invariant_nec_min']:.3g}  "
              f"miss W/N/D={r['miss_wec_pct']}/{r['miss_nec_pct']}/{r['miss_dec_pct']}  "
              f"E-_inv={r['E_minus_inv']:.3g} E-_eul={r['E_minus_eul']:.3g}")

    peaks = {r["metric"]: {"peak_deficit_inv": r["peak_deficit_inv"],
                           "peak_deficit_eul": r["peak_deficit_eul"]} for r in rows}
    rfac = reduction_factors(peaks) if "Alcubierre" in peaks else {}

    dump_json({"config": vars(args), "rows": rows, "reduction_factors": rfac},
              os.path.join(RESULTS_DIR, "invariant_verification.json"))
    print(f"\nWrote {os.path.join(RESULTS_DIR, 'invariant_verification.json')}")
    if rfac:
        print("\nPeak-deficit reduction factors vs Alcubierre (invariant / Eulerian):")
        for name in ORDER:
            if name in rfac:
                rf = rfac[name]
                print(f"  {name:>15s}  inv={rf['vs_Alcubierre_inv']}  eul={rf['vs_Alcubierre_eul']}")

    if not args.smoke:
        write_table(rows, os.path.join(TABLES_DIR, "invariant_benchmark.tex"))


if __name__ == "__main__":
    main()
