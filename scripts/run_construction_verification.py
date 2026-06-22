"""Cross-construction all-observer verification of positive-energy warp drives.

Extends the n=1 Rodal verification to a panel of published positive-energy
constructions -- Fuchs constant-velocity shell (arXiv:2405.02709) and the
Bobrick-Martire / Fell-Heisenberg WarpShell -- alongside the Alcubierre baseline
and the Rodal global-Type-I drive. The source-first S-/T-shells remain available
in the construction registry as a toolkit, but are introduced and audited in the
companion note (arXiv:2605.25417), not certified here, to keep the contributions
disjoint.

Each construction flows through the SAME frame-independent eigenstructure certifier
and all-observer verification, wall-restricted and volume-weighted, with a
resolution gate that withholds numbers for any wall spanning fewer than
``MIN_WALL_CELLS`` cells. The output adds a certified-vs-claimed agreement note:
the all-observer reality (Type-IV wall fraction, NEC severity, single-frame
miss) set against each construction's published claim. This is a diagnostic
cross-check, not a refutation of any author's algebra.

Outputs
-------
- results/construction_verification.json
- ../warpax_arxiv/tables/construction_verification.tex
"""
from __future__ import annotations

import argparse
import os

from _json_io import dump_json

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpax.analysis.construction_adapter import construction_registry, is_resolved
from warpax.analysis.invariant_verification import (
    integrated_exotic_content,
    peak_proper_energy_deficit,
    single_frame_miss,
)
from warpax.energy_conditions.filtering import shape_function_mask
from warpax.energy_conditions.frame_free import certify_grid_frame_free, type_fractions
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.grids import wall_clustered

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

F_LOW, F_HIGH = 0.1, 0.9
ORDER = ["Alcubierre", "Rodal", "Fuchs", "WarpShell", "Garattini"]


def verify_one(spec, speed, n):
    resolved, cells = is_resolved(spec, speed=speed, n=n)
    row = {
        "metric": spec.name,
        "speed_param": spec.speed_param,
        "speed": speed,
        "N": n,
        "wall_cells": cells,
        "resolved": bool(resolved),
        "is_comoving": spec.is_comoving,
        "claim": spec.claim,
    }
    if not resolved:
        row["note"] = (
            f"wall spans {cells:.1f} cells (< 4); numbers withheld -- increase N"
        )
        return row

    metric = spec.metric(speed)
    grid = wall_clustered(metric, list(spec.bounds), (n, n, n), a=1.2)
    curv = evaluate_curvature_grid(metric, grid, batch_size=256)
    T, g, gi = curv.stress_energy, curv.metric, curv.metric_inv

    coords = build_coord_batch(grid, t=0.0)
    mask = shape_function_mask(metric, coords, (n, n, n), f_low=F_LOW, f_high=F_HIGH)
    mask_flat = np.asarray(jnp.reshape(mask, (-1,))).astype(bool)
    vol_w = grid.volume_weights_array
    vol_flat = np.asarray(jnp.reshape(vol_w, (-1,)))

    ff = certify_grid_frame_free(T, g, gi, solver="auto")
    fr = type_fractions(ff, mask=mask, volume_weights=vol_w)
    nec_inv = np.asarray(ff.nec_margins).ravel()
    typeI_wall = mask_flat & (np.asarray(ff.he_types).ravel() == 1.0) & np.isfinite(nec_inv)
    nec_min = float(np.min(nec_inv[typeI_wall])) if typeI_wall.any() else float("nan")

    exotic = integrated_exotic_content(T, g, gi, vol_w, mask=mask)
    peaks = peak_proper_energy_deficit(T, g, gi, mask=mask_flat)

    eulerian_valid = float(speed) < 1.0
    miss = None
    if eulerian_valid:
        miss = single_frame_miss(T, g, gi, mask=mask_flat, volume_weights=vol_flat)

    def _pct(x):
        return x * 100.0 if x is not None else None

    row.update({
        "wall_n": fr["n_selected"],
        "frac_type_i": fr["frac_type_i"],
        "frac_type_iv": fr["frac_type_iv"],
        "invariant_nec_min": nec_min,
        "E_minus_inv": exotic["E_minus_inv"],
        "peak_deficit_inv": peaks["peak_deficit_inv"],
        "eulerian_valid": eulerian_valid,
        "miss_wec_pct": _pct(miss["wec"]["miss_rate"]) if miss else None,
        "miss_nec_pct": _pct(miss["nec"]["miss_rate"]) if miss else None,
        "miss_dec_pct": _pct(miss["dec"]["miss_rate"]) if miss else None,
    })
    return row


def write_table(rows, out_path):
    def _f(x, nd=1):
        return f"{x:.{nd}f}" if (x is not None and np.isfinite(x)) else "--"

    def _fnec(x):
        # NEC margin: keep sign and magnitude for tiny values (avoid "-0.000").
        if x is None or not np.isfinite(x):
            return "--"
        if abs(x) < 5e-4:
            return f"{x:.1e}"
        return f"{x:.3f}"

    lines = [
        r"\begin{tabular}{@{}l c cc c cc@{}}",
        r"  \toprule",
        r"  & Wall & Type~I & Type~IV & $\min(\rho+p_i)$ & WEC & NEC \\",
        r"  Metric & cells & (\%) & (\%) & (Type~I) & \multicolumn{2}{c}{miss (\%)} \\",
        r"  \midrule",
    ]
    for name in ORDER:
        r = next((x for x in rows if x["metric"] == name), None)
        if r is None:
            continue
        if not r.get("resolved", False):
            lines.append(
                f"  {name} & {_f(r['wall_cells'])} & "
                r"\multicolumn{5}{c}{\emph{wall unresolved}} \\"
            )
            continue
        lines.append(
            f"  {name} & {_f(r['wall_cells'])} & {_f(r['frac_type_i']*100)} & "
            f"{_f(r['frac_type_iv']*100)} & {_fnec(r['invariant_nec_min'])} & "
            f"{_f(r.get('miss_wec_pct'))} & {_f(r.get('miss_nec_pct'))} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=str, nargs="+", default=ORDER)
    p.add_argument("--N", type=int, default=None,
                   help="override grid N for all constructions (else per-spec)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    reg = construction_registry()
    metrics = args.metrics
    if args.smoke:
        metrics = ["Alcubierre", "Rodal", "Fuchs"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print("CROSS-CONSTRUCTION ALL-OBSERVER VERIFICATION")
    print("=" * 70)
    rows = []
    for name in metrics:
        spec = reg[name]
        n = args.N if args.N is not None else (24 if args.smoke else spec.grid_n)
        r = verify_one(spec, spec.default_speed, n)
        rows.append(r)
        if r.get("resolved", False):
            print(f"  {name:>12s}  cells={r['wall_cells']:.0f}  "
                  f"TypeI={r['frac_type_i']*100:5.1f}% TypeIV={r['frac_type_iv']*100:5.1f}%  "
                  f"NECmin={r['invariant_nec_min']:.3g}  "
                  f"missW/N={r.get('miss_wec_pct')}/{r.get('miss_nec_pct')}")
        else:
            print(f"  {name:>12s}  {r.get('note')}")

    out_path = os.path.join(RESULTS_DIR, "construction_verification.json")
    dump_json({"order": metrics, "rows": rows}, out_path)
    print(f"\nWrote {out_path}")

    if not args.smoke:
        write_table(rows, os.path.join(TABLES_DIR, "construction_verification.tex"))


if __name__ == "__main__":
    main()
