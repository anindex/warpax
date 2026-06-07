"""Contribution 2: independent all-observer verification of warp-drive positive-energy claims.

At matched family parameters (R=1, sigma=8) on wall-clustered grids, and using
ONLY the frame-independent eigenstructure of T^a_b, we report -- wall-restricted
and volume-weighted -- for each metric:

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
import json
import os

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
from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions.filtering import shape_function_mask
from warpax.energy_conditions.frame_free import certify_grid_frame_free, type_fractions
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.grids import wall_clustered
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

BOUNDS = [(-3, 3)] * 3
F_LOW, F_HIGH = 0.1, 0.9

METRICS = {
    "Alcubierre": (AlcubierreMetric, {}),
    "Natário": (NatarioMetric, {}),
    "Van den Broeck": (VanDenBroeckMetric,
                       {"R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0}),
    "Rodal": (RodalMetric, {}),
}
ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]


def _instantiate(name, v_s):
    cls, extra = METRICS[name]
    return cls(v_s=v_s, R=1.0, sigma=8.0, **extra)


def verify_metric(name, v_s, N):
    shape = (N, N, N)
    metric = _instantiate(name, v_s)
    grid = wall_clustered(metric, BOUNDS, shape, a=1.2)
    curv = evaluate_curvature_grid(metric, grid, batch_size=256)
    T, g, gi = curv.stress_energy, curv.metric, curv.metric_inv

    coords = build_coord_batch(grid, t=0.0)
    mask = shape_function_mask(metric, coords, shape, f_low=F_LOW, f_high=F_HIGH)
    mask_flat = np.asarray(jnp.reshape(mask, (-1,))).astype(bool)
    vol_w = grid.volume_weights_array

    ff = certify_grid_frame_free(T, g, gi, solver="standard")
    fr = type_fractions(ff, mask=mask, volume_weights=vol_w)
    miss = single_frame_miss(T, g, gi, mask=mask_flat, volume_weights=np.asarray(jnp.reshape(vol_w, (-1,))))
    exotic = integrated_exotic_content(T, g, gi, vol_w, mask=mask)
    peaks = peak_proper_energy_deficit(T, g, gi, mask=mask_flat)

    nec_inv = np.asarray(ff.nec_margins).ravel()
    typeI_wall = mask_flat & (np.asarray(ff.he_types).ravel() == 1.0) & np.isfinite(nec_inv)
    nec_min = float(np.min(nec_inv[typeI_wall])) if typeI_wall.any() else float("nan")

    def _pct(x):
        return x * 100.0 if x is not None else None

    return {
        "metric": name, "v_s": v_s, "N": N,
        "frac_type_i": fr["frac_type_i"], "frac_type_iv": fr["frac_type_iv"],
        "wall_n": fr["n_selected"],
        "invariant_nec_min": nec_min,
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
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v-s", type=float, default=0.5)
    p.add_argument("--N", type=int, default=50)
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

    with open(os.path.join(RESULTS_DIR, "invariant_verification.json"), "w") as f:
        json.dump({"config": vars(args), "rows": rows, "reduction_factors": rfac},
                  f, indent=2)
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
