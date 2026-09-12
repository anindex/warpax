"""Resolution stability of empirical curvature-invariant speed fits.

Refit wall-peak magnitudes to A v_s^q on the same speed window at each
spatial resolution. The observed exponent spread does not certify a leading
asymptotic power or a continuum maximum. The invariants exist for all types.

Outputs: results/curvature_convergence.json and tables/curvature_convergence.tex.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from _benchmark_grid import BOX, CLUSTER_A, N_LADDER
from _json_io import dump_json
from _json_io import write_table as write_tex_table
from run_curvature_scaling import (
    METRIC_ORDER,
    fit_power_law,
    run_point,
)

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

# Subluminal branch used for the exponent fit (three or more points required).
# Same subluminal window as run_curvature_scaling.py.
VELOCITIES = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
INVARIANTS = (("weyl_squared", r"Weyl $|C^2|$"), ("ricci_squared", r"Ricci $|R_{ab}R^{ab}|$"))


def _f(x, nd=2):
    return f"{x:.{nd}f}" if (x is not None and np.isfinite(x)) else "--"


def _stable_q(qs, tol=0.15):
    """Spread of the fitted exponent across the resolution ladder."""
    vals = [q for q in qs if q is not None and np.isfinite(q)]
    if len(vals) < 2:
        return {"mean": (vals[0] if vals else None), "max_dev": None, "stable": False}
    mean = float(np.mean(vals))
    max_dev = float(np.max(np.abs(np.array(vals) - mean)))
    return {"mean": mean, "max_dev": max_dev, "stable": bool(max_dev <= tol)}


def write_table(fits, out_path):
    ns = N_LADDER
    hdr = " & ".join(f"$N{{=}}{n}$" for n in ns)
    lines = [
        r"\begin{tabular}{@{}l l ccc c@{}}",
        r"  \toprule",
        r"  Metric & Invariant & \multicolumn{3}{c}{fitted $q$} & Spread \\",
        rf"   & & {hdr} & \\",
        r"  \midrule",
    ]
    for name in METRIC_ORDER:
        first = True
        for key, label in INVARIANTS:
            series = [fits[name][str(n)][key]["q"] for n in ns]
            r2s = [fits[name][str(n)][key].get("r_squared") for n in ns]
            clean = any(r is not None and np.isfinite(r) and r >= 0.99 for r in r2s)
            stab = _stable_q(series)
            mcol = name if first else ""
            first = False
            if not clean or all(q is None for q in series):
                lines.append(
                    rf"  {mcol} & {label} & \multicolumn{{4}}{{c}}{{poor power-law fit "
                    rf"($R^2_{{\log}}<0.99$)}} \\"
                )
                continue
            cells = " & ".join(_f(q) for q in series)
            spread = f"{_f(stab['max_dev'])}" if stab["max_dev"] is not None else "--"
            lines.append(f"  {mcol} & {label} & {cells} & {spread} \\\\")
        lines.append(r"  \midrule")
    lines[-1] = r"  \bottomrule"
    lines.append(r"\end{tabular}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_tex_table(
        out_path,
        lines,
        script="scripts/run_curvature_convergence.py",
        sources="results/curvature_convergence.json",
    )
    print(f"  Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=str, nargs="+", default=METRIC_ORDER)
    p.add_argument("--velocities", type=float, nargs="+", default=VELOCITIES)
    p.add_argument("--ladder", type=int, nargs="+", default=N_LADDER)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.ladder = [16, 20]
        args.velocities = [0.2, 0.3, 0.5]
        args.metrics = ["Alcubierre", "Rodal"]

    print("=" * 72)
    print(f"CURVATURE-EXPONENT CONVERGENCE (ladder N={args.ladder}, a*={CLUSTER_A}, box +-{BOX})")
    print("=" * 72)

    fits = {name: {} for name in args.metrics}
    for N in args.ladder:
        print(f"\nN = {N}")
        rows = []
        for name in args.metrics:
            for v_s in args.velocities:
                rows.append(run_point(name, v_s, N))
        for name in args.metrics:
            fits[name][str(N)] = {key: fit_power_law(rows, name, key) for key, _ in INVARIANTS}
            qs = {key: fits[name][str(N)][key]["q"] for key, _ in INVARIANTS}
            print(
                f"  {name:16s} q(C^2)={_f(qs['weyl_squared'])}  q(Ricci)={_f(qs['ricci_squared'])}"
            )

    # Stability summary across the ladder.
    summary = {}
    for name in args.metrics:
        summary[name] = {}
        for key, _ in INVARIANTS:
            series = [fits[name][str(n)][key]["q"] for n in args.ladder]
            summary[name][key] = _stable_q(series)
    print("\n  Exponent stability across the wall-resolved ladder:")
    for name in args.metrics:
        for key, _ in INVARIANTS:
            s = summary[name][key]
            print(
                f"    {name:16s} {key:14s} mean q={_f(s['mean'])}  "
                f"spread={_f(s['max_dev'])}  stable={s['stable']}"
            )

    dump_json(
        {
            "ladder_N": args.ladder,
            "cluster_a": CLUSTER_A,
            "box": BOX,
            "velocities": args.velocities,
            "fits": fits,
            "summary": summary,
        },
        os.path.join(RESULTS_DIR, "curvature_convergence.json"),
    )
    if not args.smoke:
        write_table(fits, os.path.join(TABLES_DIR, "curvature_convergence.tex"))


if __name__ == "__main__":
    main()
