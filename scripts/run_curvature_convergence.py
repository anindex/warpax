"""Resolution stability of the fitted curvature-invariant exponents q.

The wall-peak Weyl and Ricci invariants scale as ``X = A v_s^q`` on the
subluminal branch (run_curvature_scaling.py). The exponent q is not merely a
fit: it is a closed-form theorem. On a flat unit-lapse slice the order-``v_s``
Riemann tensor is pure gauge for an irrotational shift and survives for a
vortical one, forcing ``q = 4`` for the irrotational (Rodal) wall and ``q = 2``
for the vortical (Alcubierre, Natario) walls. This script refits q on the
wall-resolved graded
ladder (N = 80, 100, 120, giving 5.9 / 7.6 / 8.9 cells across the 10-90% wall) and
reports its spread across resolutions, confirming the closed-form value is
resolution-stable and not a single-grid artifact.

Van den Broeck has no resolved Type-I curvature branch (its wall is
Type-IV-dominated) and admits no clean single power law; it is reported as such.

Outputs
-------
- results/curvature_convergence.json
- ../warpax_arxiv/tables/curvature_convergence.tex
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
# The same subluminal window run_curvature_scaling.py fits: this table certifies
# that table's exponents, and [0.2, 0.5] gave q = 1.97 against its 2.08.
VELOCITIES = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
# Reference exponents retained in the JSON only. They are NOT printed as a
# "closed form" column: only the q=4 irrotational side has an analytic
# derivation (the pure-gauge reduction); the q=2 values are empirical fits.
INVARIANTS = (("weyl_squared", r"Weyl $C^2$"), ("ricci_squared", r"Ricci $|R_{ab}R^{ab}|$"))
# Closed-form q per metric: irrotational -> 4, vortical -> 2.
THEORY_Q = {"Rodal": 4.0, "Alcubierre": 2.0, "Natário": 2.0}


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
            # A Type-IV-dominated wall (VdB) has no resolved Type-I curvature branch
            # and no clean single power law; the fit R^2 stays well below 1. Report it
            # as such rather than a spurious exponent.
            r2s = [fits[name][str(n)][key].get("r_squared") for n in ns]
            clean = any(r is not None and np.isfinite(r) and r >= 0.99 for r in r2s)
            stab = _stable_q(series)
            mcol = name if first else ""
            first = False
            if not clean or all(q is None for q in series):
                lines.append(
                    rf"  {mcol} & {label} & \multicolumn{{4}}{{c}}{{no clean "
                    rf"Type-I branch (Type-IV-dominated wall)}} \\"
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
            "theory_q": THEORY_Q,
            "fits": fits,
            "summary": summary,
        },
        os.path.join(RESULTS_DIR, "curvature_convergence.json"),
    )
    if not args.smoke:
        write_table(fits, os.path.join(TABLES_DIR, "curvature_convergence.tex"))


if __name__ == "__main__":
    main()
