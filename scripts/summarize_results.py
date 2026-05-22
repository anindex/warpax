"""Print a compact summary of the analysis result files.

Reads results/*.npz and results/*.json and prints the key per-metric
quantities (wall-restricted Type-IV fractions, vacuum counts, miss rates,
convergence diagnostics) so they can be inspected without opening each
file by hand.

Usage
-----
    PYTHONPATH=src python scripts/summarize_results.py
"""
from __future__ import annotations

import json
import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def fmt(x, fmt_spec="{:.3g}"):
    if x is None:
        return "--"
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    try:
        return fmt_spec.format(float(x))
    except Exception:
        return str(x)


def load_npz(name, v_s):
    path = os.path.join(RESULTS_DIR, f"{name}_vs{v_s}.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path, allow_pickle=False))


def report_metric(name, v_s):
    d = load_npz(name, v_s)
    if d is None:
        print(f"  {name:>14s} v_s={v_s:<4.2f}  (cache missing)")
        return
    line = f"  {name:>14s} v_s={v_s:<4.2f} "
    # Wall stats (new wall-restricted fields)
    wn = d.get("wall_n_total", None)
    wtIV = d.get("wall_frac_type_iv", None)
    if wn is not None and wtIV is not None:
        wIV_pct = 100.0 * float(wtIV)
        line += f"Wall N={int(wn):4d}  Wall%TypeIV={wIV_pct:5.1f}  "
    # Vacuum count
    nv = d.get("n_vacuum", None)
    if nv is not None and int(nv) >= 0:
        line += f"n_vacuum={int(nv)}  "
    # Miss rates
    for cond in ("nec", "wec", "sec", "dec"):
        m = d.get(f"{cond}_missed", None)
        if m is not None and m.size > 0:
            pct = 100.0 * float(np.nanmean(m))
            line += f"{cond.upper()}_miss={pct:5.2f}%  "
    print(line)


def main():
    print("=" * 100)
    print("Result summary")
    print("=" * 100)
    print()
    print("PER-METRIC × VELOCITY SUMMARY")
    print("-" * 100)
    metrics = ["alcubierre", "rodal", "vdb", "natario", "warpshell"]
    velocities = [0.1, 0.5, 0.9, 0.99]
    for m in metrics:
        for v in velocities:
            report_metric(m, v)
        print()

    # Lentz (excluded from benchmark, but should still cache)
    print("LENTZ (excluded from benchmark, qualitative check only)")
    print("-" * 100)
    for v in velocities:
        report_metric("lentz", v)
    print()

    # Schwarzschild
    print("SCHWARZSCHILD (v_s = 0.0)")
    print("-" * 100)
    report_metric("schwarzschild", 0.0)
    print()

    # Comparison table aggregate
    ct_path = os.path.join(RESULTS_DIR, "comparison_table.json")
    if os.path.exists(ct_path):
        with open(ct_path) as f:
            rows = json.load(f)
        print(f"COMPARISON TABLE ({len(rows)} rows)")
        print("-" * 100)
        for row in rows:
            print(
                f"  {row.get('metric','?'):>14s} v_s={row.get('v_s',0):.2f}  "
                f"NEC miss={row.get('nec_pct_missed',0):.2f}%  "
                f"WEC miss={row.get('wec_pct_missed',0):.2f}%  "
                f"SEC miss={row.get('sec_pct_missed',0):.2f}%  "
                f"DEC miss={row.get('dec_pct_missed',0):.2f}%"
            )
        print()

    # Wall-restricted aggregate
    wr_path = os.path.join(RESULTS_DIR, "wall_restricted_analysis.json")
    if os.path.exists(wr_path):
        with open(wr_path) as f:
            wr = json.load(f)
        print(f"WALL-RESTRICTED ANALYSIS")
        print("-" * 100)
        print(json.dumps(wr, indent=2)[:2000])
        print()

    # Clustered convergence
    cc_path = os.path.join(RESULTS_DIR, "clustered_convergence_alcubierre.json")
    if os.path.exists(cc_path):
        with open(cc_path) as f:
            cc = json.load(f)
        print(f"CLUSTERED CONVERGENCE (Alcubierre)")
        print("-" * 100)
        for r in cc.get("results", []):
            print(
                f"  {r['grid']:<16s} N_wall={r.get('wall_n_total',0):4d}  "
                f"Wall%TypeIV={100*r.get('wall_frac_type_iv',0):5.1f}  "
                f"Full%TypeI={100*r.get('full_frac_type_i',0):5.1f}  "
                f"NEC_min={r.get('nec_min_margin',0):.3e}  "
                f"DEC_min={r.get('dec_min_margin',0):.3e}"
            )
        print()

    rmc_path = os.path.join(RESULTS_DIR, "clustered_convergence_rodal_matched.json")
    if os.path.exists(rmc_path):
        with open(rmc_path) as f:
            rmc = json.load(f)
        print(f"CLUSTERED CONVERGENCE (Rodal matched at R=1, sigma=8)")
        print("-" * 100)
        for r in rmc.get("results", []):
            if "error" in r:
                print(f"  {r['grid']:<16s} ERROR: {r['error'][:100]}")
            else:
                print(
                    f"  {r['grid']:<16s} N_wall={r.get('wall_n_total',0):4d}  "
                    f"Wall%TypeIV={100*r.get('wall_frac_type_iv',0):5.1f}  "
                    f"NEC_miss={fmt(r.get('wall_nec_miss_rate'))}  "
                    f"DEC_miss={fmt(r.get('wall_dec_miss_rate'))}"
                )
        print()


if __name__ == "__main__":
    main()
