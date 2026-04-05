
"""Test Rodal at matched parameters R=1, sigma=8 for numerical feasibility.

Runs Eulerian vs robust EC comparison at resolutions N=30/50/70 on a [-3,3]^3
domain (same as Alcubierre/Natario/VDB). Feasibility criterion: f_miss
stability across resolutions within +/-5%.

See also: ``run_rodal_resolution.py`` for the basic Rodal resolution study
at native parameters and the broader N = {25, 50, 100} sweep.

Outputs:
  - results/rodal_matched_resolution.json (per-resolution f_miss data)
  - results/rodal_matched_report.md (feasibility verdict)

Usage
-----
    python scripts/run_rodal_matched_resolution.py
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpax.metrics import RodalMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.analysis import compare_eulerian_vs_robust


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESOLUTIONS = [30, 50, 70]  # three-point resolution sweep
V_S = 0.5
R = 1.0       # matched parameter (was R=100 in the native Rodal config)
SIGMA = 8.0   # matched parameter (was sigma=0.03 in the native Rodal config)
DOMAIN = [(-3, 3)] * 3  # compact domain for R=1 (not [-300,300]^3)
TOLERANCE = 0.05  # +/-5% stability criterion across adjacent resolutions
CONDITIONS = ["nec", "wec", "sec", "dec"]


# ---------------------------------------------------------------------------
# Resolution study
# ---------------------------------------------------------------------------


def run_resolution_study():
    """Run Rodal at multiple resolutions and report missed fractions."""
    print("=" * 70)
    print("Rodal Matched-Parameter Resolution Stability Study")
    print(f"v_s={V_S}, R={R}, sigma={SIGMA}, domain={DOMAIN[0]}")
    print(f"Resolutions: {RESOLUTIONS}")
    print("=" * 70)

    results = []
    for N in RESOLUTIONS:
        grid = GridSpec(bounds=DOMAIN, shape=(N, N, N))
        n_total = N ** 3

        print(f"\n--- Resolution: {N}^3 = {n_total} points ---")
        metric = RodalMetric(v_s=V_S, R=R, sigma=SIGMA)

        t0 = time.time()
        curv = evaluate_curvature_grid(metric, grid, batch_size=256)

        # Use n_starts=8 for consistency with paper analysis
        comparison = compare_eulerian_vs_robust(
            curv.stress_energy,
            curv.metric,
            curv.metric_inv,
            grid.shape,
            n_starts=8,
            batch_size=64,
        )
        elapsed = time.time() - t0

        # Extract missed fractions per condition
        row = {
            "N": N,
            "n_total": n_total,
        }
        for cond in CONDITIONS:
            rob_margins = comparison.robust_margins[cond]
            eul_margins = comparison.eulerian_margins[cond]

            # Total violations (from robust/hybrid truth)
            violated = float(jnp.sum(rob_margins < -1e-10)) / n_total * 100
            # Missed: Eulerian says OK but robust says violated
            # Use >= 0.0 threshold to match comparison.py convention
            missed = float(jnp.sum(
                (eul_margins >= 0.0) & (rob_margins < -1e-10)
            )) / n_total * 100
            min_margin = float(jnp.nanmin(rob_margins))

            row[f"{cond}_total"] = violated
            row[f"{cond}_missed"] = missed
            row[f"{cond}_min"] = min_margin

        # Classification stats
        type_i_count = int(jnp.sum(comparison.he_types == 1))
        row["type_i_count"] = type_i_count
        row["type_i_pct"] = type_i_count / n_total * 100
        row["elapsed"] = elapsed
        results.append(row)

        print(f"  NEC: {row['nec_total']:.2f}% violated, {row['nec_missed']:.2f}% missed")
        print(f"  WEC: {row['wec_total']:.2f}% violated, {row['wec_missed']:.2f}% missed")
        print(f"  SEC: {row['sec_total']:.2f}% violated, {row['sec_missed']:.2f}% missed")
        print(f"  DEC: {row['dec_total']:.2f}% violated, {row['dec_missed']:.2f}% missed")
        print(f"  Type I: {type_i_count} / {n_total} ({row['type_i_pct']:.1f}%)")
        print(f"  Time: {elapsed:.1f}s")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Missed Fraction (%) by Resolution")
    print("=" * 70)
    print(f"{'N':>5s} {'NEC_miss':>10s} {'WEC_miss':>10s} {'SEC_miss':>10s} {'DEC_miss':>10s}")
    for r in results:
        print(f"{r['N']:>5d} {r['nec_missed']:>10.2f} {r['wec_missed']:>10.2f} "
              f"{r['sec_missed']:>10.2f} {r['dec_missed']:>10.2f}")

    print(f"\n{'N':>5s} {'NEC_tot':>10s} {'WEC_tot':>10s} {'SEC_tot':>10s} {'DEC_tot':>10s}")
    for r in results:
        print(f"{r['N']:>5d} {r['nec_total']:>10.2f} {r['wec_total']:>10.2f} "
              f"{r['sec_total']:>10.2f} {r['dec_total']:>10.2f}")

    return results


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------


def check_f_miss_stability(results):
    """Check f_miss stability across resolutions within +/-5%.

    For each condition, collect f_miss values across resolutions and check
    whether max deviation from mean is within TOLERANCE.

    Returns
    -------
    dict
        Per-condition stability verdict with mean, max_deviation, stable flag.
    """
    stability = {}
    for cond in CONDITIONS:
        values = [r[f"{cond}_missed"] for r in results]
        mean_val = np.mean(values)

        if mean_val == 0.0:
            # Zero miss rate everywhere is trivially stable
            max_deviation = 0.0
            stable = True
        else:
            deviations = [abs(v - mean_val) / mean_val for v in values]
            max_deviation = max(deviations)
            stable = max_deviation <= TOLERANCE

        stability[cond] = {
            "values": values,
            "mean": float(mean_val),
            "max_deviation": float(max_deviation),
            "stable": bool(stable),
        }

    return stability


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def save_json(results, stability, start_time):
    """Save structured results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_stable = all(v["stable"] for v in stability.values())

    output = {
        "metadata": {
            "date": start_time,
            "script": "scripts/run_rodal_matched_resolution.py",
            "parameters": {"v_s": V_S, "R": R, "sigma": SIGMA},
            "resolutions": RESOLUTIONS,
            "domain": [list(b) for b in DOMAIN],
            "tolerance": TOLERANCE,
        },
        "per_resolution": results,
        "stability": stability,
        "verdict": {
            "feasible": all_stable,
            "criterion": (
                f"f_miss stable within +/-{TOLERANCE * 100:.0f}% "
                f"across N={RESOLUTIONS}"
            ),
        },
    }

    outpath = os.path.join(RESULTS_DIR, "rodal_matched_resolution.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved to {outpath}")
    return output


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def save_report(results, stability, start_time):
    """Save human-readable feasibility report to markdown."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_stable = all(v["stable"] for v in stability.values())
    verdict_word = "FEASIBLE" if all_stable else "NOT FEASIBLE"
    verb = "is" if all_stable else "is not"

    lines = []
    lines.append("# Rodal Matched-Parameter Feasibility Report\n")
    lines.append(f"**Date:** {start_time}\n")
    lines.append(f"**Script:** `scripts/run_rodal_matched_resolution.py`\n")
    lines.append(
        f"**Parameters:** v_s={V_S}, R={R}, sigma={SIGMA}, "
        f"domain=[{DOMAIN[0][0]},{DOMAIN[0][1]}]^3\n"
    )

    # Verdict
    lines.append("## Verdict\n")
    lines.append(
        f"**{verdict_word}**: f_miss {verb} stable within +/-{TOLERANCE * 100:.0f}% "
        f"across resolutions N={', '.join(str(n) for n in RESOLUTIONS)}.\n"
    )

    # Per-resolution table
    lines.append("## Per-Resolution Results\n")
    lines.append(
        "| N | n_total | NEC_miss% | WEC_miss% | SEC_miss% | DEC_miss% "
        "| Type_I_pct | Time (s) |"
    )
    lines.append(
        "|--:|--------:|----------:|----------:|----------:|----------:"
        "|-----------:|---------:|"
    )
    for r in results:
        lines.append(
            f"| {r['N']} | {r['n_total']} "
            f"| {r['nec_missed']:.2f} | {r['wec_missed']:.2f} "
            f"| {r['sec_missed']:.2f} | {r['dec_missed']:.2f} "
            f"| {r['type_i_pct']:.1f} | {r['elapsed']:.1f} |"
        )
    lines.append("")

    # Stability table
    lines.append("## Stability Analysis\n")
    lines.append("| Condition | Stable | Mean f_miss | Max Deviation |")
    lines.append("|-----------|--------|------------:|--------------:|")
    for cond in CONDITIONS:
        s = stability[cond]
        stable_str = "Yes" if s["stable"] else "No"
        lines.append(
            f"| {cond.upper()} | {stable_str} "
            f"| {s['mean']:.4f} | {s['max_deviation']:.4f} |"
        )
    lines.append("")

    # Note for paper
    lines.append("## Note for Paper\n")
    if all_stable:
        lines.append(
            "Matched parameters (R=1.0, sigma=8.0) produce stable f_miss across "
            "resolutions N=30, 50, 70 on the compact [-3,3]^3 domain. These "
            "parameters are suitable for the main cross-metric comparison table "
            "alongside Alcubierre, Natario, and Van den Broeck.\n"
        )
    else:
        # Identify which conditions are unstable
        unstable = [c.upper() for c in CONDITIONS if not stability[c]["stable"]]
        lines.append(
            f"Rodal at matched parameters (R=1.0, sigma=8.0) exhibits "
            f"unstable f_miss for {', '.join(unstable)} across resolutions. "
            f"Report Rodal at native parameters (R=100, sigma=0.03) in the main "
            f"comparison table with a comparability caveat.\n"
        )

    report_path = os.path.join(RESULTS_DIR, "rodal_matched_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run Rodal matched-parameter feasibility study."""
    start_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    results = run_resolution_study()
    stability = check_f_miss_stability(results)

    # Print stability summary
    print("\n" + "=" * 70)
    print("STABILITY ANALYSIS (+/-5% criterion)")
    print("=" * 70)
    for cond in CONDITIONS:
        s = stability[cond]
        status = "STABLE" if s["stable"] else "UNSTABLE"
        print(
            f"  {cond.upper()}: {status} "
            f"(mean={s['mean']:.4f}, max_dev={s['max_deviation']:.4f})"
        )

    all_stable = all(v["stable"] for v in stability.values())
    print(f"\nOverall verdict: {'FEASIBLE' if all_stable else 'NOT FEASIBLE'}")

    save_json(results, stability, start_time)
    save_report(results, stability, start_time)


if __name__ == "__main__":
    main()
