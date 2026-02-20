
"""C1-vs-C2 WarpShell transition comparison.

Computes Eulerian vs robust EC analysis for WarpShell at both transition
orders (C1 cubic and C2 quintic) across 4 warp velocities.  Generates:

    results/c1_vs_c2_comparison.json   machine-readable comparison data
    results/c1_vs_c2_comparison.tex    LaTeX table for paper appendix
    figures/c1_vs_c2_margins.pdf       2-panel comparison plot

This script is self-contained: it computes both C1 and C2 in-memory and
does NOT read from or write to the main cached .npz files.  Run it before
the main analysis overwrites the results cache.

Usage
-----
    python scripts/run_c1_vs_c2_comparison.py
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

# Non-interactive backend (before any other matplotlib import)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

from warpax.analysis import compare_eulerian_vs_robust
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.metrics import WarpShellMetric
from warpax.visualization._style import apply_style, DOUBLE_COL, COLORS

# ---------------------------------------------------------------------------
# Configuration (matches paper exactly)
# ---------------------------------------------------------------------------

VELOCITIES = [0.1, 0.5, 0.9, 0.99]
METRIC_PARAMS = {"R_1": 0.5, "R_2": 1.0}
GRID_SPEC = GridSpec(bounds=[(-5, 5)] * 3, shape=(50, 50, 50))
N_STARTS = 8
BATCH_SIZE = 64

# Output paths
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
JSON_PATH = os.path.join(RESULTS_DIR, "c1_vs_c2_comparison.json")
TEX_PATH = os.path.join(RESULTS_DIR, "c1_vs_c2_comparison.tex")
PLOT_PATH = os.path.join(FIGURES_DIR, "c1_vs_c2_margins.pdf")


# ---------------------------------------------------------------------------
# C2 continuity measurement: seam discontinuity via third finite difference
# ---------------------------------------------------------------------------


def measure_seam_continuity(
    metric: WarpShellMetric,
    n_points: int = 500,
) -> float:
    """Measure second-derivative discontinuity at shell transition seams.

    Sweeps along the x-axis near each shell boundary and computes the
    maximum absolute third finite difference of the lapse divided by dr.
    This quantifies the second-derivative jump (Riemann tensor
    discontinuity) at the transition seams.

    Parameters
    ----------
    metric : WarpShellMetric
        WarpShell metric instance.
    n_points : int
        Number of sample points in each boundary sweep.

    Returns
    -------
    float
        Maximum |d^3 alpha / dx^3| across both boundaries.
    """
    # Compute smooth_width (same default as WarpShellMetric)
    sw = metric.smooth_width if metric.smooth_width is not None else 0.12 * (metric.R_2 - metric.R_1)

    # Boundary locations: inner seam at R_1 - sw, outer seam at R_2 + sw
    boundaries = [metric.R_1 - sw, metric.R_2 + sw]

    # Use jax.grad for analytic derivatives of lapse w.r.t. x
    def lapse_x(x_val):
        """Evaluate lapse at (t=0, x=x_val, y=0, z=0)."""
        coords = jnp.array([0.0, x_val, 0.0, 0.0])
        return metric.lapse(coords)

    d1 = jax.grad(lapse_x)
    d2 = jax.grad(d1)
    d3 = jax.grad(d2)
    d3_jit = jax.jit(d3)

    max_d3 = 0.0
    for r_b in boundaries:
        r_lo = r_b - 1.0
        r_hi = r_b + 1.0
        xs = jnp.linspace(r_lo, r_hi, n_points)
        d3_vals = jax.vmap(d3_jit)(xs)
        peak = float(jnp.max(jnp.abs(d3_vals)))
        if peak > max_d3:
            max_d3 = peak

    return max_d3


# ---------------------------------------------------------------------------
# Per-velocity comparison
# ---------------------------------------------------------------------------


def run_comparison_at_velocity(
    v_s: float,
    order: int,
) -> dict:
    """Run full EC analysis for WarpShell at a given velocity and order.

    Returns a dict with classification stats, EC margins, and seam
    continuity metric.
    """
    metric = WarpShellMetric(
        **METRIC_PARAMS,
        v_s=v_s,
        transition_order=order,
    )

    # Curvature grid
    curv = evaluate_curvature_grid(metric, GRID_SPEC, batch_size=256)

    # Eulerian vs robust comparison
    comparison = compare_eulerian_vs_robust(
        curv.stress_energy,
        curv.metric,
        curv.metric_inv,
        GRID_SPEC.shape,
        n_starts=N_STARTS,
        batch_size=BATCH_SIZE,
    )

    # Classification
    n_total = int(jnp.prod(jnp.array(GRID_SPEC.shape)))
    stats = comparison.classification_stats
    n_type_i = int(stats["n_type_i"])
    n_type_ii = int(stats["n_type_ii"])
    n_type_iv = int(stats["n_type_iv"])
    pct_type_i = n_type_i / n_total * 100.0
    pct_type_ii = n_type_ii / n_total * 100.0
    pct_type_iv = n_type_iv / n_total * 100.0

    # EC margins
    min_nec_robust = float(jnp.nanmin(jnp.array(comparison.robust_margins["nec"])))
    min_wec_robust = float(jnp.nanmin(jnp.array(comparison.robust_margins["wec"])))
    min_dec_robust = float(jnp.nanmin(jnp.array(comparison.robust_margins["dec"])))

    pct_missed_nec = comparison.pct_missed["nec"]
    pct_missed_wec = comparison.pct_missed["wec"]
    pct_missed_dec = comparison.pct_missed["dec"]

    # Seam continuity
    seam_d3 = measure_seam_continuity(metric)

    return {
        "n_type_i": n_type_i,
        "n_type_ii": n_type_ii,
        "n_type_iv": n_type_iv,
        "pct_type_i": pct_type_i,
        "pct_type_ii": pct_type_ii,
        "pct_type_iv": pct_type_iv,
        "min_nec_robust": min_nec_robust,
        "min_wec_robust": min_wec_robust,
        "min_dec_robust": min_dec_robust,
        "pct_missed_nec": pct_missed_nec,
        "pct_missed_wec": pct_missed_wec,
        "pct_missed_dec": pct_missed_dec,
        "max_d3_lapse": seam_d3,
    }


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------


def generate_latex_table(rows: list[dict]) -> str:
    """Generate a LaTeX tabular environment from comparison rows."""
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\caption{WarpShell C1 vs C2 transition comparison across warp velocities.}")
    lines.append(r"\label{tab:c1-vs-c2}")
    lines.append(r"\begin{tabular}{c cc cc cc cc}")
    lines.append(r"\hline\hline")
    lines.append(
        r"$v_s$ & \multicolumn{2}{c}{Type I (\%)} & "
        r"\multicolumn{2}{c}{Type IV (\%)} & "
        r"\multicolumn{2}{c}{$\min\;m_{\mathrm{NEC}}$} & "
        r"\multicolumn{2}{c}{$\max|d^3\alpha/dx^3|$} \\"
    )
    lines.append(
        r" & C1 & C2 & C1 & C2 & C1 & C2 & C1 & C2 \\"
    )
    lines.append(r"\hline")

    for row in rows:
        v_s = row["v_s"]
        c1 = row["c1"]
        c2 = row["c2"]

        # Format NEC margins in scientific notation
        def fmt_sci(val):
            if abs(val) < 1e-3 or abs(val) > 1e3:
                exp = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
                mantissa = val / (10 ** exp)
                return f"${mantissa:.2f} \\times 10^{{{exp}}}$"
            return f"${val:.4f}$"

        def fmt_d3(val):
            if val < 1e-3:
                exp = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
                mantissa = val / (10 ** exp)
                return f"${mantissa:.1f} \\times 10^{{{exp}}}$"
            return f"${val:.2f}$"

        lines.append(
            f"  {v_s} & "
            f"{c1['pct_type_i']:.1f} & {c2['pct_type_i']:.1f} & "
            f"{c1['pct_type_iv']:.1f} & {c2['pct_type_iv']:.1f} & "
            f"{fmt_sci(c1['min_nec_robust'])} & {fmt_sci(c2['min_nec_robust'])} & "
            f"{fmt_d3(c1['max_d3_lapse'])} & {fmt_d3(c2['max_d3_lapse'])} \\\\"
        )

    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def generate_plot(rows: list[dict]) -> None:
    """Generate 2-panel C1 vs C2 comparison plot."""
    apply_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    velocities = [row["v_s"] for row in rows]
    c1_type_i = [row["c1"]["pct_type_i"] for row in rows]
    c2_type_i = [row["c2"]["pct_type_i"] for row in rows]
    c1_nec = [abs(row["c1"]["min_nec_robust"]) for row in rows]
    c2_nec = [abs(row["c2"]["min_nec_robust"]) for row in rows]

    # Left panel: grouped bar chart of Type I percentage
    x = np.arange(len(velocities))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, c1_type_i, width, label="C1 (cubic)",
            color=COLORS[0], alpha=0.85)
    bars2 = ax1.bar(x + width / 2, c2_type_i, width, label="C2 (quintic)",
            color=COLORS[1], alpha=0.85)
    ax1.bar_label(bars1, fmt="%.2f", fontsize=6, padding=1)
    ax1.bar_label(bars2, fmt="%.2f", fontsize=6, padding=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(v) for v in velocities])
    ax1.set_xlabel(r"$v_s$")
    ax1.set_ylabel("Type I (\\%)")
    # Start y-axis near the minimum value with tight zoom
    all_pcts = c1_type_i + c2_type_i
    ymin = max(0, min(all_pcts) - 0.3)
    ax1.set_ylim(ymin, 100.2)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_title("(a) Hawking--Ellis Type I", fontsize=10)

    # Right panel: min NEC margin (log scale)
    ax2.plot(velocities, c1_nec, "o--", color=COLORS[0], label="C1 (cubic)",
             markersize=5)
    ax2.plot(velocities, c2_nec, "s-", color=COLORS[1], label="C2 (quintic)",
             markersize=5)
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$v_s$")
    ax2.set_ylabel(r"$|\min\;m_{\mathrm{NEC}}|$")
    ax2.legend(loc="best", fontsize=8)
    ax2.set_title("(b) Worst-case NEC margin", fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=300)
    plt.close(fig)
    print(f"  Saved plot: {PLOT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("WarpShell C1 vs C2 Transition Comparison")
    print("=" * 60)
    print(f"Grid: {GRID_SPEC.shape}, Velocities: {VELOCITIES}")
    print(f"N_STARTS={N_STARTS}, BATCH_SIZE={BATCH_SIZE}")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    t_total_start = time.time()
    rows = []

    for i, v_s in enumerate(VELOCITIES):
        print(f"--- v_s = {v_s} ({i + 1}/{len(VELOCITIES)}) ---")

        row = {"v_s": v_s}

        for order, label in [(1, "c1"), (2, "c2")]:
            print(f"  Computing {label.upper()} (transition_order={order})...")
            t0 = time.time()
            result = run_comparison_at_velocity(v_s, order)
            dt = time.time() - t0
            print(f"    Done in {dt:.1f}s | "
                  f"Type I: {result['pct_type_i']:.1f}% | "
                  f"Type IV: {result['pct_type_iv']:.1f}% | "
                  f"min NEC: {result['min_nec_robust']:.4e} | "
                  f"max |d^3 alpha|: {result['max_d3_lapse']:.4e}")
            row[label] = result

        # Prefix keys for JSON compatibility
        row["c1_pct_type_i"] = row["c1"]["pct_type_i"]
        row["c2_pct_type_i"] = row["c2"]["pct_type_i"]

        rows.append(row)
        print()

    t_total = time.time() - t_total_start
    print(f"Total compute time: {t_total:.1f}s ({t_total / 60:.1f} min)")
    print()

    # Save JSON
    json_data = {
        "metadata": {
            "grid_shape": list(GRID_SPEC.shape),
            "n_starts": N_STARTS,
            "batch_size": BATCH_SIZE,
            "date": datetime.now(timezone.utc).isoformat(),
            "velocities": VELOCITIES,
            "metric_params": METRIC_PARAMS,
        },
        "rows": [],
    }
    for row in rows:
        json_row = {
            "v_s": row["v_s"],
            "c1_pct_type_i": row["c1"]["pct_type_i"],
            "c2_pct_type_i": row["c2"]["pct_type_i"],
            "c1_pct_type_iv": row["c1"]["pct_type_iv"],
            "c2_pct_type_iv": row["c2"]["pct_type_iv"],
            "c1_min_nec_robust": row["c1"]["min_nec_robust"],
            "c2_min_nec_robust": row["c2"]["min_nec_robust"],
            "c1_min_wec_robust": row["c1"]["min_wec_robust"],
            "c2_min_wec_robust": row["c2"]["min_wec_robust"],
            "c1_min_dec_robust": row["c1"]["min_dec_robust"],
            "c2_min_dec_robust": row["c2"]["min_dec_robust"],
            "c1_pct_missed_nec": row["c1"]["pct_missed_nec"],
            "c2_pct_missed_nec": row["c2"]["pct_missed_nec"],
            "c1_pct_missed_wec": row["c1"]["pct_missed_wec"],
            "c2_pct_missed_wec": row["c2"]["pct_missed_wec"],
            "c1_max_d3_lapse": row["c1"]["max_d3_lapse"],
            "c2_max_d3_lapse": row["c2"]["max_d3_lapse"],
        }
        json_data["rows"].append(json_row)

    with open(JSON_PATH, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON: {JSON_PATH}")

    # Save LaTeX table
    tex = generate_latex_table(rows)
    with open(TEX_PATH, "w") as f:
        f.write(tex)
    print(f"Saved LaTeX: {TEX_PATH}")

    # Generate plot
    generate_plot(rows)

    # Print summary table
    print()
    print("=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)
    header = (
        f"{'v_s':>5s} | "
        f"{'Type I C1':>10s} {'Type I C2':>10s} | "
        f"{'Type IV C1':>11s} {'Type IV C2':>11s} | "
        f"{'min NEC C1':>12s} {'min NEC C2':>12s} | "
        f"{'max|d3a| C1':>12s} {'max|d3a| C2':>12s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        c1 = row["c1"]
        c2 = row["c2"]
        print(
            f"{row['v_s']:5.2f} | "
            f"{c1['pct_type_i']:9.1f}% {c2['pct_type_i']:9.1f}% | "
            f"{c1['pct_type_iv']:10.1f}% {c2['pct_type_iv']:10.1f}% | "
            f"{c1['min_nec_robust']:12.4e} {c2['min_nec_robust']:12.4e} | "
            f"{c1['max_d3_lapse']:12.4e} {c2['max_d3_lapse']:12.4e}"
        )
    print("=" * 90)
    print("\nDone.")


if __name__ == "__main__":
    main()
