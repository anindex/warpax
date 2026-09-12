"""Wall-restricted Type-IV analysis for the 4 warp drive metrics (Alcubierre, Rodal, VdB, Natário).

For each warp metric, computes Hawking-Ellis Type breakdown and
conditional energy-condition miss rates restricted to the physically
meaningful warp-wall region where the shape function ``f(coords)`` lies
in ``[0.1, 0.9]``. Full-grid statistics are reported alongside for
comparison.

Outputs
------------------------------------
- results/wall_restricted_analysis.json
- results/wall_restricted_report.md
- figures/wall_restricted_type_breakdown.pdf

Usage
-----
    python scripts/run_wall_restricted_analysis.py
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime

import matplotlib
from _json_io import dump_json

matplotlib.use("Agg")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from warpax.analysis import compare_eulerian_vs_robust
from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions.filtering import (
    compute_wall_restricted_stats,
    shape_function_mask,
)
from warpax.energy_conditions.verifier import verify_grid
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.metrics import (
    LentzMetric,
    NatarioMetric,
    RodalMetric,
    VanDenBroeckMetric,
    WarpShellMetric,
)
from warpax.visualization._style import COLORS, DOUBLE_COL, apply_style

# Constants (identical to run_analysis.py so rows line up with Table 8)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

METRICS: dict[str, tuple[type, dict]] = {
    "alcubierre": (AlcubierreMetric, {"R": 1.0, "sigma": 8.0}),
    "rodal": (RodalMetric, {"R": 100.0, "sigma": 0.03}),
    "vdb": (
        VanDenBroeckMetric,
        {
            "R": 1.0,
            "sigma": 8.0,
            "R_tilde": 1.0,
            "alpha_vdb": 0.5,
            "sigma_B": 8.0,
        },
    ),
    "natario": (NatarioMetric, {"R": 1.0, "sigma": 8.0}),
    "lentz": (LentzMetric, {"R": 100.0, "sigma": 8.0}),
    "warpshell": (WarpShellMetric, {"R_1": 0.5, "R_2": 1.0}),
}

GRID_STANDARD = GridSpec(bounds=[(-5, 5)] * 3, shape=(50, 50, 50))
GRID_LARGE_R = GridSpec(bounds=[(-300, 300)] * 3, shape=(50, 50, 50))

GRID_MAP: dict[str, GridSpec] = {
    "alcubierre": GRID_STANDARD,
    "rodal": GRID_LARGE_R,
    "vdb": GRID_STANDARD,
    "natario": GRID_STANDARD,
    "lentz": GRID_LARGE_R,
    "warpshell": GRID_STANDARD,
}

WARP_METRICS = ["alcubierre", "rodal", "vdb", "natario"]

# Subluminal reference velocity.
V_S = 0.5

# Wall bounds used to define the active warp-wall region.
F_LOW = 0.1
F_HIGH = 0.9

# Shared optimizer settings (kept modest to bound total runtime across 4 metrics)
N_STARTS = 8
BATCH_SIZE_OPT = 64
BATCH_SIZE_CURV = 256


# Core analysis function


def analyze_metric(name: str, metric, grid_spec: GridSpec) -> dict:
    """Compute full-grid and wall-restricted stats for a single warp metric.

    Parameters
    ----------
    name : str
        Metric identifier (e.g. ``"alcubierre"``).
    metric : MetricSpecification
        Instantiated warp-drive metric.
    grid_spec : GridSpec
        Grid specification used for curvature evaluation.

    Returns
    -------
    dict
        Structured diagnostics with ``"full_grid"``, ``"wall_restricted"``,
        ``"caveat"`` (Lentz only), ``"wall_mask_sum"``, and
        ``"elapsed_s"`` entries.
    """
    print(f"\n--- {name} ---")
    t0 = time.time()

    # 1. Curvature chain on the grid
    t_curv0 = time.time()
    curv = evaluate_curvature_grid(metric, grid_spec, batch_size=BATCH_SIZE_CURV)
    print(f"  Curvature grid: {time.time() - t_curv0:.1f}s")

    # 2. Eulerian vs robust comparison (provides Eulerian margins + pct_missed)
    t_cmp0 = time.time()
    comparison = compare_eulerian_vs_robust(
        curv.stress_energy,
        curv.metric,
        curv.metric_inv,
        grid_spec.shape,
        n_starts=N_STARTS,
        batch_size=BATCH_SIZE_OPT,
    )
    print(f"  Comparison: {time.time() - t_cmp0:.1f}s")

    # 3. Robust grid (HE types + robust margins) for wall-restricted stats
    t_vg0 = time.time()
    ec_grid = verify_grid(
        curv.stress_energy,
        curv.metric,
        curv.metric_inv,
        n_starts=N_STARTS,
        batch_size=BATCH_SIZE_OPT,
        compute_eulerian=False,
    )
    print(f"  verify_grid: {time.time() - t_vg0:.1f}s")

    # 4. Wall mask from shape_function_value
    coords_batch = build_coord_batch(grid_spec, t=0.0)
    wall_mask = shape_function_mask(
        metric,
        coords_batch,
        grid_spec.shape,
        f_low=F_LOW,
        f_high=F_HIGH,
    )
    wall_mask_sum = int(jnp.sum(wall_mask))

    # 5. Eulerian margins dict for miss-rate computation
    eulerian_margins = {c: comparison.eulerian_margins[c] for c in ("nec", "wec", "sec", "dec")}

    # 6. Wall-restricted stats
    stats = compute_wall_restricted_stats(
        ec_grid,
        wall_mask,
        eulerian_margins=eulerian_margins,
    )

    # 7. Full-grid stats (mask of all-True) for side-by-side comparison
    full_mask = jnp.ones(grid_spec.shape, dtype=bool)
    full_stats = compute_wall_restricted_stats(
        ec_grid,
        full_mask,
        eulerian_margins=eulerian_margins,
    )

    # 8. Lentz caveat (unresolved-lower-bound, reported explicitly)
    caveat = "unresolved_lower_bound" if name == "lentz" else None
    if name == "lentz" and stats.n_total == 0:
        print(" WARNING: Lentz wall_n_total=0 at this resolution")

    elapsed = time.time() - t0
    print(
        f"  Wall: n_total={stats.n_total}, Type-IV={stats.frac_type_iv:.1%}, "
        f"DEC miss={stats.dec_miss_rate}"
    )
    print(f"  Total elapsed: {elapsed:.1f}s")

    full_grid_dict = {
        "n_type_i": int(full_stats.n_type_i),
        "n_type_ii": int(full_stats.n_type_ii),
        "n_type_iii": int(full_stats.n_type_iii),
        "n_type_iv": int(full_stats.n_type_iv),
        "n_total": int(full_stats.n_total),
        "frac_type_i": float(full_stats.frac_type_i),
        "frac_type_ii": float(full_stats.frac_type_ii),
        "frac_type_iii": float(full_stats.frac_type_iii),
        "frac_type_iv": float(full_stats.frac_type_iv),
        "nec_pct_missed": float(comparison.pct_missed["nec"]),
        "wec_pct_missed": float(comparison.pct_missed["wec"]),
        "sec_pct_missed": float(comparison.pct_missed["sec"]),
        "dec_pct_missed": float(comparison.pct_missed["dec"]),
        "max_imag_eigenvalue": (
            float(ec_grid.max_imag_eigenvalue) if ec_grid.max_imag_eigenvalue is not None else None
        ),
    }

    wall_restricted_dict = {
        "n_total": int(stats.n_total),
        "n_type_i": int(stats.n_type_i),
        "n_type_ii": int(stats.n_type_ii),
        "n_type_iii": int(stats.n_type_iii),
        "n_type_iv": int(stats.n_type_iv),
        "frac_type_i": float(stats.frac_type_i),
        "frac_type_ii": float(stats.frac_type_ii),
        "frac_type_iii": float(stats.frac_type_iii),
        "frac_type_iv": float(stats.frac_type_iv),
        "nec_miss_rate": (float(stats.nec_miss_rate) if stats.nec_miss_rate is not None else None),
        "wec_miss_rate": (float(stats.wec_miss_rate) if stats.wec_miss_rate is not None else None),
        "sec_miss_rate": (float(stats.sec_miss_rate) if stats.sec_miss_rate is not None else None),
        "dec_miss_rate": (float(stats.dec_miss_rate) if stats.dec_miss_rate is not None else None),
    }

    return {
        "full_grid": full_grid_dict,
        "wall_restricted": wall_restricted_dict,
        "caveat": caveat,
        "wall_mask_sum": wall_mask_sum,
        "elapsed_s": elapsed,
    }


# Output: JSON


def save_json(results: dict, start_time: str) -> None:
    """Persist structured wall-restricted diagnostics to JSON."""
    output = {
        "metadata": {
            "date": start_time,
            "script": "scripts/run_wall_restricted_analysis.py",
            "grid_resolution": 50,
            "wall_bounds": [F_LOW, F_HIGH],
            "velocity": V_S,
            "n_starts": N_STARTS,
        },
        "metrics": results,
    }
    outpath = os.path.join(RESULTS_DIR, "wall_restricted_analysis.json")
    dump_json(output, outpath)
    print(f"\nJSON saved to {outpath}")


# Output: human-readable markdown


def _fmt_miss(value):
    """Format a possibly-None miss rate as a percentage string."""
    if value is None:
        return "N/A (no robust violations)"
    return f"{value * 100.0:.2f}%"


def save_report(results: dict, start_time: str) -> None:
    """Render full-grid and wall statistics with their distinct denominators."""
    lines = [
        "# Wall-restricted classification and missed violations",
        "",
        f"Date: {start_time}. Source: `wall_restricted_analysis.json`.",
        "",
        f"v_s={V_S}; 50^3 grid points per metric, with bounds from `run_analysis.py`. "
        f"The wall mask is {F_LOW} <= f <= {F_HIGH}. "
        "Type fractions use point counts within the stated region.",
        "",
        "| Metric | Grid points | Wall points | Full Type IV % | Wall I / II / III / IV % | Time (s) |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for name, r in results.items():
        full, wall = r["full_grid"], r["wall_restricted"]
        types = " / ".join(f"{100 * wall[f'frac_type_{t}']:.2f}" for t in ("i", "ii", "iii", "iv"))
        lines.append(
            f"| {name} | {full['n_total']} | {wall['n_total']} | {100 * full['frac_type_iv']:.2f} "
            f"| {types} | {r['elapsed_s']:.1f} |"
        )
    lines += [
        "",
        "A missed violation passes the Eulerian test but violates the condition for a searched observer. "
        "Full-grid percentages divide by all grid points; wall percentages divide "
        "by wall points with a detected violation of that condition. "
        "`N/A` means that denominator is zero. These denominators differ.",
        "",
        "| Metric | Region | NEC miss % | WEC miss % | SEC miss % | DEC miss % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        full, wall = r["full_grid"], r["wall_restricted"]
        full_miss = " | ".join(
            f"{full[f'{c}_pct_missed']:.2f}" for c in ("nec", "wec", "sec", "dec")
        )
        wall_miss = " | ".join(
            _fmt_miss(wall[f"{c}_miss_rate"]).removesuffix("%")
            for c in ("nec", "wec", "sec", "dec")
        )
        lines += [f"| {name} | full grid | {full_miss} |", f"| {name} | wall | {wall_miss} |"]
    for name, r in results.items():
        if r.get("caveat") == "unresolved_lower_bound":
            lines.append(
                f"\n{name}: the tested grid undersamples the wall; these statistics have no continuum error bound.\n"
            )
    lines.append("")
    outpath = os.path.join(RESULTS_DIR, "wall_restricted_report.md")
    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {outpath}")


# Output: Type I/II/III/IV breakdown figure


def plot_type_breakdown(results: dict) -> None:
    """Grouped stacked bars comparing full-grid vs wall-restricted Type fractions.

    For each metric, two bars are drawn side by side: the left bar shows
    full-grid Type I/II/III/IV fractions, the right bar shows the
    wall-restricted equivalents. Lentz bars are hatched to flag the
    unresolved-lower-bound caveat.
    """
    apply_style()

    fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL, 3.0))

    metric_names = WARP_METRICS
    n_metrics = len(metric_names)
    group_width = 0.8
    bar_width = group_width / 2.0
    x_centers = np.arange(n_metrics)

    # Type colors: reuse the accessibility palette (Type I, II, III, IV)
    type_colors = [COLORS[0], COLORS[2], COLORS[4], COLORS[1]]
    type_labels = ["Type I", "Type II", "Type III", "Type IV"]

    # Draw stacked bars per metric
    for i, name in enumerate(metric_names):
        r = results[name]
        full = r["full_grid"]
        wall = r["wall_restricted"]

        full_fracs = [
            full["frac_type_i"],
            full["frac_type_ii"],
            full["frac_type_iii"],
            full["frac_type_iv"],
        ]
        wall_fracs = [
            wall["frac_type_i"],
            wall["frac_type_ii"],
            wall["frac_type_iii"],
            wall["frac_type_iv"],
        ]

        x_full = x_centers[i] - bar_width / 2.0
        x_wall = x_centers[i] + bar_width / 2.0

        hatch = "//" if r.get("caveat") == "unresolved_lower_bound" else None

        bottom_full = 0.0
        bottom_wall = 0.0
        for j in range(4):
            # full-grid (thin edge, no hatch unless lentz)
            ax.bar(
                x_full,
                full_fracs[j],
                width=bar_width,
                bottom=bottom_full,
                color=type_colors[j],
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
                label=type_labels[j] if i == 0 else None,
            )
            bottom_full += full_fracs[j]
            # wall-restricted (same color, distinct edge)
            ax.bar(
                x_wall,
                wall_fracs[j],
                width=bar_width,
                bottom=bottom_wall,
                color=type_colors[j],
                edgecolor="black",
                linewidth=1.2,
                hatch=hatch,
            )
            bottom_wall += wall_fracs[j]

    # Ticks / labels
    ax.set_xticks(x_centers)
    ax.set_xticklabels(metric_names, rotation=0)
    ax.set_ylabel("Type fraction")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Full grid (thin edge) vs wall f in [{F_LOW}, {F_HIGH}] (thick edge)")

    # Legend: one entry per type, one entry explaining pairing
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=type_colors[j], edgecolor="black", label=type_labels[j]) for j in range(4)
    ]
    legend_handles.append(
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Lentz (unresolved)")
    )
    ax.legend(handles=legend_handles, loc="upper right", ncol=2, fontsize=8)

    fig.tight_layout()
    outpath = os.path.join(FIGURES_DIR, "wall_restricted_type_breakdown.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Figure saved to {outpath}")


# Main


def _instantiate(name: str):
    """Instantiate a warp metric with V_S (skip v_s kw for WarpShell defaults)."""
    cls, params = METRICS[name]
    full_params = dict(params)
    full_params["v_s"] = V_S
    return cls(**full_params)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    start_time = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=" * 70)
    print("Wall-Restricted Type-IV Analysis (all 4 warp metrics)")
    print(f"Started: {start_time}")
    print(f"Velocity: v_s = {V_S}")
    print(f"Wall region: shape function in [{F_LOW}, {F_HIGH}]")
    print("=" * 70)

    results: dict[str, dict] = {}
    for name in WARP_METRICS:
        metric = _instantiate(name)
        grid_spec = GRID_MAP[name]
        results[name] = analyze_metric(name, metric, grid_spec)

    save_json(results, start_time)
    save_report(results, start_time)
    plot_type_breakdown(results)

    # Console summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name in WARP_METRICS:
        r = results[name]
        wall = r["wall_restricted"]
        full = r["full_grid"]
        cav = f" [{r['caveat']}]" if r.get("caveat") else ""
        print(
            f"  {name:>11s}: wall_n_total={wall['n_total']:>6d}, "
            f"Type-IV full={full['frac_type_iv']:.2%}, "
            f"wall={wall['frac_type_iv']:.2%}, "
            f"DEC miss wall={_fmt_miss(wall['dec_miss_rate'])}" + cav
        )

    print("\nInvestigation complete.")


if __name__ == "__main__":
    main()
