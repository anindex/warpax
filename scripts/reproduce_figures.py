
"""Reproduce all paper figures from cached results.

Single-command figure generator that loads cached .npz/.json results from
results/ and generates PDF figures in figures/.  No computation is performed
-- this script only reads cached data and renders plots.

Usage
-----
Generate ALL figures:
    python scripts/reproduce_figures.py

Selective regeneration:
    python scripts/reproduce_figures.py --only comparison
    python scripts/reproduce_figures.py --only velocity_convergence observer

Available figure sets: comparison, velocity_convergence, velocity, observer,
    convergence, kinematic, missed, geodesic, alignment, c1_vs_c2,
    rodal_dec_ablation, fibonacci_dec
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

# Non-interactive backend (before any other matplotlib import)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from warpax.visualization._style import apply_style, SINGLE_COL, DOUBLE_COL

apply_style()

from warpax.geometry.types import GridSpec
from warpax.visualization.comparison_plots import (
    plot_comparison_panel,
    plot_comparison_table,
    plot_velocity_sweep,
)
from warpax.visualization.convergence_plots import (
    plot_convergence,
    plot_convergence_table,
)
from warpax.visualization.kinematic_plots import plot_kinematic_scalars
from warpax.visualization.direction_fields import plot_worst_observer_field
from warpax.visualization.geodesic_plots import (
    plot_blueshift_profile,
    plot_tidal_evolution,
)
from warpax.visualization.alignment_plots import plot_alignment_histogram

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIGURES_DIR = "figures/"
RESULTS_DIR = "results/"

# Core 4 metrics for main paper figures
CORE_METRICS = ["alcubierre", "lentz", "warpshell", "rodal"]

# All 6 warp metrics for kinematic figures
ALL_WARP_METRICS = ["alcubierre", "rodal", "vdb", "natario", "lentz", "warpshell"]

# Velocity sweep values for the missed-violations figure
MISSED_VELOCITIES = [0.1, 0.5, 0.9, 0.99]


def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _load_npz(path: str) -> np.lib.npyio.NpzFile | None:
    """Load .npz file, returning None with warning if missing."""
    if not os.path.exists(path):
        warnings.warn(f"Skipping: {path} not found")
        return None
    return np.load(path)


# ---------------------------------------------------------------------------
# Figure generation functions
# ---------------------------------------------------------------------------


def generate_comparison_figures(figures_dir: str, results_dir: str) -> int:
    """Generate Eulerian vs robust comparison panels for core metrics.

    For each core metric at v_s=0.5: NEC and WEC comparison panels.
    Also generates the comparison table from comparison_table.json.

    Returns the number of figures generated.
    """
    count = 0
    _ensure_dir(figures_dir)

    # Per-metric comparison panels (NEC and WEC)
    for name in CORE_METRICS:
        npz_path = os.path.join(results_dir, f"{name}_vs0.5.npz")
        data = _load_npz(npz_path)
        if data is None:
            continue

        grid_bounds = [tuple(b) for b in data["grid_bounds"]]
        grid_shape = tuple(data["grid_shape"])

        for cond in ("nec", "wec"):
            eul_key = f"{cond}_eulerian"
            rob_key = f"{cond}_robust"
            miss_key = f"{cond}_missed"

            if eul_key not in data or rob_key not in data or miss_key not in data:
                warnings.warn(f"Skipping {name} {cond}: missing keys in .npz")
                continue

            save_path = os.path.join(figures_dir, f"{name}_{cond}_comparison.pdf")
            plot_comparison_panel(
                eulerian_margin=data[eul_key],
                robust_margin=data[rob_key],
                missed=data[miss_key],
                grid_bounds=grid_bounds,
                grid_shape=grid_shape,
                title=(
                    rf"{name.capitalize()} {cond.upper()}: Eulerian vs Robust"
                    rf" ($v_s = 0.5$, ${grid_shape[0]}^3$ grid)"
                ),
                save_path=save_path,
            )
            print(f"  Generated: {save_path}")
            count += 1

    # Comparison table
    table_path = os.path.join(results_dir, "comparison_table.json")
    if os.path.exists(table_path):
        save_path = os.path.join(figures_dir, "comparison_table.pdf")
        plot_comparison_table(table_path, save_path=save_path)
        print(f"  Generated: {save_path}")
        count += 1
    else:
        warnings.warn(f"Skipping comparison table: {table_path} not found")

    return count


def generate_velocity_sweep_figures(figures_dir: str, results_dir: str) -> int:
    """Generate velocity sweep line plots for Alcubierre NEC and WEC.

    Returns the number of figures generated.
    """
    count = 0
    _ensure_dir(figures_dir)

    for cond in ("nec", "wec"):
        save_path = os.path.join(figures_dir, f"alcubierre_velocity_sweep_{cond}.pdf")
        plot_velocity_sweep(
            results_dir, "alcubierre", condition=cond, save_path=save_path,
        )
        print(f"  Generated: {save_path}")
        count += 1

    return count


def generate_worst_observer_figures(figures_dir: str, results_dir: str) -> int:
    """Generate worst-observer boost direction quiver plots.

    For each core metric at v_s=0.5: loads worst_params from cached .npz
    and passes to plot_worst_observer_field.

    Returns the number of figures generated.
    """
    count = 0
    _ensure_dir(figures_dir)

    for name in CORE_METRICS:
        npz_path = os.path.join(results_dir, f"{name}_vs0.5.npz")
        data = _load_npz(npz_path)
        if data is None:
            continue

        if "worst_params" not in data:
            warnings.warn(f"Skipping {name} observer: worst_params not in .npz")
            continue

        if "grid_bounds" not in data or "grid_shape" not in data:
            warnings.warn(f"Skipping {name} observer: missing grid metadata")
            continue

        worst_params = data["worst_params"]
        grid_bounds = [tuple(b) for b in data["grid_bounds"]]
        grid_shape = tuple(data["grid_shape"])
        grid = GridSpec(bounds=grid_bounds, shape=grid_shape)

        # Build significance mask: where |robust - eulerian| is meaningful
        # Use 5% threshold to suppress noisy arrows in near-flat regions
        significance_mask = None
        for cond in ("nec", "wec"):
            eul_key, rob_key = f"{cond}_eulerian", f"{cond}_robust"
            if eul_key in data and rob_key in data:
                diff = np.abs(data[rob_key] - data[eul_key])
                max_diff = float(np.nanmax(diff))
                if max_diff > 0:
                    cond_mask = diff > 0.05 * max_diff
                    if significance_mask is None:
                        significance_mask = cond_mask
                    else:
                        significance_mask = significance_mask | cond_mask

        fig, ax = plt.subplots(
            1, 1, figsize=(0.7 * DOUBLE_COL, 0.7 * DOUBLE_COL * 0.8),
        )
        plot_worst_observer_field(
            worst_params,
            grid,
            slice_axis=2,
            title=rf"{name.capitalize()} Worst-Observer Boost ($v_s = 0.5$, $50^3$ grid)",
            ax=ax,
            significance_mask=significance_mask,
        )

        save_path = os.path.join(figures_dir, f"{name}_worst_observer.pdf")
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Generated: {save_path}")
        count += 1

    return count


def generate_convergence_figures(figures_dir: str, results_dir: str) -> int:
    """Generate Richardson convergence log-log plot and table.

    Returns the number of figures generated.
    """
    count = 0
    _ensure_dir(figures_dir)

    json_path = os.path.join(results_dir, "convergence_data.json")
    if not os.path.exists(json_path):
        warnings.warn(f"Skipping convergence figures: {json_path} not found")
        return 0

    # Log-log convergence plot (NEC)
    save_path = os.path.join(figures_dir, "convergence_nec.pdf")
    plot_convergence(json_path, quantity="min_margin_nec", save_path=save_path)
    print(f"  Generated: {save_path}")
    count += 1

    # Convergence table
    save_path = os.path.join(figures_dir, "convergence_table.pdf")
    plot_convergence_table(json_path, save_path=save_path)
    print(f"  Generated: {save_path}")
    count += 1

    return count


def generate_kinematic_figures(figures_dir: str, results_dir: str) -> int:
    """Generate kinematic scalar (expansion/shear/vorticity) figures.

    Returns the number of figures generated.
    """
    count = 0
    _ensure_dir(figures_dir)

    for name in ALL_WARP_METRICS:
        npz_path = os.path.join(results_dir, f"{name}_kinematic_vs0.5.npz")
        data = _load_npz(npz_path)
        if data is None:
            continue

        theta = data["theta"]
        sigma_sq = data["sigma_sq"]
        omega_sq = data["omega_sq"]
        grid_bounds = [tuple(b) for b in data["grid_bounds"]]
        grid_shape = tuple(data["grid_shape"])

        save_path = os.path.join(figures_dir, f"{name}_kinematic.pdf")
        plot_kinematic_scalars(
            theta, sigma_sq, omega_sq,
            grid_bounds=grid_bounds,
            grid_shape=grid_shape,
            title=rf"{name.capitalize()} Kinematic Scalars ($v_s = 0.5$, ${grid_shape[0]}^3$ grid)",
            save_path=save_path,
        )
        print(f"  Generated: {save_path}")
        count += 1

    return count


def generate_missed_violations_figure(figures_dir: str, results_dir: str) -> int:
    """Generate the 'money shot': missed WEC violations vs velocity for Rodal.

    2x2 grid showing missed WEC violations for Rodal at
    v_s = 0.1, 0.5, 0.9, 0.99.  Rodal shows ~15% WEC missed by Eulerian
    analysis, making it the most visually striking demonstration of
    observer-dependent violations.

    Returns the number of figures generated.
    """
    _ensure_dir(figures_dir)

    # Use Rodal WEC - the strongest demonstration of observer-dependent misses
    metric_name = "rodal"
    condition = "wec"

    available_data = {}
    for v_s in MISSED_VELOCITIES:
        npz_path = os.path.join(results_dir, f"{metric_name}_vs{v_s}.npz")
        data = _load_npz(npz_path)
        if data is not None and f"{condition}_missed" in data:
            available_data[v_s] = data

    if not available_data:
        warnings.warn(f"Skipping missed-violations figure: no {metric_name} results found")
        return 0

    n_panels = len(available_data)
    if n_panels <= 2:
        nrows, ncols = 1, n_panels
    else:
        nrows, ncols = 2, 2

    # GridSpec: 2×2 panels + 1 shared colorbar column
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.48 * nrows))
    gs = fig.add_gridspec(nrows, ncols + 1,
                          width_ratios=[1] * ncols + [0.03],
                          wspace=0.08, hspace=0.3,
                          top=0.88, bottom=0.08)
    axes_flat = []
    for r in range(nrows):
        for c in range(ncols):
            axes_flat.append(fig.add_subplot(gs[r, c]))
    cax = fig.add_subplot(gs[:, ncols])

    last_im = None
    for idx, (v_s, data) in enumerate(sorted(available_data.items())):
        if idx >= len(axes_flat):
            break

        ax = axes_flat[idx]
        missed_3d = data[f"{condition}_missed"]
        grid_bounds = [tuple(b) for b in data["grid_bounds"]]
        grid_shape = tuple(data["grid_shape"])

        mid_z = grid_shape[2] // 2
        missed_2d = missed_3d[:, :, mid_z].astype(float)

        x_ax = np.linspace(grid_bounds[0][0], grid_bounds[0][1], grid_shape[0])
        y_ax = np.linspace(grid_bounds[1][0], grid_bounds[1][1], grid_shape[1])

        last_im = ax.pcolormesh(
            x_ax, y_ax, missed_2d.T,
            cmap="Reds", vmin=0, vmax=1, shading="auto",
        )
        pct_missed = float(np.mean(missed_3d.astype(float))) * 100.0
        pct_slice = float(np.mean(missed_2d)) * 100.0
        ax.set_title(
            rf"$v_s = {v_s}$ ({pct_missed:.1f}\% vol)",
            fontsize=8,
        )
        # y-label on leftmost column only, x-label on bottom row only
        row_i, col_i = divmod(idx, ncols)
        if col_i == 0:
            ax.set_ylabel("y")
        else:
            ax.tick_params(labelleft=False)
        if row_i == nrows - 1:
            ax.set_xlabel("x")
        else:
            ax.tick_params(labelbottom=False)

    for idx in range(len(available_data), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared colorbar
    if last_im is not None:
        fig.colorbar(last_im, cax=cax, label="Missed fraction")

    fig.suptitle(
        rf"Rodal {condition.upper()} Missed Violations ($50^3$, $R = 100$, $\sigma = 0.03$)",
        fontsize=9,
    )

    save_path = os.path.join(figures_dir, "missed_violations_vs_velocity.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Generated: {save_path}")
    return 1


def generate_type_breakdown_table(figures_dir: str, results_dir: str) -> int:
    """Generate Hawking-Ellis type breakdown table across all metrics.

    Shows % Type I per metric and max |Im lambda|/scale.

    Returns the number of figures generated.
    """
    _ensure_dir(figures_dir)

    rows = []
    for name in ALL_WARP_METRICS + ["schwarzschild"]:
        # Try v_s=0.5 first, then v_s=0.0 for Schwarzschild
        npz_path = os.path.join(results_dir, f"{name}_vs0.5.npz")
        if not os.path.exists(npz_path):
            npz_path = os.path.join(results_dir, f"{name}_vs0.0.npz")
        data = _load_npz(npz_path)
        if data is None:
            continue

        n_type_i = int(data["n_type_i"]) if "n_type_i" in data else -1
        n_type_iv = int(data["n_type_iv"]) if "n_type_iv" in data else -1
        max_imag = float(data["max_imag_eigenvalue"]) if "max_imag_eigenvalue" in data else -1.0
        grid_shape = tuple(data["grid_shape"])
        n_total = int(np.prod(grid_shape))

        pct_type_i = n_type_i / n_total * 100.0 if n_type_i >= 0 else -1.0

        rows.append({
            "metric": name,
            "n_total": n_total,
            "n_type_i": n_type_i,
            "pct_type_i": pct_type_i,
            "n_type_iv": n_type_iv,
            "max_imag": max_imag,
        })

    if not rows:
        warnings.warn("No data for type breakdown table")
        return 0

    # Render as matplotlib table
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.7, 0.4 + 0.25 * len(rows)))
    ax.axis("off")

    col_labels = ["Metric", "Grid", "% Type I", "Non-Type I", r"max $|{\rm Im}\lambda|$"]
    cell_text = []
    for r in rows:
        cell_text.append([
            r["metric"].capitalize(),
            f"{r['n_total']:,}",
            f"{r['pct_type_i']:.1f}%" if r["pct_type_i"] >= 0 else "N/A",
            str(r["n_type_iv"]) if r["n_type_iv"] >= 0 else "N/A",
            f"{r['max_imag']:.2e}" if r["max_imag"] >= 0 else "N/A",
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    ax.set_title("Hawking-Ellis Type Breakdown", fontsize=9, pad=10)

    save_path = os.path.join(figures_dir, "type_breakdown_table.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Generated: {save_path}")
    return 1


def generate_geodesic_figures(figures_dir: str, results_dir: str) -> int:
    """Generate geodesic tidal force and blueshift figures.

    Loads cached .npz results from run_geodesics.py and generates:
    - Tidal eigenvalue evolution along timelike geodesic (1 figure)
    - Blueshift profile along null geodesic (1 figure)

    Returns the number of figures generated.
    """
    count = 0
    _ensure_dir(figures_dir)

    # Tidal force figure try versioned filename first, then legacy
    tidal_path = os.path.join(results_dir, "geodesic_alcubierre_tidal_vs0.5.npz")
    if not os.path.exists(tidal_path):
        tidal_path = os.path.join(results_dir, "geodesic_alcubierre_tidal.npz")
    data = _load_npz(tidal_path)
    if data is not None:
        tidal_eigenvalues = data["tidal_eigenvalues"]
        proper_times = data["proper_times"]
        save_path = os.path.join(figures_dir, "alcubierre_tidal_forces.pdf")
        plot_tidal_evolution(
            tidal_eigenvalues,
            proper_times,
            title=r"Alcubierre Tidal Eigenvalues ($v_s = 0.5$, $R = 1$, $\sigma = 8$)",
            save_path=save_path,
        )
        print(f"  Generated: {save_path}")
        count += 1

    # Blueshift figure try versioned filename first, then legacy
    bs_path = os.path.join(results_dir, "geodesic_alcubierre_blueshift_vs0.5.npz")
    if not os.path.exists(bs_path):
        bs_path = os.path.join(results_dir, "geodesic_alcubierre_blueshift.npz")
    data = _load_npz(bs_path)
    if data is not None:
        blueshift = data["blueshift"]
        positions_x = data["positions_x"]
        save_path = os.path.join(figures_dir, "alcubierre_blueshift.pdf")
        plot_blueshift_profile(
            blueshift,
            positions_x,
            title=r"Alcubierre Blueshift ($v_s = 0.5$, $R = 1$, $\sigma = 8$)",
            save_path=save_path,
            bubble_radius=float(data.get("R", 1.0)),
            bubble_sigma=float(data.get("sigma", 8.0)),
        )
        print(f"  Generated: {save_path}")
        count += 1

    return count


def generate_alignment_figures(figures_dir: str, results_dir: str) -> int:
    """Generate worst-observer alignment angle histograms.

    Returns the number of figures generated.
    """
    _ensure_dir(figures_dir)

    npz_path = os.path.join(results_dir, "alignment_rodal.npz")
    data = _load_npz(npz_path)
    if data is None:
        return 0

    velocities = data["velocities"]
    angle_arrays: dict[float, np.ndarray] = {}
    for v_s in velocities:
        key = f"angles_vs{v_s}"
        if key in data:
            angle_arrays[float(v_s)] = np.asarray(data[key])

    if not angle_arrays:
        warnings.warn("alignment_rodal.npz has no angle arrays")
        return 0

    fig = plot_alignment_histogram(angle_arrays)
    save_path = os.path.join(figures_dir, "worst_observer_alignment.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Generated: {save_path}")
    return 1


# ---------------------------------------------------------------------------
# Standalone-script figure wrappers
# ---------------------------------------------------------------------------
# These figures require computation (not just cached data).  The wrapper
# functions call the standalone scripts via subprocess so that
# ``reproduce_figures.py`` remains a single entry-point for ALL paper figures.


def _run_standalone_script(script_name, figures_dir, *extra_args):
    """Run a standalone script and return 1 if its expected figure was produced."""
    import subprocess

    script = os.path.join(os.path.dirname(__file__), script_name)
    if not os.path.isfile(script):
        print(f"  WARNING: {script} not found")
        return 0
    cmd = [sys.executable, script, *extra_args]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-300:]
        print(f"  WARNING: {script_name} exited with code {result.returncode}")
        if stderr_tail:
            print(f"  stderr: {stderr_tail}")
        return 0
    return 1


def generate_c1_vs_c2_figure(figures_dir, results_dir):
    """Generate C1 vs C2 WarpShell comparison figure."""
    expected = os.path.join(figures_dir, "c1_vs_c2_margins.pdf")
    _run_standalone_script("run_c1_vs_c2_comparison.py", figures_dir)
    if os.path.isfile(expected):
        print(f"  Generated: {expected}")
        return 1
    print(f"  WARNING: {expected} not produced")
    return 0


def generate_rodal_dec_ablation_figure(figures_dir, results_dir):
    """Generate Rodal DEC ablation study figure."""
    expected = os.path.join(figures_dir, "rodal_dec_ablation.pdf")
    _run_standalone_script("rodal_dec_ablation.py", figures_dir)
    if os.path.isfile(expected):
        print(f"  Generated: {expected}")
        return 1
    print(f"  WARNING: {expected} not produced")
    return 0


def generate_fibonacci_dec_figure(figures_dir, results_dir):
    """Generate Fibonacci vs BFGS DEC sampling figure."""
    expected = os.path.join(figures_dir, "fibonacci_vs_bfgs_dec.pdf")
    _run_standalone_script(
        "run_sampling_comparison.py", figures_dir, "--fibonacci-dec",
    )
    if os.path.isfile(expected):
        print(f"  Generated: {expected}")
        return 1
    print(f"  WARNING: {expected} not produced")
    return 0


def generate_merged_velocity_convergence(figures_dir: str, results_dir: str) -> int:
    """Generate merged velocity sweep + convergence figure (two panels).

    Panel (a): Velocity sweep of min NEC margin (Alcubierre).
    Panel (b): Richardson convergence log-log plot.

    Returns the number of figures generated.
    """
    _ensure_dir(figures_dir)

    json_path = os.path.join(results_dir, "convergence_data.json")
    has_convergence = os.path.exists(json_path)
    has_velocity = any(
        os.path.exists(os.path.join(results_dir, f"alcubierre_vs{v}.npz"))
        for v in [0.1, 0.5, 0.9, 0.99]
    )

    if not has_velocity and not has_convergence:
        warnings.warn("Skipping merged velocity+convergence: no data found")
        return 0

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.4),
    )

    # Panel (a): Velocity sweep
    plot_velocity_sweep(results_dir, "alcubierre", condition="nec", ax=ax1)
    ax1.set_title("(a) Min NEC margin vs velocity", fontsize=9)

    # Panel (b): Convergence
    plot_convergence(json_path, quantity="min_margin_nec", ax=ax2)
    ax2.set_title(r"(b) Convergence ($25^3$/$50^3$/$100^3$)", fontsize=9)

    fig.tight_layout(pad=1.0)
    save_path = os.path.join(figures_dir, "velocity_convergence_merged.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Generated: {save_path}")
    return 1


# ---------------------------------------------------------------------------
# Figure set registry
# ---------------------------------------------------------------------------

FIGURE_SETS = {
    "comparison": generate_comparison_figures,
    "velocity_convergence": generate_merged_velocity_convergence,
    "velocity": generate_velocity_sweep_figures,
    "observer": generate_worst_observer_figures,
    "convergence": generate_convergence_figures,
    "kinematic": generate_kinematic_figures,
    "missed": generate_missed_violations_figure,
    "geodesic": generate_geodesic_figures,
    "type_breakdown": generate_type_breakdown_table,
    "alignment": generate_alignment_figures,
    "c1_vs_c2": generate_c1_vs_c2_figure,
    "rodal_dec_ablation": generate_rodal_dec_ablation_figure,
    "fibonacci_dec": generate_fibonacci_dec_figure,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all paper figures from cached results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(FIGURE_SETS.keys()),
        default=None,
        help=(
            "Generate only specific figure sets. "
            f"Available: {', '.join(FIGURE_SETS.keys())}. "
            "Default: generate all."
        ),
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=FIGURES_DIR,
        help=f"Output directory for figures (default: {FIGURES_DIR}).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help=f"Input directory for cached results (default: {RESULTS_DIR}).",
    )
    args = parser.parse_args()

    figures_dir = args.figures_dir
    results_dir = args.results_dir
    _ensure_dir(figures_dir)

    if not os.path.isdir(results_dir):
        print(f"WARNING: Results directory '{results_dir}' does not exist.")
        print("Run the analysis scripts first to generate cached results.")
        print("  python scripts/run_analysis.py")
        print("  python scripts/run_convergence.py")
        print("  python scripts/run_kinematic_scalars.py")
        print("  python scripts/run_geodesics.py")
        return

    sets_to_run = args.only if args.only else list(FIGURE_SETS.keys())
    total_count = 0

    for name in sets_to_run:
        gen_fn = FIGURE_SETS[name]
        print(f"\n{'=' * 50}")
        print(f"Generating: {name} figures")
        print(f"{'=' * 50}")
        count = gen_fn(figures_dir, results_dir)
        total_count += count
        if count == 0:
            print(f"  (no figures generated cached results may be missing)")

    print(f"\nDone: {total_count} PDF figures generated in {figures_dir}")


if __name__ == "__main__":
    main()
