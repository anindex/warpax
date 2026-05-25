"""Example 10: EC-admissible transport over shell design space.

Sweeps (compactness, thickness) for T-shell and generates
a publication-quality phase diagram and 2x2 summary figure.

    .venv/bin/python examples/10_phase_diagram.py          # 8x6 demo (~5 min)
    .venv/bin/python examples/10_phase_diagram.py --full    # 20x15 (~1 hour)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Disable GPU autotuner (crashes on f64 reductions in optimistix vmap).
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import matplotlib
matplotlib.use("Agg")

import jax

jax.config.update("jax_enable_x64", True)

from warpax.optimization import sweep_transport
from warpax.visualization.phase_diagram import plot_phase_diagram, plot_phase_summary


def main():
    output_dir = Path(__file__).resolve().parents[1] / "results" / "phase_diagram"
    output_dir.mkdir(parents=True, exist_ok=True)

    if "--full" in sys.argv:
        n_compactness = 20
        n_thickness = 15
        n_probes = 15
        n_ec_starts = 8
        print("FULL sweep (20x15)...")
    else:
        n_compactness = 8
        n_thickness = 6
        n_probes = 10
        n_ec_starts = 4
        print("DEMO sweep (8x6). Use --full for paper quality.")

    sweep_path = str(output_dir / "sweep_results.npz")
    result = sweep_transport(
        ansatz="tshell",
        compactness_range=(0.01, 0.20),
        thickness_range=(0.3, 0.8),
        n_compactness=n_compactness,
        n_thickness=n_thickness,
        R_2=20.0,
        n_density=4,
        n_velocity=4,
        n_grid=512,
        n_probes=n_probes,
        n_ec_starts=n_ec_starts,
        progress=True,
        save_path=sweep_path,
    )

    n_feasible = sum(1 for pt in result.points if pt.ec_feasible)
    n_total = len(result.points)
    print(f"\nSweep complete: {n_feasible}/{n_total} EC-admissible")

    admissible = [pt for pt in result.points if pt.ec_feasible]
    if admissible:
        best = max(admissible, key=lambda pt: pt.transport)
        print(f"  Optimum: C={best.compactness:.3f}, "
              f"dR/R={best.thickness_ratio:.3f}, "
              f"|beta^x|={best.transport:.5f}, "
              f"M={best.mass:.2f}")
    else:
        print("  No EC-admissible points found.")

    plot_phase_diagram(
        result,
        save_path=str(output_dir / "phase_diagram.pdf"),
    )
    print(f"  -> {output_dir / 'phase_diagram.pdf'}")

    plot_phase_summary(
        result,
        save_path=str(output_dir / "phase_summary.pdf"),
    )
    print(f"  -> {output_dir / 'phase_summary.pdf'}")


if __name__ == "__main__":
    main()
