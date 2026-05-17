"""20x15 S-shell phase-diagram sweep.

Companion to the T-shell sweep produced by ``examples/10_phase_diagram.py``.
S-shell has zero shift, so a positive result here would indicate that the
boundary DEC failure is shift-driven; an empty admissible set indicates
the boundary failure is geometric (smooth transition curvature).
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import matplotlib
matplotlib.use("Agg")

import jax
jax.config.update("jax_enable_x64", True)

from warpax.optimization import sweep_transport
from warpax.visualization.phase_diagram import (
    plot_phase_diagram,
    plot_phase_summary,
)


def main():
    output_dir = (
        Path(__file__).resolve().parents[1] / "output" / "phase_diagram_sshell"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = str(output_dir / "sweep_results.npz")

    print("S-shell sweep: 20x15 grid, n_probes=15, n_ec_starts=8")
    result = sweep_transport(
        ansatz="sshell",
        compactness_range=(0.01, 0.20),
        thickness_range=(0.3, 0.8),
        n_compactness=20,
        n_thickness=15,
        R_2=20.0,
        n_density=4,
        n_velocity=4,
        n_grid=512,
        n_probes=15,
        n_ec_starts=8,
        progress=True,
        save_path=sweep_path,
    )

    n_feas = sum(1 for pt in result.points if pt.ec_feasible)
    margins = [pt.worst_ec_margin for pt in result.points]
    print(f"\nSweep complete: {n_feas}/{len(result.points)} EC-admissible")
    print(f"  worst-margin range: [{min(margins):.3e}, {max(margins):.3e}]")

    plot_phase_diagram(result, save_path=str(output_dir / "phase_diagram.pdf"))
    plot_phase_summary(result, save_path=str(output_dir / "phase_summary.pdf"))
    print(f"  -> {output_dir}")


if __name__ == "__main__":
    main()
