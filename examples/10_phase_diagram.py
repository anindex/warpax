"""Sampled energy-condition sweep over the prescribed T-shell profiles.

Sweeps (compactness, thickness) and draws the resulting diagnostic margins.
Passing those tests does not establish source closure or an elastic
equilibrium; example 11 solves the Einstein-elastic shell.

    python examples/10_phase_diagram.py          (8x6 demo, about 20 s)
    python examples/10_phase_diagram.py --full   (20x15, about a minute)

The GPU autotuner is disabled (``--xla_gpu_autotune_level=0``) because it
crashes on f64 reductions in the optimistix vmap. Grid points whose
construction fails carry NaN margins and are counted separately. When no
sampled case passes, the closest approach is reported as a sampled margin
without inferring a universal shell bound.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

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
        n_compactness, n_thickness, n_probes = 20, 15, 15
        print("FULL sweep (20x15)...")
    else:
        n_compactness, n_thickness, n_probes = 8, 6, 10
        print("DEMO sweep (8x6). Use --full for a larger sampled grid.")

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
        progress=True,
        save_path=sweep_path,
    )

    evaluated = [pt for pt in result.points if math.isfinite(pt.worst_ec_margin)]
    n_failed = len(result.points) - len(evaluated)
    admissible = [pt for pt in evaluated if pt.ec_feasible]
    print(
        f"\nSweep complete: {len(admissible)}/{len(result.points)} pass the sampled EC tests"
        f" ({n_failed} grid points failed to build and carry NaN)"
    )

    if admissible:
        best = max(admissible, key=lambda pt: pt.transport)
        print(
            f"  Optimum: C={best.compactness:.3f}, "
            f"dR/R={best.thickness_ratio:.3f}, "
            f"|beta^x|={best.transport:.5f}, "
            f"M={best.mass:.2f}"
        )
    elif evaluated:
        closest = max(evaluated, key=lambda pt: pt.worst_ec_margin)
        print("  No sampled case passed. Closest approach:")
        print(
            f"    C={closest.compactness:.3f}, "
            f"dR/R={closest.thickness_ratio:.3f}, "
            f"worst EC margin={closest.worst_ec_margin:+.3e}, "
            f"|beta^x|={closest.transport:.5f}"
        )

    plot_phase_diagram(result, save_path=str(output_dir / "phase_diagram.pdf"))
    print(f"  -> {output_dir / 'phase_diagram.pdf'}")

    plot_phase_summary(result, save_path=str(output_dir / "phase_summary.pdf"))
    print(f"  -> {output_dir / 'phase_summary.pdf'}")


if __name__ == "__main__":
    main()
