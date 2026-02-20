
"""Grid-level Eulerian vs observer-robust comparison.

Demonstrates warpax's core workflow: evaluating energy conditions across a
spatial grid and revealing where Eulerian analysis misses violations that
boosted observers can detect.

Produces:
- A 3-panel comparison figure (Eulerian | Robust | Missed) saved to
  examples/output/alcubierre_grid_comparison.pdf
- Summary statistics showing the fraction of grid points where Eulerian
  analysis fails to detect energy condition violations

This is the key result of the paper: observer-robust analysis is essential
because Eulerian-only analysis can miss a significant fraction of violations.
"""

import os

import jax.numpy as jnp
import numpy as np

from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.analysis import compare_eulerian_vs_robust
from warpax.visualization import plot_comparison_panel

# Setup
v_s = 0.5
metric = AlcubierreMetric(v_s=v_s, R=1.0, sigma=8.0)

# 24×24×4 grid centered on the bubble (small for fast execution)
grid = GridSpec(
    bounds=[(-3.0, 3.0), (-3.0, 3.0), (-0.5, 0.5)],
    shape=(24, 24, 4),
)

print("Alcubierre Grid Analysis (Eulerian vs Observer-Robust)")
print("=" * 60)
print(f"v_s = {v_s}, grid = {grid.shape}, bounds = {grid.bounds}")
print(f"Total grid points: {np.prod(grid.shape)}")

# Step 1: Compute curvature tensors on the grid
print("\n[1/3] Computing curvature tensors on grid...")
grid_result = evaluate_curvature_grid(metric, grid)
print(f"  Max |Ricci scalar|: {np.max(np.abs(grid_result.ricci_scalar)):.4e}")
print(f"  Max |T_ab|: {np.max(np.abs(grid_result.stress_energy)):.4e}")

# Step 2: Eulerian vs robust EC comparison
print("\n[2/3] Running Eulerian vs robust comparison (BFGS optimization)...")

T_field = grid_result.stress_energy    # (*grid_shape, 4, 4)
g_field = grid_result.metric           # (*grid_shape, 4, 4)
g_inv_field = grid_result.metric_inv   # (*grid_shape, 4, 4)

comparison = compare_eulerian_vs_robust(
    T_field, g_field, g_inv_field,
    grid_shape=grid.shape,
    n_starts=8,      # reduced for speed in example
    zeta_max=5.0,
    batch_size=64,
)

# Step 3: Print statistics
print("\n[3/3] Results\n")

for cond in ("nec", "wec"):
    eul_min = float(np.min(comparison.eulerian_margins[cond]))
    rob_min = float(np.min(comparison.robust_margins[cond]))
    n_missed = int(np.sum(comparison.missed[cond]))
    pct_missed = comparison.pct_missed[cond]
    pct_violated = comparison.pct_violated_robust[cond]
    cond_miss = comparison.conditional_miss_rate[cond]

    print(f"  {cond.upper()}:")
    print(f"    Eulerian min margin:    {eul_min:.6e}")
    print(f"    Robust min margin:      {rob_min:.6e}")
    print(f"    Points violated (robust): {pct_violated:.1f}%")
    print(f"    Points missed by Euler:   {n_missed} ({pct_missed:.1f}%)")
    print(f"    Conditional miss rate:    {cond_miss:.1f}%")
    print()

# Hawking–Ellis classification breakdown
stats = comparison.classification_stats
he_types = np.asarray(comparison.he_types).ravel()
print(f"  Hawking–Ellis classification:")
for t in range(1, 5):
    count = int(np.sum(he_types == t))
    if count > 0:
        print(f"    Type {t}: {count} points ({100*count/len(he_types):.1f}%)")

# Step 4: Generate comparison figure
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "alcubierre_grid_comparison.pdf")

fig = plot_comparison_panel(
    eulerian_margin=np.asarray(comparison.eulerian_margins["nec"]),
    robust_margin=np.asarray(comparison.robust_margins["nec"]),
    missed=np.asarray(comparison.missed["nec"]),
    grid_bounds=grid.bounds,
    grid_shape=grid.shape,
    title=f"Alcubierre NEC (v_s = {v_s}): Eulerian vs Robust",
    save_path=save_path,
)

print(f"\n  Figure saved: {save_path}")
print("\nDone! The figure shows three panels:")
print("  Left:   Eulerian NEC margin (what standard analysis sees)")
print("  Center: Robust NEC margin (worst-case over all observers)")
print("  Right:  Missed violations (red = Eulerian misses, robust catches)")
