"""Grid-level Eulerian vs observer-robust comparison.

Evaluates energy conditions across a spatial grid through the two library
routes and checks that they agree:

- ``compare_eulerian_vs_robust`` runs the rapidity-capped observer search
  (``verify_grid``) and the exact Eulerian margins independently, then
  reports where the Eulerian frame misses a violation.
- ``certify`` runs the frame-free certification on the same grid and reports
  wall-restricted, proper-volume-weighted type fractions and miss rates.

Both routes classify every point, so their Hawking-Ellis type counts must
match. Produces a 3-panel figure (Eulerian | Robust | Missed) saved to
``examples/output/alcubierre_grid_comparison.pdf`` for the condition the
Eulerian frame misses most. On the Alcubierre wall that is the SEC: the trace
line fails for a boosted observer while the Eulerian one reads it as
satisfied. NEC, WEC and DEC are violated in both frames at once, so their
missed panel would be empty.
"""

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np

from warpax.analysis import compare_eulerian_vs_robust
from warpax.benchmarks import AlcubierreMetric
from warpax.certify import certify
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.visualization import plot_comparison_panel

v_s = 0.5
metric = AlcubierreMetric(v_s=v_s, R=1.0, sigma=8.0)

grid = GridSpec(
    bounds=[(-3.0, 3.0), (-3.0, 3.0), (-0.5, 0.5)],
    shape=(24, 24, 4),
)
CONDITIONS = ("nec", "wec", "sec", "dec")

print("Alcubierre Grid Analysis (Eulerian vs Observer-Robust)")
print("=" * 60)
print(f"v_s = {v_s}, grid = {grid.shape}, bounds = {grid.bounds}")
print(f"Total grid points: {np.prod(grid.shape)}")

print("\n[1/4] Computing curvature tensors on grid...")
grid_result = evaluate_curvature_grid(metric, grid, batch_size=256)
print(f"  Max |Ricci scalar|: {np.max(np.abs(grid_result.ricci_scalar)):.4e}")
print(f"  Max |T_ab|: {np.max(np.abs(grid_result.stress_energy)):.4e}")

print("\n[2/4] Running Eulerian vs robust comparison (BFGS optimization)...")
comparison = compare_eulerian_vs_robust(
    grid_result.stress_energy,
    grid_result.metric,
    grid_result.metric_inv,
    grid_shape=grid.shape,
    n_starts=8,
    zeta_max=5.0,
    batch_size=64,
)

for cond in CONDITIONS:
    print(f"  {cond.upper()}:")
    print(f"    Eulerian min margin:      {float(np.min(comparison.eulerian_margins[cond])):.6e}")
    print(f"    Robust min margin:        {float(np.min(comparison.robust_margins[cond])):.6e}")
    print(f"    Points violated (robust): {comparison.pct_violated_robust[cond]:.1f}%")
    n_missed = int(np.sum(comparison.missed[cond]))
    print(f"    Points missed by Euler:   {n_missed} ({comparison.pct_missed[cond]:.1f}%)")
    print(f"    Conditional miss rate:    {comparison.conditional_miss_rate[cond]:.1f}%")

print("\n[3/4] Frame-free certification on the same grid (certify)...")
cert = certify(metric, bounds=list(grid.bounds), shape=grid.shape, clustered=False)
he_robust = np.asarray(comparison.he_types).ravel()
he_cert = np.asarray(cert.frame_free.he_types).ravel()
print(f"  {'Type':>6s} {'verify_grid':>12s} {'certify':>10s}")
for t in range(1, 5):
    n_robust, n_cert = int(np.sum(he_robust == t)), int(np.sum(he_cert == t))
    if n_robust or n_cert:
        print(f"  {t:>6d} {n_robust:>12d} {n_cert:>10d}")
assert np.array_equal(he_robust, he_cert), "Hawking-Ellis types differ between routes"
print(f"  Wall points (0.1 < f < 0.9): {cert.type_fractions['n_selected']}")
print(f"  Wall Type IV fraction (volume weighted): {cert.type_fractions['frac_type_iv']:.3f}")
print("  Wall Eulerian miss rates (volume weighted, expected non-zero only for SEC):")
for cond in CONDITIONS:
    miss = cert.single_frame_miss[cond]
    print(f"    {cond.upper()}: {100 * miss['miss_rate']:.1f}%  ({miss['n_missed']} points)")

print("\n[4/4] Figure")
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "alcubierre_grid_comparison.pdf")

plotted = max(CONDITIONS, key=lambda c: comparison.conditional_miss_rate[c])

plot_comparison_panel(
    eulerian_margin=np.asarray(comparison.eulerian_margins[plotted]),
    robust_margin=np.asarray(comparison.robust_margins[plotted]),
    missed=np.asarray(comparison.missed[plotted]),
    grid_bounds=grid.bounds,
    grid_shape=grid.shape,
    title=f"Alcubierre {plotted.upper()} (v_s = {v_s}): Eulerian vs Robust",
    save_path=save_path,
)

print(f"  Figure saved: {save_path}  (condition: {plotted.upper()})")
print(f"  Left:   Eulerian {plotted.upper()} margin, what a single-frame analysis sees")
print("  Center: robust margin, the worst case over every observer")
print(f"  Right:  the {int(np.sum(comparison.missed[plotted]))} points the Eulerian frame misses")
