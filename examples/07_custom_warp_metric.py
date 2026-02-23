
"""Custom warp drive metric design with robust EC validation.

Shows how a user can define their own warp manifold from scratch using
the ADMMetric interface, then run the full energy-condition verification
pipeline: single-point checks, grid-level Eulerian-vs-robust comparison,
and visualization of missed violations.

The example metric is a "Gaussian warp bubble" -- a minimal toy design
where the shift is a Gaussian envelope instead of the standard Alcubierre
top-hat.  The Gaussian wall has softer gradients, which lets us explore
how shape-function choice affects EC violation structure.

Key outputs:
- Single-point curvature and EC margins at the bubble wall
- Grid-level Eulerian vs robust comparison statistics
- 3-panel comparison figure saved to examples/output/
"""

import os

import jax.numpy as jnp
import numpy as np
import sympy as sp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from warpax.analysis import compare_eulerian_vs_robust
from warpax.energy_conditions import compute_eulerian_ec, verify_point
from warpax.geometry import (
    GridSpec,
    compute_curvature_chain,
    evaluate_curvature_grid,
    kretschner_scalar,
)
from warpax.geometry.metric import ADMMetric, SymbolicMetric
from warpax.visualization import plot_comparison_panel


# ============================================================================
# Step 1: Define a custom warp drive metric
# ============================================================================


class GaussianWarpMetric(ADMMetric):
    """Gaussian warp bubble -- a minimal custom warp metric.

    The shift uses a Gaussian profile instead of the Alcubierre top-hat:

        f(r_s) = exp(-r_s^2 / (2 * w^2))

    where w controls the bubble width.  Lapse is unity and the spatial
    metric is flat, so the geometry is entirely encoded in the shift.

    Parameters
    ----------
    v_s : float
        Bubble velocity (subluminal, 0 < v_s < 1).
    w : float
        Gaussian width parameter (controls bubble size).
    """

    v_s: float = 0.5
    w: float = 1.0

    @jaxtyped(typechecker=beartype)
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        return jnp.array(1.0)

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        t, x, y, z = coords
        dx = x - self.v_s * t
        r_s = jnp.sqrt(dx**2 + y**2 + z**2)

        # Gaussian shape function: f(r_s) = exp(-r_s^2 / (2*w^2))
        f = jnp.exp(-r_s**2 / (2.0 * self.w**2))

        # Shift only along travel direction (like Alcubierre)
        return jnp.array([-self.v_s * f, 0.0, 0.0])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        return jnp.eye(3)

    def symbolic(self) -> SymbolicMetric:
        t, x, y, z = sp.symbols("t x y z")
        v_s = sp.Symbol("v_s", positive=True)
        w = sp.Symbol("w", positive=True)

        dx = x - v_s * t
        r_s = sp.sqrt(dx**2 + y**2 + z**2)
        f = sp.exp(-r_s**2 / (2 * w**2))
        beta_x = -v_s * f

        g = sp.Matrix([
            [-(1 - beta_x**2), beta_x, 0, 0],
            [beta_x, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "GaussianWarp"


# ============================================================================
# Step 2: Single-point validation at the bubble wall
# ============================================================================

metric = GaussianWarpMetric(v_s=0.5, w=1.0)

# Probe at a point on the bubble wall (r_s ~ w, where gradients peak)
coords = jnp.array([0.0, 1.0, 0.5, 0.0])

print("Custom Gaussian Warp Bubble -- EC Validation")
print("=" * 55)
print(f"Parameters: v_s = {metric.v_s}, w = {metric.w}")

r_s = jnp.sqrt(coords[1] ** 2 + coords[2] ** 2 + coords[3] ** 2)
print(f"Probe point: (t,x,y,z) = ({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})")
print(f"Radial distance: r_s = {r_s:.4f}")

# Compute curvature chain via autodiff
result = compute_curvature_chain(metric, coords)
K = kretschner_scalar(result.riemann, result.metric, result.metric_inv)

print(f"\nRicci scalar:     {result.ricci_scalar:.6e}")
print(f"Kretschner scalar: {K:.6e}")
print(f"Max |T_ab|:        {jnp.max(jnp.abs(result.stress_energy)):.6e}")

# Observer-robust EC verification (all four conditions)
ec = verify_point(result.stress_energy, result.metric, result.metric_inv)

print(f"\nObserver-robust EC margins (negative = violated):")
print(f"  NEC: {ec.nec_margin:.6e}")
print(f"  WEC: {ec.wec_margin:.6e}")
print(f"  SEC: {ec.sec_margin:.6e}")
print(f"  DEC: {ec.dec_margin:.6e}")
print(f"  Hawking-Ellis type: {int(ec.he_type)}")
print(f"  Worst observer 4-velocity: {ec.worst_observer}")
print(f"  Worst observer params (zeta, theta, phi): {ec.worst_params}")

# Eulerian-frame EC for comparison
eulerian_ec = compute_eulerian_ec(result.stress_energy, result.metric, result.metric_inv)
print(f"\nEulerian-frame EC margins:")
print(f"  NEC: {eulerian_ec['nec']:.6e}")
print(f"  WEC: {eulerian_ec['wec']:.6e}")

gap = eulerian_ec["nec"] - float(ec.nec_margin)
if gap > 1e-10:
    print(f"\n  -> Eulerian NEC is {gap:.2e} less negative than robust NEC.")
    print("     A boosted observer sees a worse violation than the Eulerian one.")

# ============================================================================
# Step 3: Grid-level Eulerian vs robust comparison
# ============================================================================

print("\n" + "=" * 55)
print("Grid Analysis: Eulerian vs Observer-Robust")
print("=" * 55)

grid = GridSpec(
    bounds=[(-3.0, 3.0), (-3.0, 3.0), (-0.5, 0.5)],
    shape=(24, 24, 4),
)

print(f"Grid: {grid.shape}, bounds: {grid.bounds}")
print(f"Total points: {np.prod(grid.shape)}")

print("\n[1/3] Computing curvature tensors on grid...")
grid_result = evaluate_curvature_grid(metric, grid)
print(f"  Max |Ricci scalar|: {np.max(np.abs(grid_result.ricci_scalar)):.4e}")
print(f"  Max |T_ab|:         {np.max(np.abs(grid_result.stress_energy)):.4e}")

print("\n[2/3] Running Eulerian vs robust comparison...")
comparison = compare_eulerian_vs_robust(
    grid_result.stress_energy,
    grid_result.metric,
    grid_result.metric_inv,
    grid_shape=grid.shape,
    n_starts=8,
    zeta_max=5.0,
    batch_size=64,
)

print("\n[3/3] Results\n")

for cond in ("nec", "wec", "sec", "dec"):
    eul_min = float(np.min(comparison.eulerian_margins[cond]))
    rob_min = float(np.min(comparison.robust_margins[cond]))
    n_missed = int(np.sum(comparison.missed[cond]))
    pct_missed = comparison.pct_missed[cond]
    pct_violated = comparison.pct_violated_robust[cond]
    cond_miss = comparison.conditional_miss_rate[cond]

    print(f"  {cond.upper()}:")
    print(f"    Eulerian min margin:      {eul_min:.6e}")
    print(f"    Robust min margin:        {rob_min:.6e}")
    print(f"    Points violated (robust): {pct_violated:.1f}%")
    print(f"    Missed by Eulerian:       {n_missed} ({pct_missed:.1f}%)")
    print(f"    Conditional miss rate:    {cond_miss:.1f}%")
    print()

# Hawking-Ellis classification breakdown
he_types = np.asarray(comparison.he_types).ravel()
print("  Hawking-Ellis classification:")
for t in range(1, 5):
    count = int(np.sum(he_types == t))
    if count > 0:
        print(f"    Type {t}: {count} points ({100 * count / len(he_types):.1f}%)")

# ============================================================================
# Step 4: Save comparison figure
# ============================================================================

output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)
# Pick the condition with the highest miss rate for the figure
best_cond = max(("nec", "wec", "sec", "dec"), key=lambda c: comparison.pct_missed[c])
n_missed = int(np.sum(comparison.missed[best_cond]))

save_path = os.path.join(output_dir, "gaussian_warp_grid_comparison.pdf")

fig = plot_comparison_panel(
    eulerian_margin=np.asarray(comparison.eulerian_margins[best_cond]),
    robust_margin=np.asarray(comparison.robust_margins[best_cond]),
    missed=np.asarray(comparison.missed[best_cond]),
    grid_bounds=grid.bounds,
    grid_shape=grid.shape,
    title=f"Gaussian Warp {best_cond.upper()} (v_s = {metric.v_s}): Eulerian vs Robust",
    save_path=save_path,
)

print(f"\nFigure plots {best_cond.upper()} (highest miss rate: {n_missed} points)")
print(f"Figure saved: {save_path}")
print("\nThree panels:")
print(f"  Left:   Eulerian {best_cond.upper()} margin")
print(f"  Center: Robust {best_cond.upper()} margin (worst-case over all observers)")
print(f"  Right:  Missed violations (Eulerian misses, robust catches)")
