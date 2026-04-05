"""Custom warp drive metric with full EC pipeline + wall-restricted diagnostics.

This is the canonical custom-metric workflow. A user defines a new warp
spacetime by subclassing :class:`ADMMetric` (implementing ``lapse``,
``shift``, ``spatial_metric``, ``symbolic``, ``name`` and the 
abstract method ``shape_function_value``), then runs:

    1. Single-point curvature + EC verification at the bubble wall
    2. Grid-level Eulerian vs robust comparison
       (:func:`compare_eulerian_vs_robust`)
    3. Wall-restricted statistics via :func:`shape_function_mask` +
       :func:`compute_wall_restricted_stats`
    4. Figure output (3-panel comparison)

The example metric is a "Gaussian warp bubble" -- the shift uses a
Gaussian envelope rather than the Alcubierre tanh top-hat. Runtime target:
under 30 seconds on a laptop CPU (float64, no GPU). To scale up, increase
``grid_n`` from 20 to 50+ at the cost of minutes rather than seconds.

Outputs
-------
- Console: single-point margins, grid summary statistics, wall-restricted stats
- examples/output/gaussian_warp_comparison.pdf: 3-panel NEC comparison figure

Usage
-----
    python examples/07_custom_warp_metric.py
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

# Defensively ensure float64 for standalone script execution
# (warpax/__init__.py already enables this; keep the redundant call so
# the script also works when run standalone with a stale import order.)
jax.config.update("jax_enable_x64", True)

from warpax.analysis import compare_eulerian_vs_robust  # noqa: E402
from warpax.energy_conditions import (  # noqa: E402
    compute_eulerian_ec,
    compute_wall_restricted_stats,
    shape_function_mask,
    verify_grid,
    verify_point,
)
from warpax.geometry import (  # noqa: E402
    GridSpec,
    build_coord_batch,
    compute_curvature_chain,
    evaluate_curvature_grid,
    kretschner_scalar,
)
from warpax.geometry.metric import ADMMetric, SymbolicMetric  # noqa: E402
from warpax.visualization import plot_comparison_panel  # noqa: E402


# ============================================================================
# Step 1: Define a custom warp drive metric
# ============================================================================


class GaussianWarpMetric(ADMMetric):
    """Gaussian warp bubble -- a minimal custom warp metric.

    The shift uses a Gaussian profile instead of the Alcubierre top-hat:

        f(r_s) = exp(-r_s^2 / (2 * w^2))

    where ``r_s = sqrt((x - v_s*t)^2 + y^2 + z^2)`` is the bubble-centered
    radius. Lapse is unity and the spatial metric is flat, so the geometry
    is entirely encoded in the shift.

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
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:  # noqa: F722
        return jnp.array(1.0)

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:  # noqa: F722
        f = self.shape_function_value(coords)
        # Shift acts only along x (warp propagation axis), like Alcubierre.
        return jnp.array([-self.v_s * f, 0.0, 0.0])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(
        self, coords: Float[Array, "4"]  # noqa: F722
    ) -> Float[Array, "3 3"]:  # noqa: F722
        return jnp.eye(3)

    @jaxtyped(typechecker=beartype)
    def shape_function_value(
        self, coords: Float[Array, "4"]  # noqa: F722
    ) -> Float[Array, ""]:  # noqa: F722
        """ abstract method: return f(coords) in [0, 1].

        For this Gaussian bubble: ``f = exp(-r_s^2 / (2 w^2))``.
        ``f = 1`` at the bubble center, ``f -> 0`` far from the bubble.
        """
        t, x, y, z = coords
        dx = x - self.v_s * t
        r_s = jnp.sqrt(dx * dx + y * y + z * z)
        return jnp.exp(-(r_s * r_s) / (2.0 * self.w * self.w))

    def symbolic(self) -> SymbolicMetric:
        """Symbolic form for cross-validation against the JAX output."""
        t, x, y, z = sp.symbols("t x y z")
        v_s = sp.Symbol("v_s", positive=True)
        w = sp.Symbol("w", positive=True)

        dx = x - v_s * t
        r_s = sp.sqrt(dx**2 + y**2 + z**2)
        f = sp.exp(-(r_s**2) / (2 * w**2))
        beta_x = -v_s * f

        g = sp.Matrix(
            [
                [-(1 - beta_x**2), beta_x, 0, 0],
                [beta_x, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "GaussianWarp"


# ============================================================================
# Step 2: Single-point curvature + EC verification
# ============================================================================


def run_single_point(metric: GaussianWarpMetric) -> None:
    """Compute the curvature chain at a wall point; print EC margins."""
    # Probe at r_s ~ w (on the wall, where the gradient is steep):
    # f(r_s=w) = exp(-0.5) ~ 0.607.
    coords = jnp.array([0.0, 1.0, 0.5, 0.0])
    f_at_wall = float(metric.shape_function_value(coords))

    print("Custom Gaussian Warp Bubble -- EC Validation")
    print("=" * 55)
    print(f"Parameters: v_s = {metric.v_s}, w = {metric.w}")

    r_s = float(jnp.sqrt(coords[1] ** 2 + coords[2] ** 2 + coords[3] ** 2))
    print(
        f"Probe point: (t,x,y,z) = ({coords[0]}, {coords[1]}, "
        f"{coords[2]}, {coords[3]})"
    )
    print(f"Radial distance: r_s = {r_s:.4f}")
    print(f"Shape function f at probe: {f_at_wall:.4f}")

    # Compute curvature chain via autodiff
    result = compute_curvature_chain(metric, coords)
    K = kretschner_scalar(result.riemann, result.metric, result.metric_inv)

    print(f"\nRicci scalar:      {float(result.ricci_scalar):+.6e}")
    print(f"Kretschner scalar: {float(K):+.6e}")
    print(
        f"Max |T_ab|:        "
        f"{float(jnp.max(jnp.abs(result.stress_energy))):+.6e}"
    )

    # Observer-robust EC verification (all four conditions)
    ec = verify_point(result.stress_energy, result.metric, result.metric_inv)

    print("\nObserver-robust EC margins (negative = violated):")
    print(f"  NEC: {float(ec.nec_margin):+.6e}")
    print(f"  WEC: {float(ec.wec_margin):+.6e}")
    print(f"  SEC: {float(ec.sec_margin):+.6e}")
    print(f"  DEC: {float(ec.dec_margin):+.6e}")
    print(f"  Hawking-Ellis type: {int(ec.he_type)}")

    # Eulerian-frame EC for comparison
    eul_ec = compute_eulerian_ec(
        result.stress_energy, result.metric, result.metric_inv
    )
    print("\nEulerian-frame EC margins:")
    print(f"  NEC: {float(eul_ec['nec']):+.6e}")
    print(f"  WEC: {float(eul_ec['wec']):+.6e}")

    gap = float(eul_ec["nec"]) - float(ec.nec_margin)
    if gap > 1e-10:
        print(
            f"\n  -> Eulerian NEC is {gap:.2e} less negative than robust NEC."
        )
        print(
            " A boosted observer sees a worse violation than the "
            "Eulerian one."
        )


# ============================================================================
# Step 3: Grid-level Eulerian vs robust comparison
# ============================================================================


def run_grid_comparison(
    metric: GaussianWarpMetric,
    grid_n: int = 16,
) -> tuple:
    """Run the full grid EC + Eulerian comparison.

    Uses a small grid (default ``16^3 = 4096`` points) so the end-to-end
    script (grid comparison + a second verify_grid pass for wall-restricted
    stats) stays under 30 seconds on a laptop CPU. Increase
    ``grid_n`` to 50+ for publication-quality density at the cost of
    minutes rather than seconds.
    """
    grid = GridSpec(
        bounds=[(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
        shape=(grid_n, grid_n, grid_n),
    )
    print("\n" + "=" * 55)
    print(f"Grid Analysis: Eulerian vs Observer-Robust (grid_n={grid_n})")
    print("=" * 55)
    print(f"Grid: {grid.shape}, bounds: {grid.bounds}")
    print(f"Total points: {int(np.prod(grid.shape))}")

    print("\n[1/3] Computing curvature tensors on grid...")
    grid_result = evaluate_curvature_grid(metric, grid)
    print(
        f"  Max |Ricci scalar|: "
        f"{float(np.max(np.abs(grid_result.ricci_scalar))):.4e}"
    )
    print(
        f"  Max |T_ab|:         "
        f"{float(np.max(np.abs(grid_result.stress_energy))):.4e}"
    )

    print("\n[2/3] Running Eulerian vs robust comparison...")
    comparison = compare_eulerian_vs_robust(
        grid_result.stress_energy,
        grid_result.metric,
        grid_result.metric_inv,
        grid_shape=grid.shape,
        n_starts=8,
        zeta_max=5.0,
        batch_size=256,
    )

    print("\n[3/3] Comparison summary:")
    for cond in ("nec", "wec", "sec", "dec"):
        eul_min = float(np.min(comparison.eulerian_margins[cond]))
        rob_min = float(np.min(comparison.robust_margins[cond]))
        pct_viol = comparison.pct_violated_robust[cond]
        pct_miss = comparison.pct_missed[cond]
        cond_miss = comparison.conditional_miss_rate[cond]
        print(
            f"  {cond.upper()}: "
            f"eul_min={eul_min:+.3e} rob_min={rob_min:+.3e} "
            f"violated={pct_viol:.1f}% missed={pct_miss:.1f}% "
            f"(cond. {cond_miss:.1f}%)"
        )

    return grid, grid_result, comparison


# ============================================================================
# Step 4: Wall-restricted statistics via the filtering API
# ============================================================================


def run_wall_restricted(
    metric: GaussianWarpMetric,
    grid: GridSpec,
    grid_result,
    comparison,
) -> None:
    """Apply a shape-function mask and compute wall-restricted stats.

    The wall region is defined as the points where the shape function
    ``f(coords)`` is in ``[0.1, 0.9]``: neither deep interior (``f ~ 1``)
    nor far exterior (``f ~ 0``). This is where violations concentrate.
    """
    print("\n" + "=" * 55)
    print("Wall-Restricted Analysis (f in [0.1, 0.9])")
    print("=" * 55)

    # Build a flat coord batch and apply the shape-function mask.
    coords_batch = build_coord_batch(grid, t=0.0)
    wall_mask = shape_function_mask(
        metric,
        coords_batch,
        grid.shape,
        f_low=0.1,
        f_high=0.9,
    )
    n_wall = int(wall_mask.sum())
    n_total = int(np.prod(grid.shape))
    print(f"Grid points in wall region: {n_wall} / {n_total}")

    if n_wall == 0:
        print(
            " WARNING: wall mask is empty -- grid too coarse or bubble "
            "too narrow. Increase grid_n or adjust (f_low, f_high)."
        )
        return

    # compute_wall_restricted_stats consumes an ECGridResult, not the
    # ComparisonResult. Run verify_grid on the curvature data to obtain it.
    ec_grid = verify_grid(
        grid_result.stress_energy,
        grid_result.metric,
        grid_result.metric_inv,
        n_starts=8,
        zeta_max=5.0,
        batch_size=256,
    )

    # Pass the comparison's Eulerian margins to enable wall-restricted
    # conditional miss rates (Eulerian-says-satisfied / robust-says-violated).
    stats = compute_wall_restricted_stats(
        ec_grid,
        wall_mask,
        eulerian_margins=comparison.eulerian_margins,
    )

    def _fmt_rate(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2%}"

    print("\nHawking-Ellis type breakdown (wall):")
    print(f"  Type I:   {stats.n_type_i:6d} ({stats.frac_type_i:.2%})")
    print(f"  Type II:  {stats.n_type_ii:6d} ({stats.frac_type_ii:.2%})")
    print(f"  Type III: {stats.n_type_iii:6d} ({stats.frac_type_iii:.2%})")
    print(f"  Type IV:  {stats.n_type_iv:6d} ({stats.frac_type_iv:.2%})")

    print("\nWall-restricted violation fractions:")
    print(f"  NEC:  {stats.nec_frac_violated:.2%}")
    print(f"  WEC:  {stats.wec_frac_violated:.2%}")
    print(f"  SEC:  {stats.sec_frac_violated:.2%}")
    print(f"  DEC:  {stats.dec_frac_violated:.2%}")

    print("\nWall-restricted conditional miss rates (Eulerian vs robust):")
    print(f"  NEC:  {_fmt_rate(stats.nec_miss_rate)}")
    print(f"  WEC:  {_fmt_rate(stats.wec_miss_rate)}")
    print(f"  SEC:  {_fmt_rate(stats.sec_miss_rate)}")
    print(f"  DEC:  {_fmt_rate(stats.dec_miss_rate)}")


# ============================================================================
# Step 5: Save comparison figure
# ============================================================================


def save_figure(
    grid: GridSpec,
    comparison,
    out_dir: str,
    cond: str = "nec",
) -> None:
    """Save a 3-panel Eulerian/robust/missed comparison figure."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gaussian_warp_comparison.pdf")

    fig = plot_comparison_panel(
        eulerian_margin=np.asarray(comparison.eulerian_margins[cond]),
        robust_margin=np.asarray(comparison.robust_margins[cond]),
        missed=np.asarray(comparison.missed[cond]),
        grid_bounds=grid.bounds,
        grid_shape=grid.shape,
        title=f"Gaussian Warp {cond.upper()}: Eulerian vs Robust",
        save_path=out_path,
    )
    print(f"\nSaved figure: {out_path}")
    # Close the figure so the script exits cleanly in headless mode.
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    metric = GaussianWarpMetric(v_s=0.5, w=1.0)
    run_single_point(metric)
    grid, grid_result, comparison = run_grid_comparison(metric, grid_n=16)
    run_wall_restricted(metric, grid, grid_result, comparison)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    save_figure(grid, comparison, out_dir=output_dir, cond="nec")


if __name__ == "__main__":
    main()
