
"""Geodesic analysis: tidal forces and blueshift through warp bubbles.

Computes timelike geodesic deviation (tidal eigenvalues) and null geodesic
blueshift for the Alcubierre warp metric at multiple warp velocities.

Multi-velocity sweep produces per-velocity .npz files and a summary JSON
(``geodesic_scaling.json``) that documents:

- Tidal force scaling with v_s^2 / sigma.
- Blueshift divergence as v_s -> 1.

Usage
-----
Run full sweep at default velocities (0.1, 0.5, 0.9, 0.99)::

    python scripts/run_geodesics.py

Run a single velocity for quick validation::

    python scripts/run_geodesics.py --velocities 0.5

Force recomputation::

    python scripts/run_geodesics.py --force

Show help::

    python scripts/run_geodesics.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import diffrax
import jax.numpy as jnp

from warpax.benchmarks import AlcubierreMetric
from warpax.geodesics import (
    blueshift_along_trajectory,
    integrate_geodesic,
    integrate_geodesic_with_deviation,
    null_ic,
    tidal_eigenvalues,
    timelike_ic,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = "results/"

# Alcubierre metric parameters
V_S_VALUES = [0.1, 0.5, 0.9, 0.99]
R_BUBBLE = 1.0
SIGMA = 8.0

# Integration parameters
MAX_STEPS = 16384
MAX_STEPS_HIGH_VS = 32768  # For v_s >= 0.95 (near-ergo-region needs more steps)
NUM_POINTS = 500
TAU_SPAN_TIMELIKE = (0.0, 20.0)
TAU_SPAN_NULL = (0.0, 15.0)


def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Tidal force analysis (timelike geodesic with deviation)
# ---------------------------------------------------------------------------


def compute_tidal_analysis(
    results_dir: str, v_s: float = 0.5, force: bool = False
) -> str:
    """Compute tidal eigenvalues along a timelike geodesic through Alcubierre bubble.

    Sets up a timelike geodesic starting at (t=0, x=5.0, y=0.1, z=0)
    moving in the -x direction toward the bubble center. Co-integrates
    the Jacobi deviation equation and extracts tidal eigenvalues at each
    saved point.

    Parameters
    ----------
    results_dir : str
        Directory for cached .npz output.
    v_s : float
        Warp velocity parameter.
    force : bool
        If True, recompute even if cached result exists.

    Returns
    -------
    str
        Path to the saved .npz file.
    """
    cache_path = os.path.join(results_dir, f"geodesic_alcubierre_tidal_vs{v_s}.npz")
    if os.path.exists(cache_path) and not force:
        print(f"  [CACHED] {cache_path} exists, skipping.")
        return cache_path

    _ensure_dir(results_dir)

    print(f"  Setting up Alcubierre metric (v_s={v_s})...")
    metric = AlcubierreMetric(v_s=v_s, R=R_BUBBLE, sigma=SIGMA)

    # Initial conditions: timelike geodesic
    # Start at (t=0, x=5.0, y=0.1, z=0) moving in -x direction
    x0_pos = jnp.array([0.0, 5.0, 0.1, 0.0])
    v_spatial = jnp.array([-0.3, 0.0, 0.0])  # Moderate inward velocity
    x0, v0 = timelike_ic(metric, x0_pos, v_spatial)

    # Deviation initial conditions (spatial perturbation in y-direction)
    xi0 = jnp.array([0.0, 0.0, 1e-6, 0.0])
    w0 = jnp.array([0.0, 0.0, 0.0, 0.0])

    # Use more steps for high v_s (near ergo-region)
    max_steps = MAX_STEPS_HIGH_VS if v_s >= 0.95 else MAX_STEPS

    # Integrate geodesic with deviation
    print("  Integrating timelike geodesic with deviation equations...")
    print(f"    x0 = {x0}, v0 = {v0}")
    print(f"    tau_span = {TAU_SPAN_TIMELIKE}, max_steps = {max_steps}")
    t0 = time.time()
    dev_result = integrate_geodesic_with_deviation(
        metric, x0, v0, xi0, w0,
        tau_span=TAU_SPAN_TIMELIKE,
        num_points=NUM_POINTS,
        max_steps=max_steps,
        rtol=1e-10,
        atol=1e-10,
    )
    t_int = time.time() - t0
    print(f"    Integration time: {t_int:.1f}s")
    print(f"    Result code: {dev_result.result}")
    if dev_result.result == diffrax.RESULTS.max_steps_reached:
        print("    WARNING: Integrator hit max_steps. Partial data will be used.")

    # Extract tidal eigenvalues at each saved point
    print("  Computing tidal eigenvalues along trajectory...")
    t0 = time.time()
    all_eigs = []
    for i in range(NUM_POINTS):
        pos = dev_result.positions[i]
        vel = dev_result.velocities[i]
        eigs = tidal_eigenvalues(metric, pos, vel)
        all_eigs.append(np.asarray(eigs))
    tidal_eigs = np.stack(all_eigs, axis=0)  # (N, 4)
    t_tidal = time.time() - t0
    print(f"    Tidal eigenvalue computation: {t_tidal:.1f}s")

    # Save results
    np.savez(
        cache_path,
        tidal_eigenvalues=tidal_eigs,
        proper_times=np.asarray(dev_result.ts),
        positions=np.asarray(dev_result.positions),
        velocities=np.asarray(dev_result.velocities),
        deviations=np.asarray(dev_result.deviations),
        v_s=v_s,
        R=R_BUBBLE,
        sigma=SIGMA,
    )
    print(f"  Saved: {cache_path}")

    # Summary
    peak_tidal = float(np.nanmax(np.abs(tidal_eigs)))
    print(f"  Peak tidal eigenvalue magnitude: {peak_tidal:.6e}")
    print(f"  Trajectory x-range: [{np.min(np.asarray(dev_result.positions[:, 1])):.2f}, "
          f"{np.max(np.asarray(dev_result.positions[:, 1])):.2f}]")

    return cache_path


# ---------------------------------------------------------------------------
# Blueshift analysis (null geodesic)
# ---------------------------------------------------------------------------


def compute_blueshift_analysis(
    results_dir: str, v_s: float = 0.5, force: bool = False
) -> str:
    """Compute blueshift along a null geodesic through Alcubierre bubble.

    Sets up a null geodesic (photon) starting at (t=0, x=5.0, y=0.0, z=0)
    moving in the -x direction. Computes the blueshift factor at each saved
    step using a static observer field.

    Parameters
    ----------
    results_dir : str
        Directory for cached .npz output.
    v_s : float
        Warp velocity parameter.
    force : bool
        If True, recompute even if cached result exists.

    Returns
    -------
    str
        Path to the saved .npz file.
    """
    cache_path = os.path.join(results_dir, f"geodesic_alcubierre_blueshift_vs{v_s}.npz")
    if os.path.exists(cache_path) and not force:
        print(f"  [CACHED] {cache_path} exists, skipping.")
        return cache_path

    _ensure_dir(results_dir)

    print(f"  Setting up Alcubierre metric (v_s={v_s})...")
    metric = AlcubierreMetric(v_s=v_s, R=R_BUBBLE, sigma=SIGMA)

    # Initial conditions: null geodesic (photon)
    # Start at (t=0, x=5.0, y=0.0, z=0) moving in -x direction
    x0_pos = jnp.array([0.0, 5.0, 0.0, 0.0])
    n_spatial = jnp.array([-1.0, 0.0, 0.0])  # -x direction
    x0, k0 = null_ic(metric, x0_pos, n_spatial)

    # Use more steps for high v_s (near ergo-region)
    max_steps = MAX_STEPS_HIGH_VS if v_s >= 0.95 else MAX_STEPS

    # Integrate null geodesic
    print("  Integrating null geodesic...")
    print(f"    x0 = {x0}, k0 = {k0}")
    print(f"    tau_span = {TAU_SPAN_NULL}, max_steps = {max_steps}")
    t0 = time.time()
    null_result = integrate_geodesic(
        metric, x0, k0,
        tau_span=TAU_SPAN_NULL,
        num_points=NUM_POINTS,
        max_steps=max_steps,
        rtol=1e-10,
        atol=1e-10,
    )
    t_int = time.time() - t0
    print(f"    Integration time: {t_int:.1f}s")
    print(f"    Result code: {null_result.result}")
    if null_result.result == diffrax.RESULTS.max_steps_reached:
        print("    WARNING: Integrator hit max_steps. Partial data will be used.")

    # Compute blueshift along trajectory using static observers
    # Static observer 4-velocity: u^a = (1/sqrt(-g_00), 0, 0, 0)
    print("  Computing blueshift along trajectory...")
    t0 = time.time()

    def static_observer(x):
        """Static observer 4-velocity at position x."""
        g = metric(x)
        g00 = g[0, 0]
        # For Alcubierre, g_00 = -(1 - v_s^2 f^2) which can be positive
        # inside the bubble. Use safe sqrt with sign handling.
        u_t = 1.0 / jnp.sqrt(jnp.abs(g00))
        return jnp.array([u_t, 0.0, 0.0, 0.0])

    blueshift = blueshift_along_trajectory(metric, null_result, static_observer)
    t_bs = time.time() - t0
    print(f"    Blueshift computation: {t_bs:.1f}s")

    # Save results
    positions_x = np.asarray(null_result.positions[:, 1])
    np.savez(
        cache_path,
        blueshift=np.asarray(blueshift),
        positions_x=positions_x,
        positions=np.asarray(null_result.positions),
        velocities=np.asarray(null_result.velocities),
        proper_times=np.asarray(null_result.ts),
        v_s=v_s,
        R=R_BUBBLE,
        sigma=SIGMA,
    )
    print(f"  Saved: {cache_path}")

    # Summary
    max_bs = float(np.nanmax(np.abs(np.asarray(blueshift))))
    print(f"  Max blueshift factor: {max_bs:.6e}")
    print(f"  Trajectory x-range: [{np.min(positions_x):.2f}, "
          f"{np.max(positions_x):.2f}]")

    return cache_path


# ---------------------------------------------------------------------------
# Post-computation analysis
# ---------------------------------------------------------------------------


def analyze_tidal_scaling(results_dir: str, velocities: list[float]) -> dict:
    """Analyze tidal force scaling across velocities.

    Expected scaling: peak tidal force ~ v_s^2 / sigma.
    Normalizes to v_s=0.5 reference if available, else first velocity.

    Returns dict with peak tidal values and scaling ratios.
    """
    print(f"\n{'=' * 50}")
    print("POST-COMPUTATION: Tidal Scaling Analysis")
    print(f"{'=' * 50}")

    peak_tidal_by_vs = {}
    for v_s in velocities:
        path = os.path.join(results_dir, f"geodesic_alcubierre_tidal_vs{v_s}.npz")
        if not os.path.exists(path):
            print(f"  WARNING: Missing {path}, skipping v_s={v_s}")
            continue
        data = np.load(path)
        tidal_eigs = data["tidal_eigenvalues"]
        peak = float(np.nanmax(np.abs(tidal_eigs)))
        peak_tidal_by_vs[str(v_s)] = peak
        print(f"  v_s={v_s}: peak tidal = {peak:.6e}")

    # Compute scaling ratios normalized to v_s=0.5 (or first available)
    ref_vs = 0.5 if "0.5" in peak_tidal_by_vs else velocities[0]
    ref_peak = peak_tidal_by_vs[str(ref_vs)]
    ref_expected = ref_vs**2 / SIGMA

    scaling_ratios = {}
    print(f"\n  Scaling check (peak / (v_s^2/sigma), normalized to v_s={ref_vs}):")
    for vs_str, peak in peak_tidal_by_vs.items():
        v_s = float(vs_str)
        expected = v_s**2 / SIGMA
        # Ratio of (actual/expected) normalized to reference
        if ref_peak > 0 and ref_expected > 0:
            ratio = (peak / expected) / (ref_peak / ref_expected)
        else:
            ratio = float("nan")
        scaling_ratios[vs_str] = round(ratio, 4)
        print(f"    v_s={v_s}: ratio = {ratio:.4f} (1.0 = perfect scaling)")

    return {
        "sigma": SIGMA,
        "reference_vs": ref_vs,
        "peak_tidal_by_vs": peak_tidal_by_vs,
        "scaling_ratios": scaling_ratios,
    }


def analyze_blueshift_divergence(results_dir: str, velocities: list[float]) -> dict:
    """Analyze blueshift divergence near v_s=1.

    Max |blueshift| should increase monotonically with v_s,
    diverging as v_s -> 1.

    Returns dict with max blueshift values and monotonicity check.
    """
    print(f"\n{'=' * 50}")
    print("POST-COMPUTATION: Blueshift Divergence Analysis")
    print(f"{'=' * 50}")

    max_blueshift_by_vs = {}
    for v_s in velocities:
        path = os.path.join(results_dir, f"geodesic_alcubierre_blueshift_vs{v_s}.npz")
        if not os.path.exists(path):
            print(f"  WARNING: Missing {path}, skipping v_s={v_s}")
            continue
        data = np.load(path)
        bs = data["blueshift"]
        max_bs = float(np.nanmax(np.abs(bs)))
        max_blueshift_by_vs[str(v_s)] = max_bs
        print(f"  v_s={v_s}: max |blueshift| = {max_bs:.6e}")

    # Check monotonic increase
    sorted_vs = sorted(max_blueshift_by_vs.keys(), key=float)
    vals = [max_blueshift_by_vs[k] for k in sorted_vs]
    monotonic = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    print(f"\n  Monotonically increasing: {monotonic}")

    if len(vals) >= 2:
        ratio = vals[-1] / vals[0] if vals[0] > 0 else float("inf")
        print(f"  Divergence ratio (max/min v_s): {ratio:.1f}x")

    return {
        "max_blueshift_by_vs": max_blueshift_by_vs,
        "monotonically_increasing": monotonic,
    }


def save_scaling_json(
    results_dir: str,
    tidal_data: dict,
    blueshift_data: dict,
) -> str:
    """Save scaling and divergence data to geodesic_scaling.json."""
    out_path = os.path.join(results_dir, "geodesic_scaling.json")
    payload = {
        "tidal_scaling": tidal_data,
        "blueshift_divergence": blueshift_data,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved scaling data: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Geodesic analysis: tidal forces and blueshift through warp bubbles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cached results exist.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help=f"Output directory for results (default: {RESULTS_DIR}).",
    )
    parser.add_argument(
        "--velocities",
        nargs="+",
        type=float,
        default=None,
        help="Velocities to analyze (default: 0.1 0.5 0.9 0.99).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    velocities = args.velocities if args.velocities else V_S_VALUES

    print("=" * 60)
    print("Geodesic Analysis: Alcubierre Warp Bubble")
    print(f"  Velocities: {velocities}")
    print(f"  R = {R_BUBBLE}, sigma = {SIGMA}")
    print("=" * 60)

    # Run tidal + blueshift analysis at each velocity
    for v_s in velocities:
        print(f"\n{'=' * 50}")
        print(f"v_s = {v_s}")
        print(f"{'=' * 50}")

        print(f"\n  --- Tidal Force Analysis (timelike geodesic + deviation) ---")
        compute_tidal_analysis(results_dir, v_s=v_s, force=args.force)

        print(f"\n  --- Blueshift Analysis (null geodesic) ---")
        compute_blueshift_analysis(results_dir, v_s=v_s, force=args.force)

    # Post-computation analysis (only if we have multiple velocities)
    if len(velocities) > 1:
        tidal_data = analyze_tidal_scaling(results_dir, velocities)
        blueshift_data = analyze_blueshift_divergence(results_dir, velocities)
        scaling_path = save_scaling_json(results_dir, tidal_data, blueshift_data)
    else:
        scaling_path = None

    print(f"\n{'=' * 60}")
    print("Done. Results saved to:")
    for v_s in velocities:
        print(f"  Tidal v_s={v_s}:     {results_dir}geodesic_alcubierre_tidal_vs{v_s}.npz")
        print(f"  Blueshift v_s={v_s}: {results_dir}geodesic_alcubierre_blueshift_vs{v_s}.npz")
    if scaling_path:
        print(f"  Scaling JSON:       {scaling_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
