
"""Missed-violation detection comparison: Eulerian vs sampling vs BFGS.

For each metric/velocity where BFGS optimization finds missed violations
(Eulerian >= 0 but robust < 0), test whether dense observer sampling at
various densities (100, 1000, 10000) would also detect those violations.

This directly answers the reviewer concern: "you're only beating a
strawman Eulerian baseline, what about realistic sampling?"

Loads cached .npz data and only re-evaluates the missed-violation subset,
keeping runtime manageable.

Usage:
    python scripts/run_missed_detection_comparison.py
    python scripts/run_missed_detection_comparison.py --max-points 500
"""
from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from warpax.energy_conditions.observer import (
    compute_orthonormal_tetrad,
    timelike_from_rapidity,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CONDITIONS = ["nec", "wec", "dec"]

# Metrics and velocities to test
CASES = [
    ("rodal", 0.1),
    ("rodal", 0.5),
    ("rodal", 0.9),
    ("warpshell", 0.1),
    ("warpshell", 0.5),
    ("warpshell", 0.9),
]

SAMPLING_DENSITIES = [100, 1000, 10000]


def dense_sample_margin(
    T_ab: jnp.ndarray,
    g_ab: jnp.ndarray,
    n_samples: int,
    zeta_max: float = 5.0,
    key=None,
    condition: str = "wec",
) -> float:
    """Evaluate EC margin via dense random observer sampling."""
    if key is None:
        key = jax.random.PRNGKey(42)
    tetrad = compute_orthonormal_tetrad(g_ab)

    k1, k2, k3 = jax.random.split(key, 3)
    zetas = jax.random.uniform(k1, (n_samples,), minval=0.0, maxval=zeta_max)
    cos_thetas = jax.random.uniform(k2, (n_samples,), minval=-1.0, maxval=1.0)
    thetas = jnp.arccos(cos_thetas)
    phis = jax.random.uniform(k3, (n_samples,), minval=0.0, maxval=2.0 * jnp.pi)

    def eval_single(zeta, theta, phi):
        u = timelike_from_rapidity(zeta, theta, phi, tetrad)
        rho = jnp.einsum("a,ab,b->", u, T_ab, u)
        return rho

    margins = jax.vmap(eval_single)(zetas, thetas, phis)
    return float(jnp.min(margins))


# JIT the inner loop for speed
@jax.jit
def _batch_tetrad(g_batch):
    return jax.vmap(compute_orthonormal_tetrad)(g_batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-points", type=int, default=1000,
                        help="Max missed points to test per case")
    args = parser.parse_args()

    print("=" * 80)
    print("Missed-Violation Detection Comparison: Eulerian vs Sampling vs BFGS")
    print("=" * 80)

    all_results = {}

    for metric_name, v_s in CASES:
        cache_path = os.path.join(RESULTS_DIR, f"{metric_name}_vs{v_s}.npz")
        if not os.path.exists(cache_path):
            print(f"\n  {metric_name} v_s={v_s}: SKIPPED (no cached data)")
            continue

        data = np.load(cache_path)
        case_key = f"{metric_name}_vs{v_s}"
        print(f"\n{'='*80}")
        print(f"{metric_name} (v_s={v_s})")
        print(f"{'='*80}")

        for cond in CONDITIONS:
            eul_key = f"{cond}_eulerian"
            rob_key = f"{cond}_robust"
            if eul_key not in data or rob_key not in data:
                continue

            eul = data[eul_key].ravel()
            rob = data[rob_key].ravel()

            # Points missed by Eulerian but detected by BFGS
            missed_mask = (eul >= 0) & (rob < -1e-10)
            n_missed = int(np.sum(missed_mask))
            n_total = len(eul)

            if n_missed == 0:
                print(f"\n  {cond.upper()}: No missed violations (0/{n_total})")
                continue

            print(f"\n  {cond.upper()}: {n_missed} missed violations "
                  f"({100*n_missed/n_total:.3f}%)")

            # Select subset of missed points
            missed_indices = np.where(missed_mask)[0]
            if len(missed_indices) > args.max_points:
                rng = np.random.RandomState(42)
                missed_indices = rng.choice(
                    missed_indices, args.max_points, replace=False
                )
                n_test = args.max_points
            else:
                n_test = len(missed_indices)

            print(f"  Testing {n_test} missed points...")

            # Load stress-energy and metric data
            # We need to reconstruct T_ab and g_ab from the grid
            # Since we don't store them in the .npz, we'll use the
            # margins to verify and the grid to reconstruct
            #
            # Actually - we need the actual T_ab and g_ab tensors.
            # The cached .npz only has margins. We need to recompute
            # from the metric. Let's load and reconstruct.

            # Grid params
            grid_bounds = data["grid_bounds"]
            grid_shape = tuple(data["grid_shape"])
            n_grid = int(np.prod(grid_shape))

            # Reconstruct grid coordinates
            axes = [np.linspace(lo, hi, int(n))
                    for (lo, hi), n in zip(grid_bounds, grid_shape)]
            X, Y, Z = np.meshgrid(*axes, indexing="ij")
            T_grid = np.zeros_like(X)
            coords = np.stack([T_grid, X, Y, Z], axis=-1).reshape(-1, 4)

            # Import metric class
            if metric_name == "rodal":
                from warpax.metrics import RodalMetric
                metric = RodalMetric(v_s=v_s, R=100.0, sigma=0.03)
            elif metric_name == "warpshell":
                from warpax.metrics import WarpShellMetric
                metric = WarpShellMetric(v_s=v_s, R_1=0.5, R_2=1.0)
            else:
                continue

            # Compute curvature chain for the missed points only
            from warpax.geometry.geometry import compute_curvature_chain

            missed_coords = jnp.array(coords[missed_indices])

            print(f"  Computing curvature at {n_test} points...")
            t0 = time.time()

            chain_fn = lambda x: compute_curvature_chain(metric, x)

            # Process in chunks
            chunk_size = 500
            T_missed = []
            g_missed = []
            for i in range(0, n_test, chunk_size):
                chunk = missed_coords[i:i + chunk_size]
                result = jax.vmap(chain_fn)(chunk)
                T_missed.append(np.array(result.stress_energy))
                g_missed.append(np.array(result.metric))

            T_missed = np.concatenate(T_missed, axis=0)
            g_missed = np.concatenate(g_missed, axis=0)
            print(f"  Curvature: {time.time() - t0:.1f}s")

            # Test each sampling density
            cond_results = {}
            for n_samp in SAMPLING_DENSITIES:
                print(f"\n    Sampling N={n_samp}:")
                t0 = time.time()
                detected = 0

                for j in range(n_test):
                    T_ab = jnp.array(T_missed[j])
                    g_ab = jnp.array(g_missed[j])

                    samp_margin = dense_sample_margin(
                        T_ab, g_ab,
                        n_samples=n_samp,
                        zeta_max=5.0,
                        key=jax.random.PRNGKey(j),
                        condition=cond,
                    )

                    if samp_margin < -1e-10:
                        detected += 1

                elapsed = time.time() - t0
                detection_rate = 100 * detected / n_test if n_test > 0 else 0

                cond_results[n_samp] = {
                    "n_tested": n_test,
                    "n_detected": detected,
                    "detection_rate_pct": detection_rate,
                    "time_s": elapsed,
                }

                print(f"      Detected: {detected}/{n_test} "
                      f"({detection_rate:.1f}%) in {elapsed:.1f}s")

            # Summary for this condition
            print(f"\n    Summary for {cond.upper()}:")
            print(f"      {'Method':>12s} | {'Detected':>8s} | {'Rate':>6s}")
            print(f"      {'-'*12}-+-{'-'*8}-+-{'-'*6}")
            print(f"      {'Eulerian':>12s} | {'0':>8s} | {'0.0%':>6s}")
            for n_samp in SAMPLING_DENSITIES:
                r = cond_results[n_samp]
                print(f"      {f'Sample-{n_samp}':>12s} | "
                      f"{r['n_detected']:>8d} | "
                      f"{r['detection_rate_pct']:>5.1f}%")
            print(f"      {'BFGS-8':>12s} | {n_test:>8d} | {'100.0%':>6s}")

            if case_key not in all_results:
                all_results[case_key] = {}
            all_results[case_key][cond] = {
                "n_missed_total": n_missed,
                "n_tested": n_test,
                "sampling_results": cond_results,
            }

    # Save results
    import json
    out_path = os.path.join(RESULTS_DIR, "missed_detection_comparison.json")
    # Convert to serializable
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {}
        for cond, data in v.items():
            serializable[k][cond] = {
                "n_missed_total": data["n_missed_total"],
                "n_tested": data["n_tested"],
                "sampling_results": {
                    str(ns): {
                        "n_detected": r["n_detected"],
                        "detection_rate_pct": r["detection_rate_pct"],
                    }
                    for ns, r in data["sampling_results"].items()
                },
            }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
