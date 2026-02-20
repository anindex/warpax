
"""Compare BFGS optimization vs dense observer sampling for WEC and DEC.

For a random subset of points on the Alcubierre and Rodal metrics,
compute:
  1. BFGS multi-start minimum (N_starts = 8)
  2. Dense sampling minimum (N_samples = 1000, 5000, 10000)
and report how often BFGS finds equal or lower minima.

Additionally, for Rodal DEC, compares deterministic Fibonacci-lattice
sampling (N_dir = {10, 50, 200, 1000} x N_zeta = 10) against BFGS
and algebraic truth, producing a convergence-of-sampled-min figure.
"""

from __future__ import annotations

import argparse
import json
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
from warpax.energy_conditions.optimization import optimize_wec
from warpax.geometry.geometry import compute_curvature_chain


def dense_sampling_wec(
    T_ab: jnp.ndarray,
    g_ab: jnp.ndarray,
    n_samples: int = 1000,
    zeta_max: float = 5.0,
    key: jax.random.PRNGKey = None,
) -> float:
    """Evaluate WEC margin via dense observer sampling.

    Sample n_samples random (zeta, theta, phi) and return the minimum
    T_{ab} u^a u^b.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    tetrad = compute_orthonormal_tetrad(g_ab)

    k1, k2, k3 = jax.random.split(key, 3)
    zetas = jax.random.uniform(k1, (n_samples,), minval=0.0, maxval=zeta_max)
    # Uniform on S^2: cos(theta) ~ U[-1,1], phi ~ U[0,2pi]
    cos_thetas = jax.random.uniform(k2, (n_samples,), minval=-1.0, maxval=1.0)
    thetas = jnp.arccos(cos_thetas)
    phis = jax.random.uniform(k3, (n_samples,), minval=0.0, maxval=2.0 * jnp.pi)

    def eval_wec(zeta, theta, phi):
        u = timelike_from_rapidity(zeta, theta, phi, tetrad)
        return jnp.einsum("a,ab,b->", u, T_ab, u)

    margins = jax.vmap(eval_wec)(zetas, thetas, phis)
    return float(jnp.min(margins))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=200,
                        help="Number of random grid points to test")
    parser.add_argument("--n-samples", type=int, nargs="+",
                        default=[1000, 5000, 10000],
                        help="Sampling densities to test")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    from warpax.benchmarks.alcubierre import AlcubierreMetric
    from warpax.metrics.rodal import RodalMetric

    metrics = {
        "alcubierre": AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0),
        "rodal": RodalMetric(v_s=0.5, R=1.0, sigma=8.0),
    }

    results = {}

    for name, metric in metrics.items():
        print(f"\n=== {name} ===")
        R = 1.0
        grid_half = 2.0 * R
        N = 50

        # Generate grid
        ax = np.linspace(-grid_half, grid_half, N)
        xs, ys, zs = np.meshgrid(ax, ax, ax, indexing="ij")
        coords = np.stack([np.zeros_like(xs), xs, ys, zs], axis=-1)
        flat_coords = coords.reshape(-1, 4)

        # Evaluate curvature chain
        print("  Computing curvature chain...")
        t0 = time.time()
        chain_fn = lambda x: compute_curvature_chain(metric, x)
        chain_batch = jax.vmap(chain_fn)

        # Process in chunks to avoid memory issues
        chunk_size = 5000
        all_T = []
        all_g = []
        for i in range(0, len(flat_coords), chunk_size):
            chunk = jnp.array(flat_coords[i : i + chunk_size])
            result = chain_batch(chunk)
            all_T.append(np.array(result.stress_energy))
            all_g.append(np.array(result.metric))
        T_all = np.concatenate(all_T, axis=0)
        g_all = np.concatenate(all_g, axis=0)
        print(f"  Curvature chain: {time.time() - t0:.1f}s")

        # Select random subset of VIOLATING points (more interesting)
        key = jax.random.PRNGKey(123)
        n_pts = min(args.n_points, len(T_all))
        indices = np.random.RandomState(42).choice(len(T_all), n_pts, replace=False)

        metric_results = {"n_points": n_pts, "sampling_densities": {}}

        for n_samp in args.n_samples:
            print(f"\n  Sampling density: {n_samp}")
            bfgs_wins = 0
            sampling_wins = 0
            ties = 0
            bfgs_margins = []
            samp_margins = []

            for idx_i, idx in enumerate(indices):
                T_ab = jnp.array(T_all[idx])
                g_ab = jnp.array(g_all[idx])

                # BFGS with 8 starts
                bfgs_result = optimize_wec(
                    T_ab, g_ab, n_starts=8, zeta_max=5.0,
                    key=jax.random.PRNGKey(idx)
                )
                bfgs_min = float(bfgs_result.margin)

                # Dense sampling
                samp_min = dense_sampling_wec(
                    T_ab, g_ab, n_samples=n_samp, zeta_max=5.0,
                    key=jax.random.PRNGKey(idx + 1000000)
                )

                bfgs_margins.append(bfgs_min)
                samp_margins.append(samp_min)

                tol = 1e-6
                if bfgs_min < samp_min - tol:
                    bfgs_wins += 1
                elif samp_min < bfgs_min - tol:
                    sampling_wins += 1
                else:
                    ties += 1

                if (idx_i + 1) % 50 == 0:
                    print(f"    {idx_i + 1}/{n_pts}: "
                          f"BFGS better={bfgs_wins}, "
                          f"Samp better={sampling_wins}, "
                          f"Ties={ties}")

            bfgs_arr = np.array(bfgs_margins)
            samp_arr = np.array(samp_margins)
            diff = samp_arr - bfgs_arr  # positive = BFGS found lower

            metric_results["sampling_densities"][n_samp] = {
                "bfgs_wins": bfgs_wins,
                "sampling_wins": sampling_wins,
                "ties": ties,
                "mean_improvement": float(np.mean(diff)),
                "median_improvement": float(np.median(diff)),
                "max_improvement": float(np.max(diff)),
                "pct_bfgs_equal_or_better": float(
                    100.0 * (bfgs_wins + ties) / n_pts
                ),
            }
            print(f"  Results for N_samp={n_samp}:")
            print(f"    BFGS better: {bfgs_wins}/{n_pts} "
                  f"({100*bfgs_wins/n_pts:.1f}%)")
            print(f"    Sampling better: {sampling_wins}/{n_pts} "
                  f"({100*sampling_wins/n_pts:.1f}%)")
            print(f"    Ties: {ties}/{n_pts}")
            print(f"    Mean improvement (samp-bfgs): {np.mean(diff):.6f}")

        results[name] = metric_results

    # Save results
    out_path = f"{args.results_dir}/sampling_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """Generate approximately uniform points on S^2 via Fibonacci lattice.

    Returns (theta, phi) pairs of shape (n_points, 2).
    """
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_points)
    theta = np.arccos(1 - 2 * (indices + 0.5) / n_points)
    phi = (2 * np.pi * indices / golden_ratio) % (2 * np.pi)
    return np.stack([theta, phi], axis=-1)



def fibonacci_dec_comparison():
    """Deterministic Fibonacci-lattice DEC comparison for Rodal.

    Compares sampled DEC min vs BFGS min vs algebraic truth.
    Vectorized with JAX for performance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from warpax.metrics.rodal import RodalMetric
    from warpax.geometry import GridSpec, evaluate_curvature_grid
    from warpax.energy_conditions.verifier import verify_grid
    from warpax.energy_conditions.observer import compute_orthonormal_tetrad
    from warpax.energy_conditions.eigenvalue_checks import check_dec
    from warpax.energy_conditions.classification import classify_hawking_ellis

    n_dir_values = [10, 50, 200, 1000]
    n_zeta = 10
    zeta_max = 5.0
    v_s = 0.5
    grid = GridSpec(bounds=[(-300, 300)] * 3, shape=(25, 25, 25))

    print("\n" + "=" * 70)
    print("Fibonacci DEC Sampling vs BFGS (Rodal)")
    print("=" * 70)

    metric = RodalMetric(v_s=v_s, R=100.0, sigma=0.03)
    curv = evaluate_curvature_grid(metric, grid, batch_size=256)

    flat_T = curv.stress_energy.reshape(-1, 4, 4)
    flat_g = curv.metric.reshape(-1, 4, 4)
    flat_g_inv = curv.metric_inv.reshape(-1, 4, 4)
    n_points = flat_T.shape[0]

    # Algebraic truth
    flat_T_mixed = jax.vmap(jnp.matmul)(flat_g_inv, flat_T)
    cls_results = jax.vmap(classify_hawking_ellis)(flat_T_mixed, flat_g)
    alg_dec = np.array(jax.vmap(check_dec)(cls_results.rho, cls_results.pressures))

    # BFGS
    print("Running BFGS...")
    ec_bfgs = verify_grid(
        curv.stress_energy, curv.metric, curv.metric_inv,
        n_starts=8, batch_size=64,
    )
    bfgs_dec = np.array(ec_bfgs.dec_margins.reshape(-1))

    # Violation points
    viol_mask = alg_dec < -1e-10
    n_viol = int(np.sum(viol_mask))
    print(f"Algebraic DEC violations: {n_viol}/{n_points}")

    zetas = jnp.linspace(0, zeta_max, n_zeta)

    # ---- Vectorized DEC sampling kernel ----
    # For a single point, evaluate all (direction, zeta) pairs at once.

    @jax.jit
    def _dec_margin_single_observer(T_ab, T_mixed, g_ab, tetrad, zeta, theta, phi):
        """Compute DEC margin for one observer (zeta, theta, phi) at one point."""
        u = timelike_from_rapidity(zeta, theta, phi, tetrad)
        # WEC: T_{ab} u^a u^b
        wec = jnp.einsum("a,ab,b->", u, T_ab, u)
        # Energy-momentum current: j^a = -T^a_b u^b
        j = -jnp.einsum("ab,b->a", T_mixed, u)
        # Causal flux: -j^a g_{ab} j^b  (should be >= 0 if j is causal)
        flux = -jnp.einsum("a,ab,b->", j, g_ab, j)
        # Future-directed: -j^a n_a  (should be >= 0)
        n_down = jnp.einsum("ab,b->a", g_ab, tetrad[0])
        future = -jnp.einsum("a,a->", j, n_down)
        return jnp.minimum(wec, jnp.minimum(flux, future))

    @jax.jit
    def _sample_dec_at_point(T_ab, T_mixed, g_ab, tetrad, thetas, phis, zetas):
        """Compute min DEC margin over all (direction, zeta) pairs at one point.

        thetas, phis: shape (n_dir,)
        zetas: shape (n_zeta,)
        Returns scalar: worst margin over n_dir * n_zeta observers.
        """
        # Create all combinations: (n_dir, n_zeta)
        th_grid = jnp.repeat(thetas, len(zetas))       # (n_dir * n_zeta,)
        ph_grid = jnp.repeat(phis, len(zetas))          # (n_dir * n_zeta,)
        z_grid = jnp.tile(zetas, len(thetas))            # (n_dir * n_zeta,)

        margins = jax.vmap(
            lambda z, th, ph: _dec_margin_single_observer(
                T_ab, T_mixed, g_ab, tetrad, z, th, ph
            )
        )(z_grid, th_grid, ph_grid)
        return jnp.nanmin(margins)

    # Precompute tetrads for all points (vectorized)
    print("Computing tetrads...")
    t0 = time.time()
    all_tetrads = jax.vmap(compute_orthonormal_tetrad)(flat_g)
    all_tetrads.block_until_ready()
    print(f"  Tetrads: {time.time() - t0:.1f}s")

    results = []
    for n_dir in n_dir_values:
        print(f"\nN_dir = {n_dir}, total samples = {n_dir * n_zeta}")
        directions = fibonacci_sphere(n_dir)
        thetas_arr = jnp.array(directions[:, 0])
        phis_arr = jnp.array(directions[:, 1])

        t0 = time.time()

        # Process points in chunks to avoid excessive JIT recompilation
        # and memory usage. Each chunk is vmapped.
        chunk_size = 256
        sampled_mins = np.full(n_points, np.inf)

        # Define the per-point function for this n_dir
        def _point_fn(args):
            T_ab, T_mixed, g_ab, tetrad = args
            return _sample_dec_at_point(
                T_ab, T_mixed, g_ab, tetrad, thetas_arr, phis_arr, zetas
            )

        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            chunk_T = flat_T[start:end]
            chunk_Tm = flat_T_mixed[start:end]
            chunk_g = flat_g[start:end]
            chunk_tet = all_tetrads[start:end]

            chunk_mins = jax.vmap(
                lambda T_ab, T_mixed, g_ab, tet: _sample_dec_at_point(
                    T_ab, T_mixed, g_ab, tet, thetas_arr, phis_arr, zetas
                )
            )(chunk_T, chunk_Tm, chunk_g, chunk_tet)

            sampled_mins[start:end] = np.array(chunk_mins)

            if (start // chunk_size) % 10 == 0:
                print(f"    Chunk {start//chunk_size + 1}/"
                      f"{(n_points + chunk_size - 1)//chunk_size}")

        elapsed = time.time() - t0
        detects = int(np.sum(sampled_mins[viol_mask] < -1e-10))
        rate = detects / max(n_viol, 1) * 100

        results.append({
            "n_dir": n_dir,
            "detection_rate": rate,
            "min_sampled": float(np.nanmin(sampled_mins)),
            "elapsed": elapsed,
        })
        print(f"  Detection: {rate:.1f}% ({detects}/{n_viol}), time={elapsed:.1f}s")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'N_dir':>8} {'N_total':>8} {'Detection%':>12} {'Min margin':>14} {'Time(s)':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_dir']:>8} {r['n_dir']*n_zeta:>8} {r['detection_rate']:>11.1f}% "
              f"{r['min_sampled']:>14.6e} {r['elapsed']:>10.1f}")
    print("=" * 70)
    print(f"BFGS min DEC margin: {float(np.nanmin(bfgs_dec)):.6e}")
    print(f"Algebraic min DEC margin: {float(np.nanmin(alg_dec)):.6e}")

    # Plot
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(
        [r["n_dir"] for r in results],
        [r["detection_rate"] for r in results],
        "o-", label="Fibonacci sampling",
    )
    ax.axhline(100, ls="--", color="green", label="BFGS (100%)")
    ax.set_xlabel("Number of directions")
    ax.set_ylabel("DEC violation detection rate (%)")
    ax.set_title("Rodal DEC: Fibonacci Sampling vs BFGS")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "fibonacci_vs_bfgs_dec.pdf")
    fig.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    # Save JSON
    json_out = "results/fibonacci_dec_comparison.json"
    with open(json_out, "w") as f:
        json.dump({
            "n_zeta": n_zeta,
            "zeta_max": zeta_max,
            "n_points": n_points,
            "n_violations": n_viol,
            "bfgs_min_dec": float(np.nanmin(bfgs_dec)),
            "alg_min_dec": float(np.nanmin(alg_dec)),
            "results": results,
        }, f, indent=2)
    print(f"JSON saved: {json_out}")


if __name__ == "__main__":
    import sys
    if "--fibonacci-dec" in sys.argv:
        fibonacci_dec_comparison()
    else:
        main()
