"""- Reproduce Alcubierre tanh via 24-knot cubic B-spline.

Target: relative error < 1e-4 between the 24-knot cubic B-spline
**control points** and the canonical Alcubierre tanh shape, when
evaluated AT the knot locations (where cubic spline interpolation is
exact). This is the pipeline-consistency golden test per
Usage
-----

::

    JAX_PLATFORMS=cpu python examples/08_metric_design.py
    JAX_PLATFORMS=cpu python examples/08_metric_design.py --probe-grid dense

Output: prints the relative error and writes the golden fixture
``tests/fixtures/e2_optimal_parameters_v1_1_0.npy`` .

The optimizer runs with ``max_steps=0`` - the reproduction
path that short-circuits and returns the input spline unchanged.
This verifies pipeline consistency at the spline's control knots,
independent of any BFGS convergence behavior.

: with ``--probe-grid dense`` the example additionally
evaluates rel_err at 100 uniformly-spaced probe points on [0, 12] -
probing mid-interval spline reconstruction error that the knot-only
default cannot see. Results are written to

"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

# Defensive float64 (in case this is run standalone before warpax import)
jax.config.update("jax_enable_x64", True)

from warpax.design import ShapeFunction, design_metric


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Alcubierre tanh reproduction via 24-knot cubic B-spline.",
    )
    parser.add_argument(
        "--probe-grid",
        choices=["sparse", "dense"],
        default="sparse",
        help=(
            "sparse: evaluate rel_err at the 24 spline knots only "
            "(v1.1 behaviour, default). "
            "dense: also evaluate rel_err at 100 uniformly-spaced probe "
            "points on [0, 12] and write a JSON measurement to "
            
        ),
    )
    parser.add_argument(
        "--probe-grid-size",
        type=int,
        default=100,
        help="Number of dense probe points when --probe-grid=dense (default: 100).",
    )
    args = parser.parse_args(argv)

    R = 1.0
    sigma = 0.1
    n_knots = 24

    # 24 knots uniformly over [0, 12]
    knots = jnp.linspace(0.0, 12.0, n_knots)
    # Alcubierre bubble profile: 1 - tanh((r - R)/sigma)^2
    # (bump of width ~sigma centred at R, bounded in [0, 1]).
    values = 1.0 - jnp.tanh((knots - R) / sigma) ** 2

    # Build the starting shape-function
    shape = ShapeFunction.spline(knots, values)

    # Optimize with max_steps=0 short-circuit (reproduction path)
    metric, report = design_metric(
        shape,
        objective="nec",
        strategy="hard_bound",
        n_starts=16,
        max_steps=0,  # at extreme: preserve starting spline bit-exactly
        key=jax.random.PRNGKey(42),  # 
    )

    # Verify: sample at the knots (cubic spline interpolation exact there)
    recovered = jax.vmap(metric.shape_fn)(knots)
    rel_err = jnp.max(jnp.abs(recovered - values)) / jnp.max(jnp.abs(values))

    print(f"Reproduction (24-knot cubic B-spline, R={R}, sigma={sigma})")
    print(f"  Relative error at knots : {float(rel_err):.2e}")
    print(f"  Target                   : < 1e-4   ")
    print(f"  Physical                 : {report.physicality.overall}")
    print(f"  Optimizer converged      : {report.converged}")
    print(f"  Strategy                 : {report.strategy}")

    assert rel_err < 1e-4, (
        f"FAILED: rel_err={float(rel_err):.2e} >= 1e-4"
    )

    # Save golden fixture for 
    fixture_path = (
        pathlib.Path(__file__).parent.parent
        / "tests" / "fixtures" / "e2_optimal_parameters_v1_1_0.npy"
    )
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    final_values = np.asarray(metric.shape_fn.params["values"])
    np.save(fixture_path, final_values)
    print(f"\nGolden fixture saved: {fixture_path}")
    print(f"  Shape: {final_values.shape}, dtype: {final_values.dtype}")

    if args.probe_grid == "dense":
        # : dense probe grid rel_err.
        # 100-point uniform 1D grid on [0, 12].
        # scope-honest adaptive - measure, record, do not gate.
        n_probe = args.probe_grid_size
        x_probe = jnp.linspace(0.0, 12.0, n_probe)
        f_true_probe = 1.0 - jnp.tanh((x_probe - R) / sigma) ** 2
        f_recon_probe = jax.vmap(metric.shape_fn)(x_probe)
        eps = 1e-12
        rel_err_probe = jnp.abs(f_recon_probe - f_true_probe) / jnp.maximum(
            jnp.abs(f_true_probe), eps
        )
        max_rel_err = float(jnp.max(rel_err_probe))
        mean_rel_err = float(jnp.mean(rel_err_probe))
        median_rel_err = float(jnp.median(rel_err_probe))

        print()
        print(f"Dense probe grid ({n_probe} uniform points on [0, 12])")
        print(f"  max rel_err    : {max_rel_err:.3e}")
        print(f"  mean rel_err   : {mean_rel_err:.3e}")
        print(f"  median rel_err : {median_rel_err:.3e}")
        print(
            f"  rel_err at knots : "
            f"{float(rel_err):.2e} (exact by spline construction)"
        )

        # Capture commit SHA (best-effort; non-fatal if git/repo unavailable).
        try:
            commit_sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=pathlib.Path(__file__).parent.parent,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            commit_sha = "unknown"

        payload = {
            "artifact_type": "research_extension_measurement",
            "metric": "alcubierre_tanh_shape_function",
            "R": R,
            "sigma": sigma,
            "knot_count": n_knots,
            "knot_bounds": [0.0, 12.0],
            "probe_grid_size": n_probe,
            "probe_grid_bounds": [0.0, 12.0],
            "probe_grid_kind": "uniform_1d",
            "max_steps": 0,
            "strategy": "hard_bound",
            "objective": "nec",
            "rel_err_definition": "|f_recon - f_true| / max(|f_true|, 1e-12)",
            "max_rel_err": max_rel_err,
            "mean_rel_err": mean_rel_err,
            "median_rel_err": median_rel_err,
            "rel_err_at_knots": 0.0,
            "recipe": (
                "JAX_PLATFORMS=cpu python examples/08_metric_design.py "
                "--probe-grid dense"
            ),
            "jax_version": jax.__version__,
            "date": datetime.datetime.now(datetime.UTC).isoformat(
                timespec="seconds"
            ),
            "commit_sha": commit_sha,
        }

        json_path = (
            pathlib.Path(__file__).parent.parent
            / "results"
            / "design_dense_probe.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

        print(f"\nDense probe JSON written: {json_path}")


if __name__ == "__main__":
    main()
