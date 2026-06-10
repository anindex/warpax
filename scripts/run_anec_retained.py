"""Averaged null energy along null rays for the retained warp metrics.

For each retained metric (Alcubierre, Natário, Van den Broeck, Rodal) at
matched family parameters (R_b = 1, sigma = 8, v_s = 0.5) we integrate the
null contraction ``T_ab k^a k^b`` along a family of axial null rays at varying
perpendicular impact parameter ``b``, using the per-point null-projected
tangent so the integrand is an exact null observable at each sample.

This is a *coordinate null-ray line-integral diagnostic*, not a geodesic ANEC:
the path is the coordinate ray ``x^mu(lambda) = (lambda, x_0 + lambda, b, 0)``
rather than an integrated null geodesic, which for these strong-shift bubbles
drifts off the null cone within the adaptive-RK tolerance budget (see
``run_anec_geodesic_check.py``). A negative line integral is therefore
consistent with, but not a proof of, a violation of the averaged null energy
condition along a complete geodesic. The Minkowski ray integrates to zero and
is retained as a sentinel.

Outputs:
- ../results/anec/retained.json
"""
from __future__ import annotations

import os
from pathlib import Path

from _json_io import dump_json

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.averaged.anec import anec
from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results", "anec")

V_S, R_B, SIGMA = 0.5, 1.0, 8.0
X_START, X_END = -8.0, 8.0
N_SAMPLES = 1024
TANGENT_NORM = "null_projected"
# Impact parameters: dense near the wall (r_s ~ R_b = 1) where the off-axis
# null violations concentrate.
B_SCAN = np.linspace(1.0e-3, 2.5, 80)
SENTINEL_TOL = 1.0e-8

METRICS = {
    "Alcubierre": (AlcubierreMetric, {}),
    "Natário": (NatarioMetric, {}),
    "Van den Broeck": (
        VanDenBroeckMetric,
        {"R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0},
    ),
    "Rodal": (RodalMetric, {}),
}
ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]


def _instantiate(name: str):
    cls, extra = METRICS[name]
    return cls(v_s=V_S, R=R_B, sigma=SIGMA, **extra)


def _axial_ray(b: float):
    """Coordinate null ray x = x_start + lambda, y = b, advancing in t."""

    def geo(affine):
        return jnp.stack(
            [
                jnp.asarray(affine),
                jnp.asarray(X_START + affine),
                jnp.asarray(b),
                jnp.asarray(0.0),
            ]
        )

    return geo


def _anec_along(metric, b: float) -> float:
    res = anec(
        metric,
        _axial_ray(b),
        tangent_norm=TANGENT_NORM,
        n_samples=N_SAMPLES,
        affine_bounds=(0.0, X_END - X_START),
    )
    return float(res.line_integral)


def _minkowski_sentinel() -> float:
    worst = 0.0
    for b in (1.0e-3, 0.5, 1.0, 1.5):
        worst = max(worst, abs(_anec_along(MinkowskiMetric(), b)))
    return worst


def main() -> None:
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    sentinel = _minkowski_sentinel()
    print(f"Minkowski sentinel |ANEC|_max = {sentinel:.2e} "
          f"({'PASS' if sentinel < SENTINEL_TOL else 'FAIL'})")
    if sentinel >= SENTINEL_TOL:
        raise RuntimeError(
            f"Minkowski ANEC sentinel {sentinel:.2e} exceeds tol {SENTINEL_TOL}"
        )

    per_metric: dict[str, dict] = {}
    for name in ORDER:
        metric = _instantiate(name)
        on_axis = _anec_along(metric, float(B_SCAN[0]))
        scan = np.array([_anec_along(metric, float(b)) for b in B_SCAN])
        j = int(np.argmin(scan))
        per_metric[name] = {
            "on_axis": on_axis,
            "min_line_integral": float(scan[j]),
            "b_at_min": float(B_SCAN[j]),
            "max_line_integral": float(scan.max()),
            "b_scan": B_SCAN.tolist(),
            "line_integral_scan": scan.tolist(),
        }
        print(f"  {name:16s} on-axis={on_axis:+.4e}  "
              f"min={scan[j]:+.4e} @ b={B_SCAN[j]:.3f}  max={scan.max():+.3e}")

    out = {
        "params": {
            "v_s": V_S, "R_b": R_B, "sigma": SIGMA,
            "x_bounds": [X_START, X_END], "n_samples": N_SAMPLES,
            "tangent_norm": TANGENT_NORM,
        },
        "minkowski_sentinel_abs": sentinel,
        "order": ORDER,
        "metrics": per_metric,
    }
    out_path = os.path.join(RESULTS_DIR, "retained.json")
    dump_json(out, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
