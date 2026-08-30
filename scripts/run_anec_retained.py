"""Averaged null energy along null rays for the retained warp metrics.

For each retained metric (Alcubierre, Natário, Van den Broeck, Rodal) at
matched family parameters (R_b = 1, sigma = 8, v_s = 0.5) we integrate the
null contraction ``T_ab k^a k^b`` along a family of axial null rays at varying
perpendicular impact parameter ``b``, using the per-point null-projected
tangent so the integrand is an exact null observable at each sample.

This is a *coordinate null-ray line-integral diagnostic*, not a geodesic ANEC:
the path is the coordinate ray ``x^mu(lambda) = (lambda, x_0 + lambda, b, 0)``
rather than an integrated null geodesic, which for these strong-shift bubbles
drifts off the null cone within an adaptive-RK tolerance budget; the integrated
geodesic the paper reports is the symplectic one of ``run_anec_symplectic.py``,
which holds the Killing energy to better than ``1e-5`` along every retained ray.
A negative line integral here is therefore
consistent with, but not a proof of, a violation of the averaged null energy
condition along a complete geodesic. The Minkowski ray integrates to zero and
is retained as a sentinel.

Outputs:
- ../results/anec/retained.json
"""
from __future__ import annotations

import os
from pathlib import Path

from _anec_window import crossing_span
from _paper_metrics import instantiate
from _json_io import dump_json

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.averaged.anec import anec
from warpax.benchmarks import MinkowskiMetric

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results", "anec")

V_S, R_B, SIGMA = 0.5, 1.0, 8.0
X_START, X_END = -8.0, 8.0
# The span was X_END - X_START = 16 = 2|X_START|, which ends at the receding
# bubble's centre: every published value here was exactly half (2.00, 2.02 for
# Rodal). Measured now, not assumed, see _anec_window.
SPAN0 = X_END - X_START
N_SAMPLES = 1024  # at SPAN0; scaled with the span so the step density is fixed
TANGENT_NORM = "null_projected"
# Impact parameters: dense near the wall (r_s ~ R_b = 1) where the off-axis
# null violations concentrate.
B_SCAN = np.linspace(1.0e-3, 2.5, 80)
SENTINEL_TOL = 1.0e-8

ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]




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


def _anec_along(metric, b: float, span: float) -> float:
    res = anec(
        metric,
        _axial_ray(b),
        tangent_norm=TANGENT_NORM,
        # Fixed step density: the sample count scales with the span, so a
        # longer window does not silently coarsen the quadrature.
        n_samples=int(round(N_SAMPLES * span / SPAN0)),
        affine_bounds=(0.0, span),
    )
    return float(res.line_integral)


# Same truncation radius and doubling margin as run_anec_symplectic.py. This
# table sits beside the geodesic one in the paper, so the two must share a window
# rule; they did not. This script used converged_window, "double until the
# on-axis integral is stationary", which is exactly the rule the geodesic run
# had to abandon, because past the crossing a longer window adds no physics and
# does add drift, so stationarity is reached before the crossing is covered.
WALL_SUPPORT_R = 3.0
PROBE_SPAN = 128.0
N_PROBE = 4096


def _measure_span(metric) -> tuple[float, bool]:
    """Affine span covering the crossing, from the ray's own trajectory.

    The path here is analytic, ``x^mu(lam) = (lam, X_START + lam, b, 0)`` with
    the bubble centre at ``x_s = v_s lam``, so ``r_s(lam)`` is closed form and
    needs no integration to measure.
    """
    b0 = float(B_SCAN[0])
    lam = np.linspace(0.0, PROBE_SPAN, N_PROBE)
    r_s = np.sqrt(((1.0 - V_S) * lam + X_START) ** 2 + b0**2)
    return crossing_span(lam, r_s, WALL_SUPPORT_R)


def _minkowski_sentinel() -> float:
    worst = 0.0
    for b in (1.0e-3, 0.5, 1.0, 1.5):
        worst = max(worst, abs(_anec_along(MinkowskiMetric(), b, SPAN0)))
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
        metric = instantiate(name, V_S, R_B, SIGMA)
        span, span_converged = _measure_span(metric)
        on_axis = _anec_along(metric, float(B_SCAN[0]), span)
        scan = np.array([_anec_along(metric, float(b), span) for b in B_SCAN])
        j = int(np.argmin(scan))
        per_metric[name] = {
            "on_axis": on_axis,
            "min_line_integral": float(scan[j]),
            "b_at_min": float(B_SCAN[j]),
            "b_bracketed": bool(0 < j < len(B_SCAN) - 1),
            "affine_span": float(span),
            "affine_span_covers_crossing": bool(span_converged),
            "max_line_integral": float(scan.max()),
            "b_scan": B_SCAN.tolist(),
            "line_integral_scan": scan.tolist(),
        }
        print(f"  {name:16s} on-axis={on_axis:+.4e}  "
              f"min={scan[j]:+.4e} @ b={B_SCAN[j]:.3f}  max={scan.max():+.3e}  "
              f"span={span:.1f}{'' if span_converged else ' [RAY DID NOT LEAVE]'}"
              f"{'' if 0 < j < len(B_SCAN) - 1 else ' [argmin on endpoint]'}",
              flush=True)

    out = {
        "params": {
            "v_s": V_S, "R_b": R_B, "sigma": SIGMA,
            "x_start": X_START, "affine_span_start": SPAN0,
            "n_samples_at_span_start": N_SAMPLES,
            "affine_span_note": (
                "the window is measured per metric from the ray's own trajectory: "
                "out to where it leaves r_s = 3, with a factor-2 margin, the same "
                "rule as run_anec_symplectic.py, so the two ANEC tables in the "
                "paper share a window. This is a quantified truncation margin, not "
                "a support theorem: no bound on T_ab k^a k^b outside r_s = 3 is "
                "computed. See each metric's affine_span"
            ),
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
