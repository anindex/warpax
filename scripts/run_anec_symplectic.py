"""Rigorous geodesic-integrated ANEC via the symplectic integrator (K6b).

For each retained warp metric (Alcubierre, Natário, Van den Broeck, Rodal) at
matched family parameters (R_b = 1, sigma = 8, v_s = 0.5) we integrate the
*actual* null geodesic with the structure-preserving symplectic integrator
(:func:`warpax.averaged.anec.anec_rigorous`) along a fan of axial null rays at
varying perpendicular impact parameter ``b``, and evaluate the ANEC line
integral with an on-cone rigor witness ``max|g(k,k)|``.

This upgrades the coordinate null-ray *diagnostic* of ``run_anec_retained.py``
(K6) to a defensible geodesic-integrated *result*: the witness certifies that
the integrated tangent stayed on the null cone (where the adaptive-RK integrator
would drift off it for long crossings). Where the witness exceeds tolerance the
projection-corrected fallback value is recorded and flagged.

The Minkowski ray integrates to zero (and witness to ~0) and is retained as a
sentinel.

Outputs:
- ../results/anec/retained_symplectic.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.averaged.anec import anec_rigorous
from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results", "anec")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

V_S, R_B, SIGMA = 0.5, 1.0, 8.0
X_START = -8.0
AFFINE_SPAN = 16.0
NUM_STEPS = 8192
ORDER = 4
# g(k,k) < 1e-6 certifies the tangent as null to 6 digits; the ANEC integrand
# T_ab k^a k^b is O(0.01-1), so this off-cone budget is negligible. Smooth-wall
# drives clear it comfortably; the Natário oscillatory wall does not and takes
# the projection-corrected fallback (reported and flagged).
NULL_TOL = 1e-6
# Impact parameters: dense near the wall (r_s ~ R_b = 1).
B_SCAN = np.linspace(1.0e-3, 2.5, 30)
SENTINEL_TOL = 1.0e-6

METRICS = {
    "Alcubierre": (AlcubierreMetric, {}),
    "Natário": (NatarioMetric, {}),
    "Van den Broeck": (
        VanDenBroeckMetric,
        {"R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0},
    ),
    "Rodal": (RodalMetric, {}),
}
METRIC_ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]


def _instantiate(name: str):
    cls, extra = METRICS[name]
    return cls(v_s=V_S, R=R_B, sigma=SIGMA, **extra)


def _rigorous_at(metric, b: float):
    x0 = jnp.array([0.0, X_START, b, 0.0], dtype=jnp.float64)
    return anec_rigorous(
        metric, x0, jnp.array([1.0, 0.0, 0.0]),
        affine_bounds=(0.0, AFFINE_SPAN),
        num_steps=NUM_STEPS, order=ORDER, null_tol=NULL_TOL,
    )


def _minkowski_sentinel() -> tuple[float, float]:
    """Return (max |ANEC|, max witness) over a few impact parameters."""
    worst_anec, worst_wit = 0.0, 0.0
    for b in (1.0e-3, 0.5, 1.0, 1.5):
        r = _rigorous_at(MinkowskiMetric(), b)
        worst_anec = max(worst_anec, abs(float(r.symplectic.line_integral)))
        worst_wit = max(worst_wit, float(r.symplectic.max_abs_g_kk))
    return worst_anec, worst_wit


def main() -> None:
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    sent_anec, sent_wit = _minkowski_sentinel()
    print(f"Minkowski sentinel: |ANEC|_max={sent_anec:.2e}  witness_max={sent_wit:.2e}")
    if sent_anec >= SENTINEL_TOL:
        raise RuntimeError(
            f"Minkowski ANEC sentinel {sent_anec:.2e} exceeds tol {SENTINEL_TOL}"
        )
    # The flat-space rays must also stay on the null cone; a regressed
    # integrator that drifts off-cone even in Minkowski would invalidate the
    # on-cone witness reported for the warp metrics below.
    if sent_wit >= NULL_TOL:
        raise RuntimeError(
            f"Minkowski g(k,k) witness {sent_wit:.2e} exceeds tol {NULL_TOL}"
        )

    per_metric: dict[str, dict] = {}
    for name in METRIC_ORDER:
        metric = _instantiate(name)
        anec_scan, witness_scan, preserved_scan, method_scan = [], [], [], []
        proj_scan = []
        for b in B_SCAN:
            r = _rigorous_at(metric, float(b))
            anec_scan.append(float(r.symplectic.line_integral))
            witness_scan.append(float(r.symplectic.max_abs_g_kk))
            preserved_scan.append(bool(r.symplectic.null_preserved))
            method_scan.append(r.method_used)
            proj_scan.append(
                None if r.projection is None
                else float(r.projection.line_integral)
            )
        anec_arr = np.array(anec_scan)
        j = int(np.argmin(anec_arr))
        worst_witness = float(np.max(witness_scan))
        frac_preserved = float(np.mean(preserved_scan))
        per_metric[name] = {
            "on_axis": anec_scan[0],
            "min_line_integral": float(anec_arr[j]),
            "b_at_min": float(B_SCAN[j]),
            "max_line_integral": float(anec_arr.max()),
            "worst_witness_g_kk": worst_witness,
            "fraction_null_preserved": frac_preserved,
            "all_null_preserved": bool(all(preserved_scan)),
            "b_scan": B_SCAN.tolist(),
            "line_integral_scan": anec_scan,
            "witness_scan": witness_scan,
            "method_scan": method_scan,
            "projection_scan": proj_scan,
        }
        flag = "" if all(preserved_scan) else " [some rays needed projection]"
        print(f"  {name:16s} on-axis={anec_scan[0]:+.4e}  "
              f"min={anec_arr[j]:+.4e} @ b={B_SCAN[j]:.3f}  "
              f"worst|g(k,k)|={worst_witness:.2e}{flag}")

    out = {
        "params": {
            "v_s": V_S, "R_b": R_B, "sigma": SIGMA,
            "x_start": X_START, "affine_span": AFFINE_SPAN,
            "num_steps": NUM_STEPS, "order": ORDER, "null_tol": NULL_TOL,
            "integrator": "symplectic (Tao 2016 extended phase space, Yoshida-4)",
        },
        "minkowski_sentinel_abs": sent_anec,
        "minkowski_sentinel_witness": sent_wit,
        "order": METRIC_ORDER,
        "metrics": per_metric,
    }
    out_path = os.path.join(RESULTS_DIR, "retained_symplectic.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")

    # Paper table: rigorous geodesic ANEC + on-cone rigor witness.
    def _w(b):
        return ("symplectic" if b else "fallback")
    tlines = [
        r"\begin{tabular}{@{}l rr c l@{}}",
        r"  \toprule",
        r"  Metric & on-axis & min ($b^\ast$) & $\max|g(k,k)|$ & method \\",
        r"  \midrule",
    ]
    for name in METRIC_ORDER:
        m = per_metric[name]
        tlines.append(
            f"  {name} & ${m['on_axis']:+.4f}$ & "
            f"${m['min_line_integral']:+.4f}$ (${m['b_at_min']:.2f}$) & "
            f"${m['worst_witness_g_kk']:.1e}$ & {_w(m['all_null_preserved'])} \\\\"
        )
    tlines += [r"  \bottomrule", r"\end{tabular}"]
    tab_path = os.path.join(TABLES_DIR, "anec_symplectic.tex")
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(tab_path, "w") as f:
        f.write("\n".join(tlines) + "\n")
    print(f"Wrote {tab_path}")


if __name__ == "__main__":
    main()
