"""Rigorous geodesic-integrated ANEC via the symplectic integrator.

For each retained warp metric (Alcubierre, Natário, Van den Broeck, Rodal) at
matched family parameters (R_b = 1, sigma = 8, v_s = 0.5) we integrate the
*actual* null geodesic with the structure-preserving symplectic integrator
(:func:`warpax.averaged.anec.anec_rigorous`) along a fan of axial null rays at
varying perpendicular impact parameter ``b``, and evaluate the ANEC line
integral with an on-cone rigor witness ``max|g(k,k)|``.

This upgrades the coordinate null-ray *diagnostic* of ``run_anec_retained.py``
to a defensible geodesic-integrated *result*: the witness certifies that
the integrated tangent stayed on the null cone (where the adaptive-RK integrator
would drift off it for long crossings). Where the witness exceeds tolerance the
projection-corrected fallback value is recorded and flagged.

The Minkowski ray integrates to zero (and witness to ~0) and is retained as a
sentinel.

Outputs:
- ../results/anec/retained_symplectic.json
"""
from __future__ import annotations

import os
from pathlib import Path

from _anec_window import crossing_span
from _json_io import dump_json
from _json_io import write_table as write_tex_table
from _paper_metrics import METRIC_ORDER, instantiate

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.averaged.anec import anec_rigorous, null_ic_canonical
from warpax.benchmarks import MinkowskiMetric
from warpax.geodesics import eulerian_affine_scale, integrate_geodesic_symplectic

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results", "anec")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

V_S, R_B, SIGMA = 0.5, 1.0, 8.0
X_START = -8.0
# The ray must clear the bubble, not merely reach it. The published span 16.0 =
# 2|X_START| forgets that the centre recedes at v_s, so it stopped at the centre
# and integrated only the rear half (ratio 2.0000 for Alcubierre, VdB, Rodal).
# Natario was unaffected: its exterior shift drags the ray clear inside the same
# span. The window is read off the geodesic now (_measure_span below).
SPAN0 = 16.0       # reference span for the step density
# 8192 left one ray of 50 (Natario, b = 0.307) at |g(k,k)| = 1.4e-05, which
# downgraded the whole row to the projection fallback. 32768 clears every ray by
# two orders; the line integrals move in the 5th digit.
NUM_STEPS = 32768  # at SPAN0; scaled with the span so the step density is fixed
ORDER = 4
# g(k,k) < 1e-6 certifies the tangent as null to 6 digits; the ANEC integrand
# T_ab k^a k^b is O(0.01-1), so this off-cone budget is negligible. A ray that
# misses it takes the projection-corrected fallback, reported and flagged.
NULL_TOL = 1e-6
# Impact parameters, dense near the wall (r_s ~ R_b = 1). The upper end was 2.5
# and Rodal's minimum sat on it; "b_bracketed" below records interiority.
B_SCAN = np.linspace(1.0e-3, 5.0, 50)
SENTINEL_TOL = 1.0e-6
# The coarse scan has db = 0.102, and a minimum narrower than that sitting between
# two nodes is missed: on Natario it is, and the coarse grid understates the deepest
# line integral by a factor of two. So the argmin bracket is refined until the
# minimum stops moving, and both the coarse and the refined values are recorded.
# The witness is carried through the refinement because the refined Natario minimum
# is a narrow feature whose off-cone deviation is orders worse than its neighbours';
# a deeper value on a worse-conditioned ray is not straightforwardly a better number.
B_REFINE_POINTS = 21
B_REFINE_LEVELS = 4
B_REFINE_RTOL = 1.0e-4





def _affine_scale(metric, x0) -> float:
    """Factor pinning the free null scale to the common normalization -g(k,n)=1.

    ``null_ic`` solves only ``k^0`` and passes the spatial direction through, so
    the affine parameter of a null geodesic is left free; the ANEC line integral
    scales linearly under ``k -> c k`` and its magnitude is therefore undefined
    until this is fixed. We pin it against the Eulerian normal (unit timelike at
    every warp speed) on the stated initial surface.

    This is not a formality. Alcubierre, Van den Broeck and Rodal are written in
    the lab frame (shift vanishing at infinity) and already satisfy
    ``-g(k,n) = 1`` for a unit seed, so their reported values are unchanged. The
    Natario shift follows Natario's own bubble-at-rest convention and tends to
    ``-v_s x_hat`` at infinity, giving ``-g(k,n) = 1/(1 + v_s) = 2/3`` at
    ``v_s = 1/2``; its ANEC magnitude was therefore on a different footing from
    the others by exactly ``3/2`` and is corrected here. (The factor is
    ``1/(1 + v_s)``, not ``1 - v_s``; they coincide only at ``v_s = 1/2``.)
    """
    return float(eulerian_affine_scale(metric, x0))


def _rigorous_at(metric, b: float, span: float):
    x0 = jnp.array([0.0, X_START, b, 0.0], dtype=jnp.float64)
    s = _affine_scale(metric, x0)
    # Rescale the tangent AND shrink the affine window by the same factor, so the
    # geodesic covers an identical coordinate path and only its parametrization
    # (hence the reported magnitude) is pinned.
    return anec_rigorous(
        metric, x0, jnp.array([s, 0.0, 0.0]),
        affine_bounds=(0.0, span / s),
        # Fixed step density.
        num_steps=int(round(NUM_STEPS * span / SPAN0)),
        num_save=None,  # quadrature nodes = every step
        order=ORDER, null_tol=NULL_TOL,
        # K = d_t + v_s d_x is Killing; E_K = -p_a K^a is the second witness.
        killing=jnp.array([1.0, V_S, 0.0, 0.0], dtype=jnp.float64),
    )


def _refine_min(metric, span, b_lo: float, b_hi: float):
    """Refine the b-scan minimum inside [b_lo, b_hi] until it stops moving.

    Returns (b, value, witness, killing_drift, history, converged). Item A4 asks for
    convergence of the impact-parameter search, not only of the integral along each
    ray; this supplies it, and the history is what makes the claim checkable.
    """
    best = None
    history: list[dict] = []
    converged = False
    for level in range(B_REFINE_LEVELS):
        grid = np.linspace(b_lo, b_hi, B_REFINE_POINTS)
        recs = []
        for b in grid:
            r = _rigorous_at(metric, float(b), span)
            recs.append((
                float(r.symplectic.line_integral),
                float(r.symplectic.max_abs_g_kk),
                float(r.killing_drift),
            ))
        vals = np.array([v for v, _, _ in recs])
        k = int(np.argmin(vals))
        history.append({
            "level": level + 1,
            "db": float(grid[1] - grid[0]),
            "b": float(grid[k]),
            "line_integral": float(vals[k]),
            "witness_g_kk": recs[k][1],
            "killing_drift": recs[k][2],
            "interior": bool(0 < k < len(grid) - 1),
        })
        if best is not None and abs(vals[k] - best[1]) <= B_REFINE_RTOL * abs(best[1]):
            best = (float(grid[k]), float(vals[k]), recs[k][1], recs[k][2])
            converged = True
            break
        best = (float(grid[k]), float(vals[k]), recs[k][1], recs[k][2])
        b_lo, b_hi = float(grid[max(k - 1, 0)]), float(grid[min(k + 1, len(grid) - 1)])
    return (*best, history, converged)


# tail_bound certifies the shape function below 1.3e-14 outside this radius. That
# bounds f, not T_ab k^a k^b: a truncation margin, not a support theorem.
WALL_SUPPORT_R = 3.0
PROBE_SPAN = 128.0


def _measure_span(metric) -> tuple[float, bool]:
    """Affine span covering the crossing, read off the geodesic itself."""
    b0 = float(B_SCAN[0])
    x0 = jnp.array([0.0, X_START, b0, 0.0], dtype=jnp.float64)
    sc = _affine_scale(metric, x0)
    x0c, p0 = null_ic_canonical(metric, x0, jnp.array([sc, 0.0, 0.0]))
    geo = integrate_geodesic_symplectic(
        metric, x0c, p0, (0.0, PROBE_SPAN / sc),
        num_steps=int(round(NUM_STEPS * PROBE_SPAN / SPAN0)), order=ORDER,
    )
    pos = np.asarray(geo.positions)
    lam = np.asarray(geo.ts) * sc
    r_s = np.sqrt((pos[:, 1] - V_S * pos[:, 0]) ** 2 + pos[:, 2] ** 2 + pos[:, 3] ** 2)
    return crossing_span(lam, r_s, WALL_SUPPORT_R)


def _minkowski_sentinel() -> tuple[float, float]:
    """Return (max |ANEC|, max witness) over a few impact parameters."""
    worst_anec, worst_wit = 0.0, 0.0
    for b in (1.0e-3, 0.5, 1.0, 1.5):
        r = _rigorous_at(MinkowskiMetric(), b, SPAN0)
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
        metric = instantiate(name, V_S, R_B, SIGMA)
        span, span_converged = _measure_span(metric)
        print(f"  {name:16s} affine window {span:.1f} "
              f"({'crossing covered' if span_converged else 'RAY DID NOT LEAVE'})", flush=True)
        anec_scan, witness_scan, preserved_scan, method_scan = [], [], [], []
        proj_scan, killing_scan = [], []
        for b in B_SCAN:
            r = _rigorous_at(metric, float(b), span)
            anec_scan.append(float(r.symplectic.line_integral))
            witness_scan.append(float(r.symplectic.max_abs_g_kk))
            preserved_scan.append(bool(r.symplectic.null_preserved))
            method_scan.append(r.method_used)
            killing_scan.append(r.killing_drift)
            proj_scan.append(
                None if r.projection is None
                else float(r.projection.line_integral)
            )
        anec_arr = np.array(anec_scan)
        j = int(np.argmin(anec_arr))
        b_ref, v_ref, w_ref, k_ref, ref_hist, ref_conv = _refine_min(
            metric, span,
            float(B_SCAN[max(j - 1, 0)]),
            float(B_SCAN[min(j + 1, len(B_SCAN) - 1)]),
        )
        worst_witness = float(max(np.max(witness_scan), w_ref))
        worst_killing = float(max(np.max(killing_scan), k_ref))
        frac_preserved = float(np.mean(preserved_scan))
        per_metric[name] = {
            "affine_scale_to_unit_eulerian_frequency": _affine_scale(
                metric, jnp.array([0.0, X_START, 0.0, 0.0], dtype=jnp.float64)
            ),
            "on_axis": anec_scan[0],
            # The reported minimum is the refined one: the coarse grid is too wide to
            # resolve it on every drive.
            "min_line_integral": v_ref,
            "b_at_min": b_ref,
            "min_line_integral_coarse": float(anec_arr[j]),
            "b_at_min_coarse": float(B_SCAN[j]),
            "refinement_deepening_rel": float(
                (v_ref - anec_arr[j]) / abs(anec_arr[j])
            ) if anec_arr[j] != 0 else None,
            "refinement_witness_g_kk": w_ref,
            "refinement_killing_drift": k_ref,
            "refinement_converged": bool(ref_conv),
            "refinement_history": ref_hist,
            # An argmin at an endpoint is not a minimum. Record it rather than let a
            # reader assume the scan bracketed the extremum.
            "b_bracketed": bool(0 < j < len(B_SCAN) - 1),
            "affine_span": float(span),
            "affine_span_covers_crossing": bool(span_converged),
            "max_line_integral": float(anec_arr.max()),
            "worst_witness_g_kk": worst_witness,
            "worst_killing_energy_drift": worst_killing,
            "fraction_null_preserved": frac_preserved,
            "all_null_preserved": bool(all(preserved_scan)),
            "b_scan": B_SCAN.tolist(),
            "line_integral_scan": anec_scan,
            "witness_scan": witness_scan,
            "killing_drift_scan": killing_scan,
            "method_scan": method_scan,
            "projection_scan": proj_scan,
        }
        flag = "" if all(preserved_scan) else " [some rays needed projection]"
        deep = (v_ref - anec_arr[j]) / abs(anec_arr[j]) * 100.0 if anec_arr[j] else 0.0
        print(f"  {name:16s} on-axis={anec_scan[0]:+.4e}  "
              f"coarse min={anec_arr[j]:+.4e} @ b={B_SCAN[j]:.3f}  "
              f"refined={v_ref:+.4e} @ b={b_ref:.4f} ({deep:+.1f}%, "
              f"{'converged' if ref_conv else 'NOT CONVERGED'}, "
              f"|g(k,k)|={w_ref:.1e})  "
              f"worst|g(k,k)|={worst_witness:.2e}  "
              f"worst dE_K/E_K={worst_killing:.2e}{flag}")

    out = {
        "params": {
            "v_s": V_S, "R_b": R_B, "sigma": SIGMA,
            "x_start": X_START, "affine_span_start": SPAN0,
            "affine_span_note": (
                "the window is measured per metric from the geodesic's own "
                "trajectory: out to where it leaves r_s = 3, the radius beyond "
                "which tail_bound certifies the shape function below 1.3e-14, "
                "with a factor-2 margin; see each metric's affine_span. This is "
                "a quantified truncation margin, not a support theorem: no bound "
                "on T_ab k^a k^b outside r_s = 3 is computed"
            ),
            "num_steps_at_span_start": NUM_STEPS, "order": ORDER, "null_tol": NULL_TOL,
            "quadrature_nodes": "every symplectic step (num_save = num_steps + 1)",
            "killing_vector": [1.0, V_S, 0.0, 0.0],
            "integrator": "symplectic (Tao 2016 extended phase space, Yoshida-4)",
        },
        "minkowski_sentinel_abs": sent_anec,
        "minkowski_sentinel_witness": sent_wit,
        "order": METRIC_ORDER,
        "metrics": per_metric,
    }
    out_path = os.path.join(RESULTS_DIR, "retained_symplectic.json")
    dump_json(out, out_path)
    print(f"Wrote {out_path}")

    # Paper table: rigorous geodesic ANEC + on-cone rigor witness.
    def _w(b):
        return ("symplectic" if b else "fallback")
    tlines = [
        r"\begin{tabular}{@{}l rr cc l@{}}",
        r"  \toprule",
        r"  Metric & on-axis & min ($b^\ast$) & $\max|g(k,k)|$"
        r" & $\max|\Delta E_K/E_K|$ & method \\",
        r"  \midrule",
    ]
    for name in METRIC_ORDER:
        m = per_metric[name]
        tlines.append(
            f"  {name} & ${m['on_axis']:+.4f}$ & "
            f"${m['min_line_integral']:+.4f}$ (${m['b_at_min']:.2f}$) & "
            f"${m['worst_witness_g_kk']:.1e}$ & "
            f"${m['worst_killing_energy_drift']:.1e}$ & "
            f"{_w(m['all_null_preserved'])} \\\\"
        )
    tlines += [r"  \bottomrule", r"\end{tabular}"]
    tab_path = os.path.join(TABLES_DIR, "anec_symplectic.tex")
    os.makedirs(TABLES_DIR, exist_ok=True)
    write_tex_table(tab_path, tlines, script="scripts/run_anec_symplectic.py",
                    sources="results/anec/retained_symplectic.json")
    print(f"Wrote {tab_path}")


if __name__ == "__main__":
    main()
