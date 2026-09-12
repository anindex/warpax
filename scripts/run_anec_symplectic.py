"""Finite-segment null-geodesic diagnostics at fixed affine normalization.

The impact-parameter search is basin-local. Selected near-axis and minimum-found
rays are recomputed at at least three step densities with fixed endpoints and
impact parameter; unstable rays receive further step doubling. Constraint drift
and observed spreads do not bound omitted tails or trajectory error and do not
establish complete-geodesic ANEC.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
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
SPAN0 = 16.0  # reference span for the step density
NUM_STEPS = 32768  # at SPAN0; scaled with the span so the step density is fixed
ORDER = 4
NULL_TOL = 1e-6
# Impact parameters, dense near the wall (r_s ~ R_b = 1). The upper end was 2.5
# and Rodal's minimum sat on it; "b_bracketed" below records interiority.
B_SCAN = np.linspace(1.0e-3, 5.0, 50)
SENTINEL_TOL = 1.0e-6
# The coarse scan has db = 0.102 and misses a narrower minimum between nodes, by a
# factor of two on Natario, so the argmin bracket is refined until it stops moving.
# Both values are kept, with the witness, since the refined ray is worse conditioned.
B_REFINE_POINTS = 21
B_REFINE_LEVELS = 4
B_REFINE_RTOL = 1.0e-4
SELECTED_MAX_LEVELS = 6
STEP_ATOL = 1.0e-8
STEP_RTOL = 1.0e-4


def _atomic_json(data: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    dump_json(data, temporary)
    temporary.replace(path)


class Checkpoint:
    """Resume only results with identical configuration and source bytes."""

    def __init__(self, path: Path, provenance: dict):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text())
            if self.data["provenance"] != provenance:
                raise ValueError(f"Stale ray checkpoint: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.data = {"provenance": provenance, "records": {}}

    def save(self, key: str, value):
        self.data["records"][key] = value
        _atomic_json(self.data, self.path)
        return value

    def get(self, key: str, compute):
        if key not in self.data["records"]:
            self.save(key, compute())
        return self.data["records"][key]


def _provenance() -> dict:
    root = Path(HERE).resolve().parent
    sources = [
        *sorted((root / "src" / "warpax").rglob("*.py")),
        Path(__file__).resolve(),
        root / "scripts/_paper_metrics.py",
        root / "scripts/_anec_window.py",
        root / "scripts/_json_io.py",
    ]
    return {
        "schema": 1,
        "sources_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "equinox", "diffrax", "interpax")
        },
        "configuration": {
            "v_s": V_S,
            "R_b": R_B,
            "sigma": SIGMA,
            "x_start": X_START,
            "reference_span": SPAN0,
            "base_steps": NUM_STEPS,
            "order": ORDER,
            "null_tol": NULL_TOL,
            "b_scan": B_SCAN.tolist(),
            "b_refine_points": B_REFINE_POINTS,
            "b_refine_levels": B_REFINE_LEVELS,
            "b_refine_rtol": B_REFINE_RTOL,
            "selected_max_levels": SELECTED_MAX_LEVELS,
            "step_atol": STEP_ATOL,
            "step_rtol": STEP_RTOL,
            "wall_support_r": WALL_SUPPORT_R,
            "probe_span": PROBE_SPAN,
            "metric_parameters": {
                name: repr(instantiate(name, V_S, R_B, SIGMA)) for name in METRIC_ORDER
            },
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": True,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
        },
    }


def _ray_record(metric, b, span, base_steps=NUM_STEPS, checkpoint=None, name=""):
    def compute():
        result = _rigorous_at(metric, b, span, base_steps)
        return {
            "steps_per_reference_span": base_steps,
            "num_steps": int(round(base_steps * span / SPAN0)),
            "line_integral": float(result.symplectic.line_integral),
            "max_abs_g_kk": float(result.symplectic.max_abs_g_kk),
            "killing_drift": float(result.killing_drift),
            "null_preserved": bool(result.symplectic.null_preserved),
            "geodesic_complete": bool(result.symplectic.geodesic_complete),
            "method": result.method_used,
            "projection": None
            if result.projection is None
            else float(result.projection.line_integral),
        }

    if checkpoint is None:
        return compute()
    key = f"{name}:ray:{float(b).hex()}:{float(span).hex()}:{base_steps}"
    return checkpoint.get(key, compute)


def _affine_scale(metric, x0) -> float:
    """Factor pinning the free null scale to the common normalization -g(k,n)=1.

    ``null_ic`` solves only ``k^0`` and passes the spatial direction through, so
    the affine parameter of a null geodesic is left free; the ANEC line integral
    scales linearly under ``k -> c k`` and its magnitude is therefore undefined
    until this is fixed. We pin it against the Eulerian normal (unit timelike at
    every warp speed) on the stated initial surface.

    """
    return float(eulerian_affine_scale(metric, x0))


def _rigorous_at(metric, b: float, span: float, base_steps: int = NUM_STEPS):
    x0 = jnp.array([0.0, X_START, b, 0.0], dtype=jnp.float64)
    s = _affine_scale(metric, x0)
    # Rescale the tangent AND shrink the affine window by the same factor, so the
    # geodesic covers an identical coordinate path and only its parametrization
    # (hence the reported magnitude) is pinned.
    return anec_rigorous(
        metric,
        x0,
        jnp.array([s, 0.0, 0.0]),
        affine_bounds=(0.0, span / s),
        # Fixed step density.
        num_steps=int(round(base_steps * span / SPAN0)),
        num_save=None,  # quadrature nodes = every step
        order=ORDER,
        null_tol=NULL_TOL,
        # K = d_t + v_s d_x is Killing; E_K = -p_a K^a is the second witness.
        killing=jnp.array([1.0, V_S, 0.0, 0.0], dtype=jnp.float64),
    )


def _finite_argmin(values, eligible):
    valid = np.array(
        [
            bool(ok) and value is not None and np.isfinite(value)
            for value, ok in zip(values, eligible, strict=True)
        ]
    )
    if not np.any(valid):
        raise RuntimeError("No completed finite null ray in this basin")
    return int(
        np.argmin([value if ok else np.inf for value, ok in zip(values, valid, strict=True)])
    )


def _coarse_basin(values, eligible):
    """Select an interior finite local minimum with two completed neighbours."""
    finite = [
        bool(ok) and value is not None and np.isfinite(value)
        for value, ok in zip(values, eligible, strict=True)
    ]
    bracketed = [False] * len(values)
    for i in range(1, len(values) - 1):
        bracketed[i] = (
            all(finite[i - 1 : i + 2]) and values[i] <= values[i - 1] and values[i] <= values[i + 1]
        )
    return _finite_argmin(values, bracketed)


def _refine_min(metric, span, b_lo: float, b_hi: float, checkpoint=None, name=""):
    """Refine the b-scan minimum inside [b_lo, b_hi] until it stops moving.

    Returns (b, value, witness, killing_drift, history, converged). Item A3 asks for
    convergence of the impact-parameter search, not only of the integral along each
    ray; this supplies it, and the history is what makes the claim checkable.
    """
    best = None
    history: list[dict] = []
    converged = False
    for level in range(B_REFINE_LEVELS):
        grid = np.linspace(b_lo, b_hi, B_REFINE_POINTS)
        recs = []
        eligible = []
        for b in grid:
            r = _ray_record(metric, float(b), span, checkpoint=checkpoint, name=name)
            eligible.append(r["geodesic_complete"] and r["null_preserved"])
            recs.append(
                (
                    r["line_integral"],
                    r["max_abs_g_kk"],
                    r["killing_drift"],
                )
            )
        vals = [v for v, _, _ in recs]
        k = _finite_argmin(vals, eligible)
        history.append(
            {
                "level": level + 1,
                "db": float(grid[1] - grid[0]),
                "b": float(grid[k]),
                "line_integral": float(vals[k]),
                "witness_g_kk": recs[k][1],
                "killing_drift": recs[k][2],
                "interior": bool(0 < k < len(grid) - 1),
                "excluded_rays": len(grid) - sum(eligible),
            }
        )
        if best is not None and abs(vals[k] - best[1]) <= B_REFINE_RTOL * abs(best[1]):
            best = (float(grid[k]), float(vals[k]), recs[k][1], recs[k][2])
            converged = True
            break
        best = (float(grid[k]), float(vals[k]), recs[k][1], recs[k][2])
        b_lo, b_hi = float(grid[max(k - 1, 0)]), float(grid[min(k + 1, len(grid) - 1)])
    return (*best, history, converged)


def selected_ray_convergence(metric, b: float, span: float, checkpoint=None, name="") -> dict:
    """Refine a fixed ray, retaining each level and the unchanged tolerance."""
    records = []
    summary = None
    for level in range(SELECTED_MAX_LEVELS):
        base_steps = (NUM_STEPS // 2) * 2**level
        record = _ray_record(metric, b, span, base_steps, checkpoint, name)
        records.append(record)
        print(
            f"    {name} b={b:.17g} steps={base_steps}: "
            f"I={record['line_integral']!s}, "
            f"|g(k,k)|={record['max_abs_g_kk']!s}, "
            f"dE/E={record['killing_drift']!s}",
            flush=True,
        )
        values = [r["line_integral"] for r in records]
        finite = all(v is not None and np.isfinite(v) for v in values)
        finest_change = abs(values[-1] - values[-2]) if finite and len(values) > 1 else None
        stable = bool(
            len(records) >= 3
            and finite
            and finest_change <= STEP_ATOL + STEP_RTOL * abs(values[-1])
        )
        summary = {
            "b": b,
            "span": span,
            "records": records,
            "observed_spread": max(values) - min(values) if finite else None,
            "finest_change": finest_change,
            "absolute_tolerance": STEP_ATOL,
            "relative_tolerance": STEP_RTOL,
            "step_stable": stable,
        }
        if checkpoint is not None:
            checkpoint.save(f"{name}:selected:{float(b).hex()}:{float(span).hex()}", summary)
        if stable:
            break
    return summary


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
        metric,
        x0c,
        p0,
        (0.0, PROBE_SPAN / sc),
        num_steps=int(round(NUM_STEPS * PROBE_SPAN / SPAN0)),
        order=ORDER,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first", choices=METRIC_ORDER, help="Evaluate this metric first; all are retained."
    )
    args = parser.parse_args()
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    provenance = _provenance()
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    checkpoint = Checkpoint(Path(RESULTS_DIR) / "checkpoints" / f"{digest}.json", provenance)
    print(
        f"Ray checkpoint: {checkpoint.path} ({len(checkpoint.data['records'])} saved entries)",
        flush=True,
    )

    sent_anec, sent_wit = checkpoint.get("minkowski_sentinel", _minkowski_sentinel)
    print(f"Minkowski sentinel: |ANEC|_max={sent_anec:.2e}  witness_max={sent_wit:.2e}")
    if sent_anec >= SENTINEL_TOL:
        raise RuntimeError(f"Minkowski ANEC sentinel {sent_anec:.2e} exceeds tol {SENTINEL_TOL}")
    # The flat-space rays must also stay on the null cone; a regressed
    # integrator that drifts off-cone even in Minkowski would invalidate the
    # on-cone witness reported for the warp metrics below.
    if sent_wit >= NULL_TOL:
        raise RuntimeError(f"Minkowski g(k,k) witness {sent_wit:.2e} exceeds tol {NULL_TOL}")

    per_metric: dict[str, dict] = {}
    execution_order = ([args.first] if args.first else []) + [
        name for name in METRIC_ORDER if name != args.first
    ]
    for name in execution_order:
        metric = instantiate(name, V_S, R_B, SIGMA)
        span, span_converged = checkpoint.get(f"{name}:span", lambda: _measure_span(metric))
        print(
            f"  {name:16s} affine window {span:.1f} "
            f"({'crossing covered' if span_converged else 'RAY DID NOT LEAVE'})",
            flush=True,
        )
        anec_scan, witness_scan, preserved_scan, method_scan = [], [], [], []
        proj_scan, killing_scan, complete_scan = [], [], []
        for b in B_SCAN:
            r = _ray_record(metric, float(b), span, checkpoint=checkpoint, name=name)
            anec_scan.append(r["line_integral"])
            witness_scan.append(r["max_abs_g_kk"])
            preserved_scan.append(r["null_preserved"])
            method_scan.append(r["method"])
            killing_scan.append(r["killing_drift"])
            proj_scan.append(r["projection"])
            complete_scan.append(r["geodesic_complete"])
        anec_arr = np.array([np.nan if v is None else v for v in anec_scan])
        eligible_scan = [
            complete and preserved
            for complete, preserved in zip(complete_scan, preserved_scan, strict=True)
        ]
        j = _coarse_basin(anec_scan, eligible_scan)
        failed_indices = [i for i, ok in enumerate(eligible_scan) if not ok]
        checkpoint.save(
            f"{name}:scan_summary",
            {
                "excluded_indices": failed_indices,
                "b_bracket": B_SCAN[j - 1 : j + 2].tolist(),
            },
        )
        print(
            f"  {name}: {len(failed_indices)}/{len(B_SCAN)} scan rays excluded "
            f"(incomplete or failed null check); selected finite basin "
            f"[{B_SCAN[j - 1]:.17g}, {B_SCAN[j + 1]:.17g}]",
            flush=True,
        )
        b_ref, v_ref, w_ref, k_ref, ref_hist, ref_conv = _refine_min(
            metric,
            span,
            float(B_SCAN[max(j - 1, 0)]),
            float(B_SCAN[min(j + 1, len(B_SCAN) - 1)]),
            checkpoint,
            name,
        )
        selected = {
            "near_axis": selected_ray_convergence(metric, float(B_SCAN[0]), span, checkpoint, name),
            "minimum_found": selected_ray_convergence(metric, b_ref, span, checkpoint, name),
        }
        checkpoint.save(
            f"{name}:refinement", {"b": b_ref, "history": ref_hist, "converged": ref_conv}
        )
        if not all(ray["step_stable"] for ray in selected.values()):
            print(json.dumps(selected, indent=2), flush=True)
            raise RuntimeError(
                f"{name}: selected-ray step refinement has not stabilized; "
                f"all levels retained in {checkpoint.path}"
            )
        basin_result = {
            "b": b_ref,
            "history": ref_hist,
            "converged": ref_conv,
            "selected_ray_convergence": selected["minimum_found"],
        }
        minimum_bracketed = True
        coarse_lowest = _finite_argmin(anec_scan, eligible_scan)
        outer_candidate = None
        if coarse_lowest != j:
            outer_candidate = selected_ray_convergence(
                metric, float(B_SCAN[coarse_lowest]), span, checkpoint, name
            )
            checkpoint.save(f"{name}:lowest_coarse_candidate", outer_candidate)
            outer_finest = outer_candidate["records"][-1]
            if (
                outer_candidate["step_stable"]
                and outer_finest["geodesic_complete"]
                and outer_finest["null_preserved"]
                and outer_finest["line_integral"]
                < selected["minimum_found"]["records"][-1]["line_integral"]
            ):
                selected["minimum_found"] = outer_candidate
                b_ref = float(B_SCAN[coarse_lowest])
                minimum_bracketed = False
                ref_conv = False
                print(
                    f"  {name}: lower stable finite coarse ray retained at b={b_ref:.17g}; "
                    "impact-parameter minimum unbracketed",
                    flush=True,
                )
        finest = selected["minimum_found"]["records"][-1]
        v_ref, w_ref, k_ref = (
            finest["line_integral"],
            finest["max_abs_g_kk"],
            finest["killing_drift"],
        )
        selected_records = [record for ray in selected.values() for record in ray["records"]]
        worst_witness = float(
            max(
                *(v for v, ok in zip(witness_scan, eligible_scan, strict=True) if ok),
                *(r["max_abs_g_kk"] for r in selected_records),
            )
        )
        worst_killing = float(
            max(
                *(v for v, ok in zip(killing_scan, eligible_scan, strict=True) if ok),
                *(r["killing_drift"] for r in selected_records),
            )
        )
        frac_preserved = float(np.mean(preserved_scan))
        per_metric[name] = {
            "affine_scale_to_unit_eulerian_frequency": {
                label: _affine_scale(
                    metric, jnp.array([0.0, X_START, ray["b"], 0.0], dtype=jnp.float64)
                )
                for label, ray in selected.items()
            },
            "on_axis": selected["near_axis"]["records"][-1]["line_integral"],
            "selected_ray_convergence": selected,
            "diagnostic_scope": "finite-segment, basin-local minimum found",
            "scan_excluded_count": len(failed_indices),
            "scan_excluded_indices": failed_indices,
            "scan_excluded_b": [float(B_SCAN[i]) for i in failed_indices],
            "scan_completed": complete_scan,
            "scan_null_preserved": preserved_scan,
            "selected_basin": [float(B_SCAN[j - 1]), float(B_SCAN[j + 1])]
            if minimum_bracketed
            else None,
            "interior_basin_tested": [float(B_SCAN[j - 1]), float(B_SCAN[j + 1])],
            "interior_basin_result": basin_result,
            "lowest_coarse_candidate": outer_candidate,
            "basin_selection": "finite interior basin and separately refined lowest finite coarse candidate",
            "constraint_drift_scope": "completed null scan rays and all selected refinement levels",
            "all_selected_null_preserved": all(
                r["null_preserved"] and r["geodesic_complete"] for r in selected_records
            ),
            # The reported minimum is the refined one: the coarse grid is too wide to
            # resolve it on every drive.
            "min_line_integral": v_ref,
            "b_at_min": b_ref,
            "min_line_integral_coarse": float(anec_arr[j]),
            "b_at_min_coarse": float(B_SCAN[j]),
            "refinement_deepening_rel": float((v_ref - anec_arr[j]) / abs(anec_arr[j]))
            if anec_arr[j] != 0
            else None,
            "refinement_witness_g_kk": w_ref,
            "refinement_killing_drift": k_ref,
            "refinement_converged": bool(ref_conv),
            "refinement_history": ref_hist,
            # An argmin at an endpoint is not a minimum. Record it rather than let a
            # reader assume the scan bracketed the extremum.
            "b_bracketed": minimum_bracketed,
            "affine_span": float(span),
            "affine_span_covers_crossing": bool(span_converged),
            "max_line_integral": max(
                v for v, ok in zip(anec_scan, eligible_scan, strict=True) if ok
            ),
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
        checkpoint.save(f"{name}:complete", per_metric[name])
        flag = "" if all(preserved_scan) else " [some rays needed projection]"
        deep = (v_ref - anec_arr[j]) / abs(anec_arr[j]) * 100.0 if anec_arr[j] else 0.0
        print(
            f"  {name:16s} on-axis={anec_scan[0]:+.4e}  "
            f"coarse min={anec_arr[j]:+.4e} @ b={B_SCAN[j]:.3f}  "
            f"refined={v_ref:+.4e} @ b={b_ref:.4f} ({deep:+.1f}%, "
            f"{'converged' if ref_conv else 'NOT CONVERGED'}, "
            f"|g(k,k)|={w_ref:.1e})  "
            f"worst|g(k,k)|={worst_witness:.2e}  "
            f"worst dE_K/E_K={worst_killing:.2e}{flag}"
        )

    out = {
        "provenance": provenance,
        "checkpoint_sha256": digest,
        "params": {
            "v_s": V_S,
            "R_b": R_B,
            "sigma": SIGMA,
            "x_start": X_START,
            "affine_span_start": SPAN0,
            "affine_span_note": (
                "the window is measured per metric from the geodesic's own "
                "trajectory: out to where it leaves r_s = 3, the radius beyond "
                "which tail_bound certifies the shape function below 1.3e-14, "
                "with a factor-2 margin; see each metric's affine_span. This is "
                "a window selection rule, not a stress-tail bound: no bound "
                "on T_ab k^a k^b outside r_s = 3 is computed"
            ),
            "num_steps_at_span_start": NUM_STEPS,
            "order": ORDER,
            "null_tol": NULL_TOL,
            "selected_step_atol": STEP_ATOL,
            "selected_step_rtol": STEP_RTOL,
            "selected_max_levels": SELECTED_MAX_LEVELS,
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
    _atomic_json(out, Path(out_path))
    print(f"Wrote {out_path}")

    # Paper table: finite-segment diagnostics and monitored constraint drifts.
    def _w(b):
        return "symplectic" if b else "fallback"

    tlines = [
        r"\begin{tabular}{@{}l rr cc l@{}}",
        r"  \toprule",
        r"  Metric & $b=0.001$ & minimum found ($b^\ast$) & $\max|g(k,k)|$"
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
            f"{_w(m['all_selected_null_preserved'])} \\\\"
        )
    tlines += [r"  \bottomrule", r"\end{tabular}"]
    tab_path = os.path.join(TABLES_DIR, "anec_symplectic.tex")
    os.makedirs(TABLES_DIR, exist_ok=True)
    write_tex_table(
        tab_path,
        tlines,
        script="scripts/run_anec_symplectic.py",
        sources="results/anec/retained_symplectic.json",
    )
    print(f"Wrote {tab_path}")


if __name__ == "__main__":
    main()
