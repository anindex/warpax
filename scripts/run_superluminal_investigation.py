"""Characterize Alcubierre and Lentz metrics at v_s=1, 1.5, 2.

The unit-lapse, flat-spatial ADM determinant is -1 at every speed.
A positive g_00 makes coordinate-stationary worldlines spacelike;
the Eulerian normal remains timelike and the signature stays Lorentzian.
Save radial diagnostics to results/superluminal_characterization.json
and a summary to results/superluminal_report.md.

Run: python scripts/run_superluminal_investigation.py
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime

import matplotlib
from _json_io import dump_json

matplotlib.use("Agg")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions import classify_hawking_ellis, verify_point
from warpax.geometry import compute_curvature_chain
from warpax.metrics import LentzMetric

# Constants

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

VELOCITIES = [1.0, 1.5, 2.0]

# Shared parameters for both metrics
METRIC_CONFIGS = {
    "alcubierre": {"R": 100.0, "sigma": 8.0},
    "lentz": {"R": 100.0, "sigma": 8.0},
}

# Radial sample points for pointwise analysis.
# Dense near wall (r=R=100) to capture g_00 transition.
# For Lentz: use y=0.01 offset to avoid L1 kink at origin.
N_RADIAL = 50

# Number of radial points for full EC characterization (more expensive).
N_CHARACTERIZE = 10


# Core functions


def compute_g00_transition(metric, v_s, r_values, t=0.0):
    """Compute g_00 along radial axis to locate sign-flip surface.

    Parameters
    ----------
    metric : MetricSpecification
        Warp drive metric instance.
    v_s : float
        Warp velocity parameter.
    r_values : array-like
        Radial positions to sample.
    t : float
        Time coordinate (default 0.0).

    Returns
    -------
    list[dict]
        Per-point diagnostics with g_00, det(g), eigenvalues, shape
        function value, and horizon flag.
    """
    results = []
    for r in r_values:
        coords = jnp.array([t, float(r), 0.01, 0.0])
        g = metric(coords)
        f_val = float(metric.shape_function_value(coords))
        g_00 = float(g[0, 0])
        det_g = float(jnp.linalg.det(g))
        evals_g = jnp.linalg.eigvalsh(g).tolist()
        v_s_times_f = v_s * f_val
        in_horizon = v_s_times_f > 1.0

        results.append(
            {
                "r": float(r),
                "f": f_val,
                "g_00": g_00,
                "det_g": det_g,
                "metric_eigenvalues": evals_g,
                "v_s_times_f": v_s_times_f,
                "in_horizon": in_horizon,
            }
        )
    return results


def characterize_velocity(metric, metric_name, v_s, r_values):
    """Run full EC characterization at selected radial points.

    Parameters
    ----------
    metric : MetricSpecification
        Warp drive metric instance.
    metric_name : str
        Human-readable metric name.
    v_s : float
        Warp velocity parameter.
    r_values : array-like
        Radial positions for characterization.

    Returns
    -------
    list[dict]
        Per-point diagnostics with curvature, HE type, and EC margins.
    """
    results = []
    for r in r_values:
        coords = jnp.array([0.0, float(r), 0.01, 0.0])
        curv = compute_curvature_chain(metric, coords)
        has_nan = bool(jnp.any(jnp.isnan(curv.stress_energy)))
        T_max = float(jnp.max(jnp.abs(curv.stress_energy)))

        T_mixed = curv.metric_inv @ curv.stress_energy
        cls = classify_hawking_ellis(T_mixed, curv.metric)
        he_type = int(float(cls.he_type))

        ec = verify_point(
            curv.stress_energy,
            curv.metric,
            curv.metric_inv,
            n_starts=8,
            zeta_max=5.0,
        )

        results.append(
            {
                "r": float(r),
                "coords": [0.0, float(r), 0.01, 0.0],
                "has_nan_stress_energy": has_nan,
                "T_max": T_max,
                "he_type": he_type,
                "nec_margin": float(ec.nec_margin),
                "wec_margin": float(ec.wec_margin),
                "sec_margin": float(ec.sec_margin),
                "dec_margin": float(ec.dec_margin),
            }
        )
    return results


def _select_characterization_points(r_min, r_max, R, n_points):
    """Select radial points for detailed EC characterization.

    Concentrates points near the bubble center and wall region.

    Parameters
    ----------
    r_min, r_max : float
        Radial range endpoints.
    R : float
        Bubble radius.
    n_points : int
        Number of characterization points.

    Returns
    -------
    np.ndarray
        Selected radial positions.
    """
    # Core regions: center, mid-interior, near-wall, wall, exterior
    center = r_min
    mid_interior = R * 0.5
    near_wall = R - 5.0
    wall = R
    past_wall = R + 5.0
    exterior = r_max

    # Add intermediate points to reach n_points
    base_points = [center, mid_interior, near_wall, wall, past_wall, exterior]
    # Fill gaps
    extras = np.linspace(r_min, r_max, n_points - len(base_points) + 2)[1:-1]
    all_points = sorted(set(base_points) | set(extras.tolist()))
    return np.array(all_points[:n_points])


def analyze_metric(metric_cls, metric_name, config):
    """Run superluminal investigation for a single metric family.

    Parameters
    ----------
    metric_cls : type
        Metric class (AlcubierreMetric or LentzMetric).
    metric_name : str
        Name for output labeling.
    config : dict
        Metric construction keyword arguments (R, sigma).

    Returns
    -------
    dict
        Per-velocity results.
    """
    R = config["R"]
    r_values = np.linspace(0.1, 200.0, N_RADIAL)
    char_points = _select_characterization_points(0.1, 200.0, R, N_CHARACTERIZE)

    velocity_results = {}
    for v_s in VELOCITIES:
        print(f"\n  --- {metric_name} at v_s = {v_s} ---")
        t0 = time.time()

        # Construct metric with the target velocity
        metric = metric_cls(v_s=v_s, **config)

        # v_s=1.0 is the transition case; do a focused pointwise analysis
        # near the wall in addition to the radial sweep.
        wall_analysis = None
        if v_s == 1.0:
            wall_r = np.linspace(R - 2.0, R + 2.0, 5)
            wall_analysis = compute_g00_transition(metric, v_s, wall_r)
            print(f"    Wall analysis (5 points near R={R}): done")

        # g_00 transition profile across full radial range
        g00_transition = compute_g00_transition(metric, v_s, r_values)
        n_positive = sum(1 for pt in g00_transition if pt["g_00"] > 0)
        print(f"    g_00 transition: {n_positive}/{len(g00_transition)} points with g_00 > 0")

        # Find g_00 = 0 crossing by interpolation
        g00_zero_r = _find_g00_zero_crossing(g00_transition)
        if g00_zero_r is not None:
            print(f"    g_00 = 0 crossing at r ~ {g00_zero_r:.2f}")
        else:
            print(" No g_00 = 0 crossing found in sampled range")

        # Full EC characterization at selected radial points
        characterization = characterize_velocity(metric, metric_name, v_s, char_points)
        n_nan = sum(1 for pt in characterization if pt["has_nan_stress_energy"])
        elapsed = time.time() - t0
        print(f"    EC characterization: {len(characterization)} points, {n_nan} with NaN")
        print(f"    Elapsed: {elapsed:.1f}s")

        result = {
            "v_s": v_s,
            "g00_transition": g00_transition,
            "characterization": characterization,
            "n_g00_positive": n_positive,
            "g00_zero_crossing_r": g00_zero_r,
            "n_nan_stress_energy": n_nan,
            "elapsed_s": elapsed,
        }
        if wall_analysis is not None:
            result["wall_analysis_v1"] = wall_analysis

        velocity_results[str(v_s)] = result

    return velocity_results


def _find_g00_zero_crossing(g00_data):
    """Find the radius where g_00 crosses zero by linear interpolation.

    Only detects sign changes where one side is clearly negative
    and the other is clearly positive (both exceed a tolerance threshold).
    This avoids false crossings at v_s = 1.0 where g_00 ~ 0 throughout
    the bubble interior.

    Parameters
    ----------
    g00_data : list[dict]
        Output from compute_g00_transition.

    Returns
    -------
    float or None
        Interpolated radius of g_00 = 0 crossing, or None if not found.
    """
    tol = 1e-6
    for i in range(len(g00_data) - 1):
        g00_a = g00_data[i]["g_00"]
        g00_b = g00_data[i + 1]["g_00"]
        # Require a sign change with both sides clearly nonzero
        if g00_a < -tol and g00_b > tol:
            r_a = g00_data[i]["r"]
            r_b = g00_data[i + 1]["r"]
            frac = abs(g00_a) / (abs(g00_a) + abs(g00_b))
            return r_a + frac * (r_b - r_a)
        if g00_a > tol and g00_b < -tol:
            r_a = g00_data[i]["r"]
            r_b = g00_data[i + 1]["r"]
            frac = abs(g00_a) / (abs(g00_a) + abs(g00_b))
            return r_a + frac * (r_b - r_a)
    return None


def compute_summary(all_results):
    """Aggregate results across metrics and velocities.

    Parameters
    ----------
    all_results : dict
        Per-metric results from analyze_metric.

    Returns
    -------
    dict
        Summary diagnostics confirming det(g)=-1, g_00 sign flip, EC
        pipeline status.
    """
    summary = {
        "det_g_check": {},
        "g00_sign_flip": {},
        "ec_pipeline_status": {},
        "overall": {},
    }

    all_det_g_ok = True
    any_nan = False

    for metric_name, vel_results in all_results.items():
        summary["det_g_check"][metric_name] = {}
        summary["g00_sign_flip"][metric_name] = {}
        summary["ec_pipeline_status"][metric_name] = {}

        for v_key, v_data in vel_results.items():
            # Check det(g) = -1 across all sampled points
            det_g_values = [pt["det_g"] for pt in v_data["g00_transition"] if "det_g" in pt]
            det_g_deviations = [abs(d - (-1.0)) for d in det_g_values]
            max_det_deviation = max(det_g_deviations) if det_g_deviations else 0.0
            det_g_ok = max_det_deviation < 1e-6

            if not det_g_ok:
                all_det_g_ok = False

            summary["det_g_check"][metric_name][v_key] = {
                "max_deviation_from_minus1": max_det_deviation,
                "all_within_tolerance": det_g_ok,
                "n_points_checked": len(det_g_values),
            }

            # g_00 sign flip summary
            n_positive = v_data["n_g00_positive"]
            g00_zero_r = v_data["g00_zero_crossing_r"]

            summary["g00_sign_flip"][metric_name][v_key] = {
                "n_points_with_g00_positive": n_positive,
                "g00_zero_crossing_r": g00_zero_r,
                "sign_flip_present": n_positive > 0,
            }

            # EC pipeline status
            n_nan = v_data["n_nan_stress_energy"]
            if n_nan > 0:
                any_nan = True

            # Check EC margins are finite
            ec_finite = all(
                all(
                    np.isfinite(pt[k])
                    for k in ["nec_margin", "wec_margin", "sec_margin", "dec_margin"]
                )
                for pt in v_data["characterization"]
            )

            # Collect HE types
            he_types = [pt["he_type"] for pt in v_data["characterization"]]

            summary["ec_pipeline_status"][metric_name][v_key] = {
                "n_nan_stress_energy": n_nan,
                "ec_margins_all_finite": ec_finite,
                "he_types_found": sorted(set(he_types)),
                "pipeline_functional": n_nan == 0 and ec_finite,
            }

    summary["overall"] = {
        "det_g_always_minus1": all_det_g_ok,
        "any_nan_found": any_nan,
        "corrected_failure_mode": "g_00 changes sign when v_s * f(r) > 1; det(g) remains -1",
        "physical_interpretation": (
            "When g_00 > 0, coordinate-stationary worldlines are spacelike. "
            "With ds^2=-alpha^2 dt^2+gamma_ij(dx^i+beta^i dt)(dx^j+beta^j dt), "
            "the Eulerian normal n^a=(1,-beta^i)/alpha remains timelike. "
            "The metric stays Lorentzian and Eulerian energy-condition "
            "measurements remain meaningful."
        ),
    }

    return summary


def save_json(results, summary, start_time):
    """Save structured diagnostics to JSON.

    Parameters
    ----------
    results : dict
        Per-metric results.
    summary : dict
        Aggregated summary.
    start_time : str
        ISO-format timestamp.
    """
    output = {
        "metadata": {
            "date": start_time,
            "script": "scripts/run_superluminal_investigation.py",
            "velocities": VELOCITIES,
            "metric_configs": METRIC_CONFIGS,
            "n_radial": N_RADIAL,
            "n_characterize": N_CHARACTERIZE,
        },
        "results": results,
        "summary": summary,
    }
    outpath = os.path.join(RESULTS_DIR, "superluminal_characterization.json")
    dump_json(output, outpath)
    print(f"\nJSON saved to {outpath}")


def save_report(results, summary, start_time):
    """Render the stored radial diagnostics and their observer interpretation."""
    lines = [
        "# Superluminal characterization",
        "",
        f"Date: {start_time}. Source: `superluminal_characterization.json`.",
        "",
        "Alcubierre (tanh) and Lentz (L1/diamond), R=100 and sigma=8. "
        "For unit-lapse, flat-spatial ADM metrics, det(g)=-1 and the signature "
        "is (-,+,+,+) at every speed. With the convention "
        "ds^2=-alpha^2 dt^2+gamma_ij(dx^i+beta^i dt)(dx^j+beta^j dt), "
        "the Eulerian normal is n^a=(1,-beta^i)/alpha and satisfies n^a n_a=-1. "
        "Its energy-condition measurements remain meaningful when g_00>0.",
        "",
        "For these shifts, g_00=-1+v_s^2 f^2. Coordinate-stationary worldlines "
        "become null at g_00=0 and spacelike at g_00>0. This surface alone "
        "does not identify an event horizon. At v_s=1, g_00=0 wherever f=1; "
        "the metric remains invertible.",
        "",
        "| Metric | v_s | Estimated g_00 crossing r | Max det(g) error | NaN in T | HE types | NEC range | WEC range |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for v_s in VELOCITIES:
        key = str(v_s)
        for name in ("alcubierre", "lentz"):
            data = results[name][key]
            det = summary["det_g_check"][name][key]
            crossing = summary["g00_sign_flip"][name][key]["g00_zero_crossing_r"]
            ec = summary["ec_pipeline_status"][name][key]
            radius = f"{crossing:.1f}" if crossing is not None else "none bracketed"
            he = ", ".join(str(t) for t in ec["he_types_found"])
            nec = [p["nec_margin"] for p in data["characterization"]]
            wec = [p["wec_margin"] for p in data["characterization"]]
            lines.append(
                f"| {name.capitalize()} | {v_s:.1f} | {radius} | "
                f"{det['max_deviation_from_minus1']:.1e} | "
                f"{'yes' if ec['n_nan_stress_energy'] else 'no'} | {he} | "
                f"[{min(nec):.2e}, {max(nec):.2e}] | [{min(wec):.2e}, {max(wec):.2e}] |"
            )
    n_det = sum(
        d["n_points_checked"] for rows in summary["det_g_check"].values() for d in rows.values()
    )
    n_ec = sum(len(d["characterization"]) for rows in results.values() for d in rows.values())
    finite = all(
        d["ec_margins_all_finite"]
        for rows in summary["ec_pipeline_status"].values()
        for d in rows.values()
    )
    lines += [
        "",
        f"Determinant checks cover {n_det} sampled points; EC checks cover {n_ec}. "
        f"All sampled EC margins are finite: {'yes' if finite else 'no'}. "
        "The EC search uses eight starts and rapidity cap 5. Ranges are sampled "
        "extrema, and crossing locations are interpolated from a coarse radial "
        "scan at (t,x,y,z)=(0,r,0.01,0). Neither provides a continuum error bound.",
        "",
    ]
    outpath = os.path.join(RESULTS_DIR, "superluminal_report.md")
    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {outpath}")


def main():
    """Run the complete superluminal characterization investigation."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    start_time = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=" * 70)
    print("Superluminal Characterization Investigation")
    print(f"Started: {start_time}")
    print(f"Velocities: {VELOCITIES}")
    print("Metrics: Alcubierre, Lentz")
    print("=" * 70)

    all_results = {}

    print("\n[1/2] Alcubierre metric")
    all_results["alcubierre"] = analyze_metric(
        AlcubierreMetric, "alcubierre", METRIC_CONFIGS["alcubierre"]
    )

    print("\n[2/2] Lentz metric")
    all_results["lentz"] = analyze_metric(LentzMetric, "lentz", METRIC_CONFIGS["lentz"])

    summary = compute_summary(all_results)
    save_json(all_results, summary, start_time)
    save_report(all_results, summary, start_time)

    # Print overall summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  det(g) = -1 at all points: {summary['overall']['det_g_always_minus1']}")
    print(f"  Any NaN found: {summary['overall']['any_nan_found']}")
    print(f"  Failure mode: {summary['overall']['corrected_failure_mode']}")

    for metric_name in ["alcubierre", "lentz"]:
        print(f"\n  {metric_name.capitalize()}:")
        for v_key in [str(v) for v in VELOCITIES]:
            g00_info = summary["g00_sign_flip"][metric_name][v_key]
            n_pos = g00_info["n_points_with_g00_positive"]
            g00_r = g00_info["g00_zero_crossing_r"]
            r_str = f"r ~ {g00_r:.1f}" if g00_r is not None else "N/A"
            print(f"    v_s={v_key}: {n_pos} g_00>0 points, crossing at {r_str}")

    print("\n" + "=" * 70)
    print("Investigation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
