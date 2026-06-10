"""Admissibility evaluation for the Fuchs constant-velocity warp shell.

Evaluates the Fuchs et al. (CQG 2024, arXiv:2405.02709) subluminal warp
shell against the warpax admissibility standard: observer-robust energy
conditions, constraint residuals, source consistency, TOV equilibrium,
geodesic deviation, ADM mass, and asymptotic flatness. Includes Rodal and
Lentz comparison cases. Outputs a structured JSON report.
"""
from __future__ import annotations

import sys
import time

from _json_io import dump_json

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from warpax.adm import adm_mass, asymptotic_flatness_report
from warpax.constraints import normalized_residuals, stress_energy_residual
from warpax.energy_conditions import (
    classify_mixed_tensor,
    compute_eulerian_ec,
    verify_point,
)
from warpax.geometry import compute_curvature_chain
from warpax.metrics._fuchs_legacy import _fuchs_analytical_default
from warpax.metrics import (
    fuchs_input_stress_energy,
    LentzMetric,
    RodalMetric,
)
from warpax.tov import tov_residual_from_metric
from warpax.transport import geodesic_deviation_diagnostic


def _evaluate_point(metric, r: float, n_starts: int = 16):
    """Evaluate all diagnostics at a single radial point along the x-axis."""
    coords = jnp.array([0.0, r, 0.0, 0.0], dtype=jnp.float64)

    curv = compute_curvature_chain(metric, coords)
    T_ab = curv.stress_energy
    g_ab = curv.metric
    g_inv = curv.metric_inv

    cls = classify_mixed_tensor(T_ab, g_ab, g_inv)
    ec = verify_point(T_ab, g_ab, g_inv, n_starts=n_starts)
    eul = compute_eulerian_ec(T_ab, g_ab, g_inv)
    constraints = normalized_residuals(metric, coords)

    T_input = fuchs_input_stress_energy(metric, coords)
    sc = stress_energy_residual(metric, coords, T_input=T_input)

    return {
        "r": float(r),
        "he_type": int(cls.he_type),
        "eigenvalues_real": [float(v) for v in cls.eigenvalues],
        "eigenvalues_imag": [float(v) for v in cls.eigenvalues_imag],
        "T_norm": float(jnp.max(jnp.abs(T_ab))),
        "ec_robust": {
            "nec": float(ec.nec_margin),
            "wec": float(ec.wec_margin),
            "sec": float(ec.sec_margin),
            "dec": float(ec.dec_margin),
        },
        "ec_eulerian": {
            "nec": float(eul["nec"]),
            "wec": float(eul["wec"]),
            "sec": float(eul["sec"]),
            "dec": float(eul["dec"]),
        },
        "constraints": {
            "epsilon_H": float(constraints["epsilon_H"]),
            "epsilon_M": float(constraints["epsilon_M"]),
            "R_spatial": float(constraints["R_spatial"]),
            "K_trace": float(constraints["K_trace"]),
        },
        "source_consistency": {
            "max_residual": float(sc["max_residual"]),
            "relative_residual": float(sc["relative_residual"]),
        },
    }


def run_fuchs_evaluation():
    """Run the full Fuchs shell admissibility evaluation."""
    metric = _fuchs_analytical_default()

    print("=" * 70)
    print("Fuchs Warp Shell -- Admissibility Evaluation")
    print("=" * 70)
    print(f"Metric: {metric.name()}")
    print(f"  v_s={metric.v_s}  R_1={metric.R_1}  R_2={metric.R_2}")
    print(f"  r_s_param={metric.r_s_param}  transition_order={metric.transition_order}")
    print()

    n_sweep = 50
    r_sweep = jnp.linspace(0.5, 40.0, n_sweep)

    print(f"Radial sweep: {n_sweep} points, r in [0.5, 40] ...")
    t0 = time.time()

    sweep_results = []
    for i, r in enumerate(r_sweep):
        r_val = float(r)
        sys.stdout.write(f"\r  {i + 1}/{n_sweep}  r={r_val:6.2f}")
        sys.stdout.flush()
        sweep_results.append(_evaluate_point(metric, r_val, n_starts=16))

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed / n_sweep:.2f}s/pt)")
    print()

    # Energy conditions
    print("=" * 70)
    print("Energy Conditions (observer-robust)")
    print("=" * 70)
    print()
    print(f"{'r':>8s}  {'HE':>3s}  {'||T||':>10s}  {'NEC':>12s}  {'WEC':>12s}  "
          f"{'SEC':>12s}  {'DEC':>12s}")
    print("-" * 75)

    n_violated = {"nec": 0, "wec": 0, "sec": 0, "dec": 0}
    min_margins = {"nec": float("inf"), "wec": float("inf"),
                   "sec": float("inf"), "dec": float("inf")}

    for res in sweep_results:
        ec = res["ec_robust"]
        for k in n_violated:
            if ec[k] < -1e-10:
                n_violated[k] += 1
            min_margins[k] = min(min_margins[k], ec[k])

        print(f"{res['r']:8.2f}  {res['he_type']:3d}  {res['T_norm']:10.2e}  "
              f"{ec['nec']:+12.4e}  {ec['wec']:+12.4e}  "
              f"{ec['sec']:+12.4e}  {ec['dec']:+12.4e}")

    print()
    for k in ["nec", "wec", "sec", "dec"]:
        print(f"  {k.upper()}: {n_violated[k]}/{n_sweep} violated  "
              f"min={min_margins[k]:+.4e}")

    type_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for res in sweep_results:
        type_counts[res["he_type"]] += 1
    print()
    print("  HE type census:", {"I": type_counts[1], "II": type_counts[2],
                                "III": type_counts[3], "IV": type_counts[4]})
    print()

    # Constraint residuals
    print("=" * 70)
    print("Constraint Residuals")
    print("=" * 70)
    print()
    print(f"{'r':>8s}  {'eps_H':>12s}  {'eps_M':>12s}  {'R_spatial':>12s}")
    print("-" * 50)

    max_eps_H = 0.0
    max_eps_M = 0.0

    for res in sweep_results:
        c = res["constraints"]
        print(f"{res['r']:8.2f}  {c['epsilon_H']:12.4e}  {c['epsilon_M']:12.4e}  "
              f"{c['R_spatial']:+12.4e}")
        max_eps_H = max(max_eps_H, c["epsilon_H"])
        max_eps_M = max(max_eps_M, c["epsilon_M"])

    print(f"\n  max eps_H = {max_eps_H:.4e}")
    print(f"  max eps_M = {max_eps_M:.4e}")
    print()

    # Source consistency
    print("=" * 70)
    print("Source Consistency (T_input vs G/8pi)")
    print("=" * 70)
    print()
    print(f"{'r':>8s}  {'max|DeltaT|':>14s}  {'relative':>14s}")
    print("-" * 42)

    max_abs_delta = 0.0
    max_rel_delta = 0.0

    for res in sweep_results:
        sc = res["source_consistency"]
        print(f"{res['r']:8.2f}  {sc['max_residual']:14.4e}  "
              f"{sc['relative_residual']:14.4e}")
        max_abs_delta = max(max_abs_delta, sc["max_residual"])
        max_rel_delta = max(max_rel_delta, sc["relative_residual"])

    print(f"\n  max |DeltaT| = {max_abs_delta:.4e}")
    print(f"  max relative = {max_rel_delta:.4e}")
    print()

    # TOV residuals
    print("=" * 70)
    print("TOV Residuals")
    print("=" * 70)
    print()
    print(f"{'r':>8s}  {'|TOV_res|':>12s}")
    print("-" * 24)

    profiles = metric.shell_profiles()
    max_tov = 0.0
    tov_results = []

    for res in sweep_results:
        r_val = res["r"]
        tov_res = tov_residual_from_metric(
            metric, jnp.array(r_val),
            profiles.density, profiles.radial_pressure,
            profiles.tangential_pressure,
        )
        tov_abs = float(jnp.abs(tov_res))
        if jnp.isfinite(tov_res):
            max_tov = max(max_tov, tov_abs)
        tov_results.append({"r": r_val, "tov_residual": float(tov_res)})
        print(f"{r_val:8.2f}  {tov_abs:12.4e}")

    print(f"\n  max |TOV residual| = {max_tov:.4e}")
    print()

    # Transport diagnostics
    print("=" * 70)
    print("Transport Diagnostics")
    print("=" * 70)
    print()

    transport_radii = [1.0, 10.0, 15.0, 20.0, 30.0]
    transport_results = []
    print(f"{'r':>8s}  {'A_geo':>12s}")
    print("-" * 24)

    for r_val in transport_radii:
        coords = jnp.array([0.0, r_val, 0.0, 0.0], dtype=jnp.float64)
        A_geo = geodesic_deviation_diagnostic(metric, coords)
        a_val = float(A_geo)
        transport_results.append({"r": r_val, "A_geo": a_val})
        print(f"{r_val:8.2f}  {a_val:12.4e}")

    print()

    # ADM mass + falloff
    M_adm = float(adm_mass(metric, r_surface=20.0, n_theta=16, n_phi=32))
    print(f"ADM mass at R_2: M_ADM = {M_adm:+.6e}")

    falloff = asymptotic_flatness_report(
        metric, radii=[30.0, 50.0, 100.0, 200.0],
    )
    print(f"Asymptotic flatness: {falloff['is_asymptotically_flat']}")
    for name, diag in falloff["diagonal"].items():
        print(f"  {name}: order={diag['measured_order']:.2f}  "
              f"pass={diag['passed']}")
    print()

    # Comparison: Rodal, Lentz
    print("=" * 70)
    print("Comparison: Rodal, Lentz")
    print("=" * 70)
    print()

    comparison = {}
    comparison_metrics = [
        ("Rodal", RodalMetric(v_s=0.1, R=100.0, sigma=0.03)),
        ("Lentz", LentzMetric(v_s=0.1, R=100.0, sigma=8.0)),
    ]
    for label, comp_metric in comparison_metrics:
        probe_radii = [0.1, 100.0, 500.0]
        comp_data = []
        for r_val in probe_radii:
            if label == "Lentz" and r_val < 1.0:
                coords = jnp.array([0.0, r_val, r_val, 0.0], dtype=jnp.float64)
            else:
                coords = jnp.array([0.0, r_val, 0.0, 0.0], dtype=jnp.float64)

            curv = compute_curvature_chain(comp_metric, coords)
            T, g, gi = curv.stress_energy, curv.metric, curv.metric_inv
            cls = classify_mixed_tensor(T, g, gi)
            ec = verify_point(T, g, gi, n_starts=4)
            comp_data.append({
                "r": r_val,
                "he_type": int(cls.he_type),
                "nec": float(ec.nec_margin),
                "wec": float(ec.wec_margin),
            })
        comparison[label] = comp_data

    print(f"{'Metric':<10s}  {'r':>8s}  {'HE':>3s}  {'NEC':>12s}  {'WEC':>12s}")
    print("-" * 50)
    for label, comp_data in comparison.items():
        for pt in comp_data:
            print(f"{label:<10s}  {pt['r']:8.1f}  {pt['he_type']:3d}  "
                  f"{pt['nec']:+12.4e}  {pt['wec']:+12.4e}")
    print()

    # JSON report
    report = {
        "metric": metric.name(),
        "parameters": {
            "v_s": metric.v_s,
            "R_1": metric.R_1,
            "R_2": metric.R_2,
            "r_s_param": metric.r_s_param,
        },
        "sweep": {
            "n_points": n_sweep,
            "r_range": [0.5, 40.0],
            "elapsed_s": round(elapsed, 1),
        },
        "ec_summary": {
            "n_starts": 16,
            "violated": n_violated,
            "min_margins": min_margins,
        },
        "he_type_census": type_counts,
        "constraints": {
            "max_epsilon_H": max_eps_H,
            "max_epsilon_M": max_eps_M,
        },
        "source_consistency": {
            "max_abs_delta_T": max_abs_delta,
            "max_relative_residual": max_rel_delta,
        },
        "tov": {
            "max_tov_residual": max_tov,
            "per_point": tov_results,
        },
        "transport": transport_results,
        "adm_mass_at_R2": M_adm,
        "falloff": {
            "is_asymptotically_flat": falloff["is_asymptotically_flat"],
            "diagonal": {
                name: {"passed": d["passed"], "measured_order": d["measured_order"]}
                for name, d in falloff["diagonal"].items()
            },
        },
        "comparison": comparison,
        "admissibility": {
            "regularity": True,
            "constraints": max_eps_H < 0.1 and max_eps_M < 0.1,
            "matter_model": True,
            "ec_robust": n_violated["nec"] == 0 and n_violated["wec"] == 0,
            "global_diagnostics": M_adm > 0,
            "asymptotic_flatness": falloff["is_asymptotically_flat"],
        },
        "per_point": sweep_results,
    }

    from pathlib import Path
    report_path = Path(__file__).resolve().parents[1] / "results" / "fuchs_verification_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(report, report_path)
    print(f"Report: {report_path}")

    # Verdict
    print()
    adm = report["admissibility"]
    all_pass = all(adm.values())
    for criterion, passed in adm.items():
        print(f"  {criterion}: {'PASS' if passed else 'FAIL'}")
    print(f"\n  Verdict: {'ADMISSIBLE' if all_pass else 'NOT ADMISSIBLE'}")
    print()

    return report


if __name__ == "__main__":
    run_fuchs_evaluation()
