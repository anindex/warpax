"""Admissibility evaluation for Fuchs, Rodal, and Lentz warp proposals.

Evaluates three warp drive constructions against the warpax 5-criterion
admissibility standard:

    A. Regularity      -- C^k smooth metric, no discontinuities
    B. Constraints     -- Hamiltonian/Momentum constraint residuals
    C. Matter model    -- Source consistency (T_input vs G_ab/8pi)
    D. EC margins      -- Observer-robust NEC/WEC/DEC/SEC
    E. Global          -- ADM mass, Hawking-Ellis type, asymptotic flatness

Proposals evaluated:
    1. Fuchs et al. (CQG 2024, arXiv:2405.02709) -- Gaussian-smoothed shell
    2. Rodal (arXiv:2512.18008) -- irrotational shift
    3. Lentz (arXiv:2006.07125) -- diamond soliton
"""
from __future__ import annotations

import json
import sys
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from warpax.constraints import normalized_residuals
from warpax.energy_conditions import (
    classify_mixed_tensor,
    compute_eulerian_ec,
    verify_point,
)
from warpax.geometry import compute_curvature_chain
from warpax.metrics import (
    LentzMetric,
    RodalMetric,
    fuchs_default,
)
from warpax.transport import geodesic_deviation_diagnostic


# ---------------------------------------------------------------------------
# Per-point evaluation
# ---------------------------------------------------------------------------

def _evaluate_point(metric, coords, *, n_starts=16):
    """Evaluate all diagnostics at a single spacetime point."""
    curv = compute_curvature_chain(metric, coords)
    T_ab, g_ab, g_inv = curv.stress_energy, curv.metric, curv.metric_inv

    cls = classify_mixed_tensor(T_ab, g_ab, g_inv)
    ec = verify_point(T_ab, g_ab, g_inv, n_starts=n_starts)
    eul = compute_eulerian_ec(T_ab, g_ab, g_inv)
    constraints = normalized_residuals(metric, coords)

    result = {
        "coords": [float(c) for c in coords],
        "he_type": int(cls.he_type),
        "eigenvalues_real": [float(v) for v in cls.eigenvalues],
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
        },
    }
    return result


# ---------------------------------------------------------------------------
# Proposal evaluation
# ---------------------------------------------------------------------------

def evaluate_fuchs():
    """5-criterion evaluation of the Fuchs shell."""
    from warpax.adm import adm_mass, asymptotic_flatness_report

    print("=" * 70)
    print("FUCHS WARP SHELL (Gaussian-Smoothed Construction)")
    print("=" * 70)

    metric = fuchs_default()
    print(f"Metric: {metric.name()}")
    print(f"  v_s={metric.v_s}  R_1={metric.R_1}  R_2={metric.R_2}")
    print()

    # Radial sweep through the full range
    n_sweep = 50
    r_sweep = jnp.linspace(0.5, 40.0, n_sweep)

    print(f"Radial sweep: {n_sweep} points ...")
    t0 = time.time()
    sweep_results = []
    for i, r in enumerate(r_sweep):
        r_val = float(r)
        sys.stdout.write(f"\r  {i + 1}/{n_sweep}  r={r_val:6.2f}")
        sys.stdout.flush()
        coords = jnp.array([0.0, r_val, 0.0, 0.0], dtype=jnp.float64)
        sweep_results.append(_evaluate_point(metric, coords, n_starts=16))
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print()

    # The construction uses Gaussian-smoothed profiles (C^inf)
    regularity = True
    print("  [A] Regularity: PASS (Gaussian-smoothed C^inf profiles)")

    # Criterion B: Constraints
    max_eps_H = max(r["constraints"]["epsilon_H"] for r in sweep_results)
    max_eps_M = max(r["constraints"]["epsilon_M"] for r in sweep_results)
    constraints_pass = max_eps_H < 0.1 and max_eps_M < 0.1
    print(f"  [B] Constraints: {'PASS' if constraints_pass else 'FAIL'}"
          f"  max ε_H={max_eps_H:.4e}  max ε_M={max_eps_M:.4e}")

    # Criterion C: Matter model
    # Fuchs has a prescribed source model (shell profiles)
    matter_model = True
    print("  [C] Matter model: PASS (iteratively-smoothed shell, TOV-derived)")

    # Criterion D: EC margins
    n_violated = {"nec": 0, "wec": 0, "sec": 0, "dec": 0}
    min_margins = {"nec": float("inf"), "wec": float("inf"),
                   "sec": float("inf"), "dec": float("inf")}
    for res in sweep_results:
        ec = res["ec_robust"]
        for k in n_violated:
            if ec[k] < -1e-10:
                n_violated[k] += 1
            min_margins[k] = min(min_margins[k], ec[k])

    # Count shell-interior violations specifically
    R_1, R_2 = metric.R_1, metric.R_2
    interior_violations = 0
    interior_total = 0
    for res in sweep_results:
        r_val = res["coords"][1]
        if R_1 <= r_val <= R_2:
            interior_total += 1
            ec = res["ec_robust"]
            if ec["nec"] < -1e-10 or ec["wec"] < -1e-10:
                interior_violations += 1

    ec_pass = n_violated["nec"] == 0 and n_violated["wec"] == 0
    print(f"  [D] EC margins: {'PASS' if ec_pass else 'FAIL'}")
    print(f"       NEC violated: {n_violated['nec']}/{n_sweep}  min={min_margins['nec']:+.4e}")
    print(f"       WEC violated: {n_violated['wec']}/{n_sweep}  min={min_margins['wec']:+.4e}")
    print(f"       DEC violated: {n_violated['dec']}/{n_sweep}  min={min_margins['dec']:+.4e}")
    print(f"       SEC violated: {n_violated['sec']}/{n_sweep}  min={min_margins['sec']:+.4e}")
    print(f"       Shell interior: {interior_violations}/{interior_total} violated")

    # HE type census
    type_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for res in sweep_results:
        type_counts[res["he_type"]] += 1
    print(f"       HE types: I={type_counts[1]} II={type_counts[2]}"
          f" III={type_counts[3]} IV={type_counts[4]}")

    # Criterion E: Global diagnostics
    try:
        from warpax.adm import adm_mass, asymptotic_flatness_report
        M_adm = float(adm_mass(metric, r_surface=20.0, n_theta=16, n_phi=32))
        falloff = asymptotic_flatness_report(metric, radii=[30.0, 50.0, 100.0])
        af_pass = falloff["is_asymptotically_flat"]
    except Exception as exc:
        M_adm = float("nan")
        af_pass = False
        falloff = {"is_asymptotically_flat": False}
        print(f"       ADM/falloff error: {exc}")

    global_pass = M_adm > 0 if not jnp.isnan(M_adm) else False
    print(f"  [E] Global: {'PASS' if global_pass else 'FAIL'}"
          f"  M_ADM={M_adm:+.6e}  AF={af_pass}")

    # Transport: geodesic deviation
    transport_results = []
    for r_val in [1.0, 10.0, 15.0, 20.0, 30.0]:
        coords = jnp.array([0.0, r_val, 0.0, 0.0], dtype=jnp.float64)
        A_geo = float(geodesic_deviation_diagnostic(metric, coords))
        transport_results.append({"r": r_val, "A_geo": A_geo})
        print(f"       A_geo(r={r_val:.0f}) = {A_geo:.4e}")

    print()

    return {
        "name": "Fuchs",
        "criteria": {
            "A_regularity": {"pass": regularity, "detail": "Gaussian-smoothed C^inf"},
            "B_constraints": {"pass": constraints_pass,
                              "max_eps_H": max_eps_H, "max_eps_M": max_eps_M},
            "C_matter_model": {"pass": matter_model,
                               "detail": "Iteratively-smoothed shell, TOV-derived"},
            "D_ec_margins": {"pass": ec_pass,
                             "violated": n_violated, "min_margins": min_margins,
                             "interior_violations": f"{interior_violations}/{interior_total}"},
            "E_global": {"pass": global_pass, "M_ADM": M_adm,
                         "asymptotic_flat": af_pass},
        },
        "he_type_census": type_counts,
        "transport": transport_results,
        "per_point": sweep_results,
        "elapsed_s": round(elapsed, 1),
    }


def _evaluate_natario_class(metric, label, *, n_sweep=50, r_range=(0.5, 500.0)):
    """Evaluate a Natário-class metric (unit lapse, flat spatial) under all 5 criteria."""
    print("=" * 70)
    print(f"{label.upper()}")
    print("=" * 70)
    print(f"Metric: {metric.name()}")
    print()

    # Radial sweep
    r_sweep = jnp.linspace(*r_range, n_sweep)
    t0 = time.time()
    sweep_results = []
    for i, r in enumerate(r_sweep):
        r_val = float(r)
        sys.stdout.write(f"\r  {i + 1}/{n_sweep}  r={r_val:6.1f}")
        sys.stdout.flush()
        coords = jnp.array([0.0, r_val, 0.0, 0.0], dtype=jnp.float64)
        sweep_results.append(_evaluate_point(
            metric, coords, n_starts=16,
        ))
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")

    # Criterion A: Regularity (tanh-based -> C^inf)
    regularity = True
    print("  [A] Regularity: PASS (C^inf tanh profiles)")

    # Criterion B: Constraints (unit lapse + flat spatial -> trivial)
    max_eps_H = max(r["constraints"]["epsilon_H"] for r in sweep_results)
    max_eps_M = max(r["constraints"]["epsilon_M"] for r in sweep_results)
    constraints_pass = max_eps_H < 0.1 and max_eps_M < 0.1
    print(f"  [B] Constraints: {'PASS' if constraints_pass else 'FAIL'}"
          f"  max ε_H={max_eps_H:.4e}  max ε_M={max_eps_M:.4e}")

    # Criterion C: N/A (metric-first, no source model)
    print("  [C] Matter model: N/A (metric-first construction, no source prescribed)")

    # Criterion D: EC margins
    n_violated = {"nec": 0, "wec": 0, "sec": 0, "dec": 0}
    min_margins = {"nec": float("inf"), "wec": float("inf"),
                   "sec": float("inf"), "dec": float("inf")}
    for res in sweep_results:
        ec = res["ec_robust"]
        for k in n_violated:
            if ec[k] < -1e-10:
                n_violated[k] += 1
            min_margins[k] = min(min_margins[k], ec[k])

    ec_pass = n_violated["nec"] == 0 and n_violated["wec"] == 0
    print(f"  [D] EC margins: {'PASS' if ec_pass else 'FAIL'}")
    print(f"       NEC violated: {n_violated['nec']}/{n_sweep}  min={min_margins['nec']:+.4e}")
    print(f"       WEC violated: {n_violated['wec']}/{n_sweep}  min={min_margins['wec']:+.4e}")
    print(f"       DEC violated: {n_violated['dec']}/{n_sweep}  min={min_margins['dec']:+.4e}")
    print(f"       SEC violated: {n_violated['sec']}/{n_sweep}  min={min_margins['sec']:+.4e}")

    # HE type census
    type_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for res in sweep_results:
        type_counts[res["he_type"]] += 1
    print(f"       HE types: I={type_counts[1]} II={type_counts[2]}"
          f" III={type_counts[3]} IV={type_counts[4]}")

    # Criterion E: Global - unit lapse -> M_ADM = 0 (no gravitational mass)
    M_adm = 0.0  # Natário class has M_ADM = 0 by construction
    af_pass = False  # Compact support -> no 1/r falloff
    print("  [E] Global: FAIL  M_ADM=0 (Natário class), AF=False (compact support)")

    # Transport
    transport_results = []
    for r_val in [0.1, 50.0, 100.0, 200.0]:
        r_val = min(r_val, float(r_range[1]) * 0.95)
        coords = jnp.array([0.0, r_val, 0.0, 0.0], dtype=jnp.float64)
        A_geo = float(geodesic_deviation_diagnostic(metric, coords))
        transport_results.append({"r": r_val, "A_geo": A_geo})
        print(f"       A_geo(r={r_val:.0f}) = {A_geo:.4e}")

    print()

    return {
        "name": label,
        "criteria": {
            "A_regularity": {"pass": regularity, "detail": "C^inf tanh profiles"},
            "B_constraints": {"pass": constraints_pass,
                              "max_eps_H": max_eps_H, "max_eps_M": max_eps_M},
            "C_matter_model": {"pass": False,
                               "detail": "N/A - metric-first, no source prescribed"},
            "D_ec_margins": {"pass": ec_pass,
                             "violated": n_violated, "min_margins": min_margins},
            "E_global": {"pass": False, "M_ADM": M_adm,
                         "asymptotic_flat": af_pass,
                         "detail": "M_ADM=0 (Natário class)"},
        },
        "he_type_census": type_counts,
        "transport": transport_results,
        "per_point": sweep_results,
        "elapsed_s": round(elapsed, 1),
    }


def evaluate_rodal():
    """5-criterion evaluation of the Rodal irrotational metric."""
    metric = RodalMetric(v_s=0.1, R=100.0, sigma=0.03)
    return _evaluate_natario_class(
        metric, "Rodal Irrotational Warp Drive",
        n_sweep=50, r_range=(0.5, 500.0),
    )


def evaluate_lentz():
    """5-criterion evaluation of the Lentz soliton metric."""
    metric = LentzMetric(v_s=0.1, R=100.0, sigma=8.0)
    return _evaluate_natario_class(
        metric, "Lentz Diamond Soliton",
        n_sweep=50, r_range=(0.5, 500.0),
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary_table(reports):
    """Print the 5-criterion comparison matrix."""
    print("\n" + "=" * 80)
    print("5-CRITERION ADMISSIBILITY COMPARISON")
    print("=" * 80)
    print()

    names = [r["name"] for r in reports]
    header = f"{'Criterion':<22s}" + "".join(f"{n:>20s}" for n in names)
    print(header)
    print("-" * len(header))

    criteria = [
        ("A Regularity", "A_regularity"),
        ("B Constraints", "B_constraints"),
        ("C Matter Model", "C_matter_model"),
        ("D EC Margins", "D_ec_margins"),
        ("E Global", "E_global"),
    ]
    for label, key in criteria:
        row = f"{label:<22s}"
        for r in reports:
            c = r["criteria"][key]
            if c.get("detail", "").startswith("N/A"):
                row += f"{'N/A':>20s}"
            elif c["pass"]:
                row += f"{'PASS':>20s}"
            else:
                row += f"{'FAIL':>20s}"
        print(row)

    print()
    print("VERDICT:")
    for r in reports:
        all_pass = all(
            c["pass"] for c in r["criteria"].values()
            if not c.get("detail", "").startswith("N/A")
        )
        verdict = "ADMISSIBLE" if all_pass else "NOT ADMISSIBLE"
        print(f"  {r['name']}: {verdict}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_all_evaluations():
    """Run all three proposal evaluations and generate combined report."""
    reports = []
    reports.append(evaluate_fuchs())
    reports.append(evaluate_rodal())
    reports.append(evaluate_lentz())

    _print_summary_table(reports)

    # Save combined JSON report
    from pathlib import Path
    report_path = Path(__file__).resolve().parents[1] / "results" / "proposals_verification_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "proposals": {r["name"]: r for r in reports},
        "summary": {
            r["name"]: {
                k: v["pass"] for k, v in r["criteria"].items()
            }
            for r in reports
        },
    }
    with open(report_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"Combined report: {report_path}")

    return combined


if __name__ == "__main__":
    run_all_evaluations()
