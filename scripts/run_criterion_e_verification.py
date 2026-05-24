"""Uniform Criterion-E verification for all eight constructions.

Computes, for each construction, the four Criterion-E / transport observables:
  - M_ADM            : ADM mass (adm.mass.adm_mass surface integral); for the
                       source-first shells the physical conserved mass is the
                       construction's total_mass (reported as M_phys).
  - tidal (A_geo)    : passenger-cavity tidal acceleration (geodesic_deviation_diagnostic)
  - delta_tau        : null round-trip asymmetry (null_round_trip_asymmetry)
  - blueshift (B)    : blueshift hazard functional (blueshift_hazard)

Constructions:
  Natario-class (unit lapse, flat spatial, compact-support shift; M_ADM=0):
    Alcubierre, Natario, Van den Broeck, Lentz, Rodal
  Source-first shells (R_1=10, R_2=20; M_ADM>0):
    Fuchs, S-shell, T-shell

Cavity convention:
  - Bubble metrics: passenger cavity = bubble center; tidal/blueshift probed at
    a small OFF-AXIS interior point (avoids the y=z=0 sqrt-autodiff NaN gotcha).
  - Shells: cavity = interior r < R_1; tidal/blueshift probed at r = R_1/2.
  delta_tau uses an emitter/receiver pair straddling the bubble/shell along x
  (off-axis y-offset for the bubble metrics for the same NaN-safety reason).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from warpax.adm.mass import adm_mass
from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.metrics import (
    LentzMetric, NatarioMetric, RodalMetric, VanDenBroeckMetric,
)
from warpax.metrics.fuchs_construction import fuchs_default
from warpax.metrics.sshell import sshell_default
from warpax.metrics.tshell import tshell_default
from warpax.transport.diagnostics import (
    blueshift_hazard,
    geodesic_deviation_diagnostic,
    null_round_trip_asymmetry,
)


OUTPUT = Path(__file__).resolve().parents[1] / "output" / "criterion_e_verification.json"


def _safe(fn, *, default=None, reason=""):
    """Run a diagnostic, returning (value, note). NaN/exception -> N/A."""
    try:
        v = float(fn())
        if v != v:  # NaN
            return None, f"N/A (NaN: {reason})"
        return v, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"N/A ({type(exc).__name__}: {exc})"


def verify_bubble(name, metric, *, R, eps_off=0.5, adm_r=None):
    """Natario-class bubble metric. M_ADM=0 by construction (unit lapse,
    compact-support shift). Cavity = bubble center; probe slightly off-axis."""
    print(f"\n--- {name} (bubble; R={R}) ---")
    # Cavity / interior point: bubble center, nudged off-axis for sqrt safety.
    cavity = jnp.array([0.0, 0.0, eps_off, 0.0], dtype=jnp.float64)

    # M_ADM: Natario-class has M_ADM = 0 by construction (no gravitational
    # mass; unit lapse). We also report the surface integral as a numeric check.
    if adm_r is not None:
        m_adm_num, m_note = _safe(lambda: adm_mass(metric, r_surface=adm_r,
                                                   n_theta=16, n_phi=32),
                                  reason="adm surface integral")
    else:
        m_adm_num, m_note = 0.0, ""
    m_phys = 0.0  # by construction

    tidal, t_note = _safe(lambda: geodesic_deviation_diagnostic(metric, cavity),
                          reason="tidal")
    blue, b_note = _safe(lambda: blueshift_hazard(metric, cavity,
                                                  tau_max=4.0 * R, num_points=200),
                         reason="blueshift")

    # delta_tau: straddle the bubble along x, off-axis to avoid on-axis NaN.
    emitter = jnp.array([0.0, -2.0 * R, eps_off, 0.0], dtype=jnp.float64)
    receiver = jnp.array([0.0, 2.0 * R, eps_off, 0.0], dtype=jnp.float64)
    dtau, d_note = _safe(lambda: null_round_trip_asymmetry(
        metric, emitter, receiver, tau_max=10.0 * R, num_points=400),
        reason="delta_tau")

    return {
        "metric": name, "class": "bubble",
        "M_ADM": m_phys, "M_ADM_note": "0 by construction (Natario class)",
        "M_ADM_surface_numeric": m_adm_num, "M_ADM_surface_note": m_note,
        "tidal": tidal, "tidal_note": t_note,
        "delta_tau": dtau, "delta_tau_note": d_note,
        "blueshift": blue, "blueshift_note": b_note,
    }


def verify_shell(name, metric, *, R_1=10.0, R_2=20.0, adm_r=20.0):
    """Source-first shell. M_phys = total_mass (conserved construction mass);
    also report adm_mass surface integral at adm_r. Cavity = r < R_1."""
    print(f"\n--- {name} (shell; R_1={R_1}, R_2={R_2}) ---")
    cavity = jnp.array([0.0, R_1 * 0.5, 0.0, 0.0], dtype=jnp.float64)

    m_phys = float(metric.total_mass)
    m_adm_num, m_note = _safe(lambda: adm_mass(metric, r_surface=adm_r,
                                               n_theta=16, n_phi=32),
                              reason="adm surface integral")

    tidal, t_note = _safe(lambda: geodesic_deviation_diagnostic(metric, cavity),
                          reason="tidal")
    blue, b_note = _safe(lambda: blueshift_hazard(metric, cavity,
                                                  tau_max=100.0, num_points=200),
                         reason="blueshift")

    # delta_tau: straddle the shell along x (matches run_delta_tau_scan.py).
    emitter = jnp.array([0.0, -25.0, 0.0, 0.0], dtype=jnp.float64)
    receiver = jnp.array([0.0, 25.0, 0.0, 0.0], dtype=jnp.float64)
    dtau, d_note = _safe(lambda: null_round_trip_asymmetry(
        metric, emitter, receiver, tau_max=80.0, num_points=600),
        reason="delta_tau")

    return {
        "metric": name, "class": "shell",
        "M_ADM": m_phys, "M_ADM_note": "total_mass (conserved construction mass)",
        "M_ADM_surface_numeric": m_adm_num, "M_ADM_surface_note": m_note,
        "tidal": tidal, "tidal_note": t_note,
        "delta_tau": dtau, "delta_tau_note": d_note,
        "blueshift": blue, "blueshift_note": b_note,
    }


def _e_pass(rec):
    """Criterion E (positive ADM mass) pass/fail."""
    return bool(rec["M_ADM"] is not None and rec["M_ADM"] > 1e-6)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    t_all = time.time()
    # Natario-class bubbles (M_ADM = 0 by construction)
    rows.append(verify_bubble("Alcubierre", AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0),
                              R=1.0, eps_off=0.3, adm_r=5.0))
    rows.append(verify_bubble("Natario", NatarioMetric(v_s=0.1, R=100.0, sigma=0.03),
                              R=100.0, eps_off=5.0, adm_r=400.0))
    rows.append(verify_bubble("Van den Broeck",
                              VanDenBroeckMetric(v_s=0.1, R=350.0, sigma=8.0,
                                                 R_tilde=200.0, alpha_vdb=0.5, sigma_B=8.0),
                              R=350.0, eps_off=10.0, adm_r=1000.0))
    rows.append(verify_bubble("Lentz", LentzMetric(v_s=0.1, R=100.0, sigma=8.0),
                              R=100.0, eps_off=5.0, adm_r=400.0))
    rows.append(verify_bubble("Rodal", RodalMetric(v_s=0.1, R=100.0, sigma=0.03),
                              R=100.0, eps_off=5.0, adm_r=400.0))

    # Source-first shells (M_ADM > 0)
    rows.append(verify_shell("Fuchs", fuchs_default(), R_1=10.0, R_2=20.0, adm_r=20.0))
    rows.append(verify_shell("S-shell", sshell_default(), R_1=10.0, R_2=20.0, adm_r=20.0))
    rows.append(verify_shell("T-shell", tshell_default(v_0=0.1), R_1=10.0, R_2=20.0, adm_r=20.0))

    for r in rows:
        r["E_pass"] = _e_pass(r)

    out = {"rows": rows, "elapsed_s": time.time() - t_all}
    OUTPUT.write_text(json.dumps(out, indent=2))

    # Compact table to stdout
    print("\n" + "=" * 96)
    print(f"{'metric':<16}{'M_ADM':>10}{'tidal':>14}{'delta_tau':>14}{'blueshift':>14}{'E-pass':>9}")
    print("-" * 96)
    def f(v):
        return "N/A" if v is None else f"{v:.4g}"
    for r in rows:
        print(f"{r['metric']:<16}{f(r['M_ADM']):>10}{f(r['tidal']):>14}"
              f"{f(r['delta_tau']):>14}{f(r['blueshift']):>14}{str(r['E_pass']):>9}")
    print(f"\n  -> {OUTPUT}  ({out['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
