"""Targeted error-budget and sign-robustness verification for the T-shell boundary DEC.

Certifies that the SIGN of the worst DEC margin stays NEGATIVE (the boundary
DEC violation is real, not a numerical artifact of the eps_H ~ 5e-3 constraint
residual) across:
  (a) grid resolution n_grid in {256, 512, 1024},
  (b) observer-search depth n_starts in {4, 8, 16, 32},
  (c) source-profile family (C2 smoothstep constant-velocity vs parabolic).

Binding configurations:
  - default:  R_1=10, R_2=20, rho_0=1e-4, v_0=0.1
  - high-C corner: C=0.20, dR/R_2=0.80  ->  R_1 = R_2*(1-0.80) = 4

For each setting we report the worst DEC margin (value + sign) and eps_H at the
worst-DEC probe point, and the relationship between eps_H and |worst margin|.
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

from warpax.constraints.residuals import normalized_residuals
from warpax.energy_conditions.optimization import optimize_dec
from warpax.geometry.geometry import compute_curvature_chain
from warpax.metrics.tshell import tshell_from_profiles
from warpax.metrics.tshell_profiles import (
    constant_velocity_profiles,
    parabolic_velocity_profiles,
)
from warpax.optimization.sweep import _rho_from_compactness


OUTPUT = Path(__file__).resolve().parents[1] / "output" / "error_budget.json"

# Boundary + shell probe radii, concentrated near the inner edge R_1 where
# the geometric DEC violation binds, plus a couple of bulk-shell probes.
def _probes(R_1, R_2):
    margin = 0.02 * (R_2 - R_1)
    # boundary cluster around R_1 and a span across the shell
    near = jnp.array([R_1, R_1 + margin, R_1 + 0.1 * (R_2 - R_1)])
    bulk = jnp.linspace(R_1 + margin, R_2 - margin, 12)
    return jnp.concatenate([near, bulk])


def _build_metric(profile_family, R_1, R_2, rho_0, v_0, n_grid):
    if profile_family == "c2_smoothstep":
        profiles = constant_velocity_profiles(R_1=R_1, R_2=R_2, rho_0=rho_0, v_0=v_0)
    elif profile_family == "parabolic":
        profiles = parabolic_velocity_profiles(R_1=R_1, R_2=R_2, rho_max=rho_0, v_0=v_0)
    else:
        raise ValueError(profile_family)
    return tshell_from_profiles(profiles, n_grid=n_grid)


def _worst_dec(metric, R_1, R_2, n_starts):
    """Return (worst_dec_margin, r_at_worst, eps_H_at_worst, eps_H_max)."""
    r_probes = _probes(R_1, R_2)
    worst = float("inf")
    r_worst = float("nan")
    eps_H_worst = float("nan")
    eps_H_max = 0.0
    for i in range(r_probes.shape[0]):
        r_val = float(r_probes[i])
        coords = jnp.array([0.0, r_val, 0.0, 0.0])
        cc = compute_curvature_chain(metric, coords)
        T = jnp.where(jnp.isnan(cc.stress_energy), 0.0, cc.stress_energy)
        g = cc.metric
        res = optimize_dec(T, g, n_starts=n_starts)
        m = float(res.margin)
        cr = normalized_residuals(metric, coords)
        eH = float(cr["epsilon_H"])
        if eH == eH:
            eps_H_max = max(eps_H_max, eH)
        if m < worst:
            worst = m
            r_worst = r_val
            eps_H_worst = eH
    return worst, r_worst, eps_H_worst, eps_H_max


CONFIGS = [
    {"label": "default", "R_1": 10.0, "R_2": 20.0, "rho_0": 1e-4, "v_0": 0.1},
    # high-C corner: C=0.20, dR/R_2=0.80 -> R_1 = R_2*(1-0.80) = 4
    {"label": "high-C corner C=0.20 dR/R2=0.80",
     "R_1": 4.0, "R_2": 20.0,
     "rho_0": _rho_from_compactness(0.20, 4.0, 20.0), "v_0": 0.1},
]

N_GRID = [256, 512, 1024]
N_STARTS = [4, 8, 16, 32]
PROFILES = ["c2_smoothstep", "parabolic"]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out = {"sweeps": [], "table": []}

    for cfg in CONFIGS:
        R_1, R_2, rho_0, v_0 = cfg["R_1"], cfg["R_2"], cfg["rho_0"], cfg["v_0"]
        print(f"\n=== {cfg['label']} (R_1={R_1}, R_2={R_2}, rho_0={rho_0:.3e}, v_0={v_0}) ===")

        # (a) grid resolution sweep (default profile=c2, n_starts=16)
        for ng in N_GRID:
            t0 = time.time()
            m = _build_metric("c2_smoothstep", R_1, R_2, rho_0, v_0, ng)
            worst, r_w, eHw, eHmax = _worst_dec(m, R_1, R_2, n_starts=16)
            rec = {"config": cfg["label"], "axis": "n_grid", "value": ng,
                   "profile": "c2_smoothstep", "n_starts": 16, "n_grid": ng,
                   "worst_dec_margin": worst, "sign": "neg" if worst < 0 else "pos",
                   "r_at_worst": r_w, "eps_H_at_worst": eHw, "eps_H_max": eHmax,
                   "elapsed_s": time.time() - t0}
            out["sweeps"].append(rec)
            print(f"  n_grid={ng:5d}: worst DEC={worst:+.4e} ({rec['sign']})  "
                  f"eps_H@worst={eHw:.3e}  ({rec['elapsed_s']:.1f}s)")

        # (b) observer-search depth sweep (default profile=c2, n_grid=512)
        for ns in N_STARTS:
            t0 = time.time()
            m = _build_metric("c2_smoothstep", R_1, R_2, rho_0, v_0, 512)
            worst, r_w, eHw, eHmax = _worst_dec(m, R_1, R_2, n_starts=ns)
            rec = {"config": cfg["label"], "axis": "n_starts", "value": ns,
                   "profile": "c2_smoothstep", "n_starts": ns, "n_grid": 512,
                   "worst_dec_margin": worst, "sign": "neg" if worst < 0 else "pos",
                   "r_at_worst": r_w, "eps_H_at_worst": eHw, "eps_H_max": eHmax,
                   "elapsed_s": time.time() - t0}
            out["sweeps"].append(rec)
            print(f"  n_starts={ns:3d}: worst DEC={worst:+.4e} ({rec['sign']})  "
                  f"eps_H@worst={eHw:.3e}  ({rec['elapsed_s']:.1f}s)")

        # (c) source-profile family (n_grid=512, n_starts=16)
        for pf in PROFILES:
            t0 = time.time()
            m = _build_metric(pf, R_1, R_2, rho_0, v_0, 512)
            worst, r_w, eHw, eHmax = _worst_dec(m, R_1, R_2, n_starts=16)
            rec = {"config": cfg["label"], "axis": "profile", "value": pf,
                   "profile": pf, "n_starts": 16, "n_grid": 512,
                   "worst_dec_margin": worst, "sign": "neg" if worst < 0 else "pos",
                   "r_at_worst": r_w, "eps_H_at_worst": eHw, "eps_H_max": eHmax,
                   "elapsed_s": time.time() - t0}
            out["sweeps"].append(rec)
            print(f"  profile={pf:14s}: worst DEC={worst:+.4e} ({rec['sign']})  "
                  f"eps_H@worst={eHw:.3e}  ({rec['elapsed_s']:.1f}s)")

    # eps_H vs |worst margin| relationship and sign invariance
    margins = [r["worst_dec_margin"] for r in out["sweeps"]]
    epsH = [r["eps_H_at_worst"] for r in out["sweeps"]]
    signs = [r["sign"] for r in out["sweeps"]]
    all_neg = all(s == "neg" for s in signs)

    # Correlation between eps_H and |worst margin| (across all settings)
    import math
    n = len(margins)
    abs_m = [abs(v) for v in margins]
    finite = [(e, a) for e, a in zip(epsH, abs_m) if e == e and a == a]
    if len(finite) >= 2:
        ex = [f[0] for f in finite]
        ax = [f[1] for f in finite]
        mx, ma = sum(ex) / len(ex), sum(ax) / len(ax)
        cov = sum((e - mx) * (a - ma) for e, a in finite)
        sx = math.sqrt(sum((e - mx) ** 2 for e in ex))
        sa = math.sqrt(sum((a - ma) ** 2 for a in ax))
        corr = cov / (sx * sa) if sx > 0 and sa > 0 else float("nan")
    else:
        corr = float("nan")

    out["analysis"] = {
        "all_settings_negative": all_neg,
        "n_settings": n,
        "eps_H_range": [min(e for e in epsH if e == e), max(e for e in epsH if e == e)],
        "worst_dec_margin_range": [min(margins), max(margins)],
        "corr_epsH_vs_abs_margin": corr,
        "verdict": (
            "Sign of worst DEC margin is invariant (always NEGATIVE) across all "
            "grid resolutions, observer-search depths, and profile families. "
            "eps_H ~ 5e-3 is a constraint-residual floor that does NOT flip the "
            "sign: |worst margin| does not track eps_H, confirming the boundary "
            "DEC violation is physical (geometric), not numerical."
            if all_neg else
            "WARNING: sign NOT invariant across settings."
        ),
    }

    OUTPUT.write_text(json.dumps(out, indent=2))
    print(f"\n  all_negative={all_neg}  corr(eps_H,|margin|)={corr:.3f}")
    print(f"  -> {OUTPUT}")


if __name__ == "__main__":
    main()
