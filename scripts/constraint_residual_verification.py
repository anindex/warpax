"""Source-aware constraint residual verification (vacuum vs source-aware vs source-consistency).

Quantifies the difference between the published VACUUM normalized residual
(normalized_residuals(metric, coords) with E=0) and the genuine SOURCE-AWARE
constraint-satisfaction residual that passes the solver's own prescribed
Eulerian source arrays E(r), S_i(r).

For each of {S-shell, T-shell, Fuchs} at ~25 in-shell x-axis probes (r in [10,20]):
  1. VACUUM eps_H, eps_M       = normalized_residuals(metric, coords)               (no source)
  2. SOURCE-AWARE eps_H, eps_M = normalized_residuals(metric, coords, E, S_i)        (prescribed source)
  3. SOURCE-CONSISTENCY        = stress_energy_residual relative residual with T_input = prescribed fluid T

Conventions (from residuals.py / tshell_solver.py):
  - Hamiltonian: H = R + K^2 - K_sq - 16 pi E,  E = Eulerian energy density.
  - Momentum:    M_i = D_j A^j_i - 8 pi S_i,    S_i = LOWERED-index momentum density.
    The T-shell solver stores S^x (contravariant) = Gamma^2 (rho+p) v^x.
    On the x-axis the lowered component is S_x = gamma_xx S^x = e^{2 Lambda} S^x.
  - Source-consistency T_input: covariant T_{ab} of the prescribed fluid
    (static isotropic for S-shell/Fuchs; tilted for T-shell), compared to G/8pi.

Run:
    JAX_PLATFORMS=cpu warpax/.venv/bin/python warpax/scripts/constraint_residual_verification.py
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.constraints.residuals import normalized_residuals
from warpax.constraints.source_consistency import stress_energy_residual
from warpax.metrics.sshell import sshell_default
from warpax.metrics.sshell_profiles import constant_density_profiles
from warpax.metrics.tshell import tshell_default
from warpax.metrics.tshell_profiles import constant_velocity_profiles
from warpax.metrics.fuchs_construction import build_fuchs_construction, fuchs_default

R_1, R_2 = 10.0, 20.0
N_PROBES = 25
R_PROBES = np.linspace(R_1, R_2, N_PROBES)


def _peak_mean(vals):
    a = np.asarray(vals, dtype=float)
    finite = a[np.isfinite(a)]
    n_nonfinite = int(a.size - finite.size)
    if finite.size == 0:
        return {"peak": None, "mean": None, "n_nonfinite": n_nonfinite}
    return {
        "peak": float(np.max(np.abs(finite))),
        "mean": float(np.mean(np.abs(finite))),
        "n_nonfinite": n_nonfinite,
    }


def _interp1d(r, r_grid, vals):
    import interpax
    rc = jnp.clip(r, r_grid[0], r_grid[-1])
    return interpax.interp1d(rc, r_grid, vals, method="cubic")


# ---------------------------------------------------------------------------
# Static isotropic fluid T_input (S-shell, Fuchs): T_ab = (rho+p) u_a u_b + p g_ab
# u^a = (1/alpha, 0,0,0) static observer.
# ---------------------------------------------------------------------------
def _T_input_static(metric, coords, rho_val, p_val):
    g = metric(coords)
    alpha = metric.lapse(coords)
    u_up = jnp.array([1.0 / alpha, 0.0, 0.0, 0.0])
    u_down = g @ u_up  # for static observer with shift, u_a = g_{a0}/alpha
    return (rho_val + p_val) * jnp.outer(u_down, u_down) + p_val * g


# ---------------------------------------------------------------------------
# Tilted fluid T_input (T-shell): u^a = Gamma (n^a + v^a), v^a x-directed.
# n^a = (1/alpha, -beta^i/alpha); v^a = (0, v^x, 0, 0) spatial (orthogonal to n).
# T_ab = (rho+p) u_a u_b + p g_ab (isotropic comoving pressure p = p_r).
# ---------------------------------------------------------------------------
def _T_input_tilted(metric, coords, rho_val, p_val, vx_val):
    g = metric(coords)
    alpha = metric.lapse(coords)
    beta_up = metric.shift(coords)  # beta^i (3,)
    n_up = jnp.array([1.0 / alpha, -beta_up[0] / alpha, -beta_up[1] / alpha, -beta_up[2] / alpha])
    v_up = jnp.array([0.0, vx_val, 0.0, 0.0])  # spatial velocity, x-directed
    Gamma = 1.0 / jnp.sqrt(jnp.maximum(1.0 - vx_val**2, 1e-30))
    u_up = Gamma * (n_up + v_up)
    u_down = g @ u_up
    return (rho_val + p_val) * jnp.outer(u_down, u_down) + p_val * g


def verify_sshell():
    metric = sshell_default()  # v_s=0 static, rho_0=1e-4
    profiles = constant_density_profiles(R_1=R_1, R_2=R_2, rho_0=1e-4)
    rho_fn = jax.jit(profiles.density)
    p_fn = jax.jit(profiles.radial_pressure)

    vac_H, vac_M, src_H, src_M, sc = [], [], [], [], []
    for r in R_PROBES:
        coords = jnp.array([0.0, float(r), 0.0, 0.0])
        rho = float(rho_fn(jnp.float64(r)))
        p = float(p_fn(jnp.float64(r)))

        v = normalized_residuals(metric, coords)
        vac_H.append(float(v["epsilon_H"])); vac_M.append(float(v["epsilon_M"]))

        # Static flow-orthogonal: E = rho, S_i = 0
        s = normalized_residuals(
            metric, coords,
            energy_density=jnp.float64(rho),
            momentum_density=jnp.zeros(3),
        )
        src_H.append(float(s["epsilon_H"])); src_M.append(float(s["epsilon_M"]))

        T_in = _T_input_static(metric, coords, jnp.float64(rho), jnp.float64(p))
        scr = stress_energy_residual(metric, coords, T_input=T_in)
        sc.append(float(scr["relative_residual"]))

    return {
        "vacuum_eps_H": _peak_mean(vac_H), "vacuum_eps_M": _peak_mean(vac_M),
        "source_eps_H": _peak_mean(src_H), "source_eps_M": _peak_mean(src_M),
        "source_consistency_rel": _peak_mean(sc),
    }


def verify_tshell():
    metric = tshell_default()  # v_0=0.1
    profiles = constant_velocity_profiles(R_1=R_1, R_2=R_2, rho_0=1e-4, v_0=0.1)
    E_fn = jax.jit(profiles.eulerian_energy)
    Sx_fn = jax.jit(profiles.momentum_density_x)  # contravariant S^x
    rho_fn = jax.jit(profiles.density)
    p_fn = jax.jit(profiles.radial_pressure)
    vx_fn = jax.jit(profiles.velocity_x)

    vac_H, vac_M, src_H, src_M, sc = [], [], [], [], []
    for r in R_PROBES:
        coords = jnp.array([0.0, float(r), 0.0, 0.0])
        E = float(E_fn(jnp.float64(r)))
        Sx_contra = float(Sx_fn(jnp.float64(r)))
        rho = float(rho_fn(jnp.float64(r)))
        p = float(p_fn(jnp.float64(r)))
        vx = float(vx_fn(jnp.float64(r)))

        v = normalized_residuals(metric, coords)
        vac_H.append(float(v["epsilon_H"])); vac_M.append(float(v["epsilon_M"]))

        # Lowered S_x = gamma_xx S^x; on x-axis gamma_xx = e^{2Lambda}
        gamma = metric.spatial_metric(coords)
        gamma_xx = float(gamma[0, 0])
        S_lower = jnp.array([gamma_xx * Sx_contra, 0.0, 0.0])
        s = normalized_residuals(
            metric, coords,
            energy_density=jnp.float64(E),
            momentum_density=S_lower,
        )
        src_H.append(float(s["epsilon_H"])); src_M.append(float(s["epsilon_M"]))

        T_in = _T_input_tilted(metric, coords, jnp.float64(rho), jnp.float64(p), jnp.float64(vx))
        scr = stress_energy_residual(metric, coords, T_input=T_in)
        sc.append(float(scr["relative_residual"]))

    return {
        "vacuum_eps_H": _peak_mean(vac_H), "vacuum_eps_M": _peak_mean(vac_M),
        "source_eps_H": _peak_mean(src_H), "source_eps_M": _peak_mean(src_M),
        "source_consistency_rel": _peak_mean(sc),
    }


def verify_fuchs():
    metric = fuchs_default()  # canonical Gaussian-smoothed, v_s=0.02
    # The metric was built from these smoothed (rho, P) grids; use them as the
    # convention-consistent prescribed Eulerian source (static, flow-orthogonal).
    con = build_fuchs_construction()  # default paper params matching fuchs_default
    r_grid = con.r_grid
    rho_grid = con.rho_smoothed
    P_grid = con.P_smoothed

    vac_H, vac_M, src_H, src_M, sc = [], [], [], [], []
    for r in R_PROBES:
        coords = jnp.array([0.0, float(r), 0.0, 0.0])
        rho = float(_interp1d(jnp.float64(r), r_grid, rho_grid))
        p = float(_interp1d(jnp.float64(r), r_grid, P_grid))

        v = normalized_residuals(metric, coords)
        vac_H.append(float(v["epsilon_H"])); vac_M.append(float(v["epsilon_M"]))

        # Fuchs: static shell, no matter tilt -> E = rho_smoothed, S_i = 0.
        s = normalized_residuals(
            metric, coords,
            energy_density=jnp.float64(rho),
            momentum_density=jnp.zeros(3),
        )
        src_H.append(float(s["epsilon_H"])); src_M.append(float(s["epsilon_M"]))

        T_in = _T_input_static(metric, coords, jnp.float64(rho), jnp.float64(p))
        scr = stress_energy_residual(metric, coords, T_input=T_in)
        sc.append(float(scr["relative_residual"]))

    return {
        "vacuum_eps_H": _peak_mean(vac_H), "vacuum_eps_M": _peak_mean(vac_M),
        "source_eps_H": _peak_mean(src_H), "source_eps_M": _peak_mean(src_M),
        "source_consistency_rel": _peak_mean(sc),
    }


def main():
    out = {
        "config": {
            "R_1": R_1, "R_2": R_2, "n_probes": N_PROBES,
            "r_probes": R_PROBES.tolist(),
            "sshell": "sshell_default(v_s=0), rho_0=1e-4, static flow-orthogonal (E=rho, S_i=0)",
            "tshell": "tshell_default(v_0=0.1), rho_0=1e-4, tilted (E=Eulerian, S_x lowered=gamma_xx*S^x)",
            "fuchs": "fuchs_default() canonical Gaussian-smoothed, v_s=0.02; source=rho_smoothed (static, S_i=0)",
        },
        "note_vacuum": "Vacuum eps_H = normalized_residuals(metric, coords) with E=0 (published call, sweep.py:209). "
                       "For a matter shell H_vac = R+K^2-K_sq ~ 16 pi rho (matter magnitude), NOT constraint residual.",
        "note_momentum_convention": "S_i passed to normalized_residuals is LOWERED index. T-shell solver stores "
                                    "contravariant S^x; lowered via gamma_xx on x-axis.",
    }
    print("Verifying S-shell...")
    out["sshell"] = verify_sshell()
    print("Verifying T-shell...")
    out["tshell"] = verify_tshell()
    print("Verifying Fuchs (canonical)...")
    out["fuchs"] = verify_fuchs()

    out_path = Path(__file__).resolve().parents[1] / "output" / "constraint_residual_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")

    # Compact table
    def pk(d):
        return "nan" if d["peak"] is None else f"{d['peak']:.3e}"
    print("\n=== PEAK VALUES (in-shell r in [10,20]) ===")
    print(f"{'metric':<8} {'vac_eps_H':>11} {'src_eps_H':>11} {'srcConsist':>11} "
          f"{'vac_eps_M':>11} {'src_eps_M':>11}")
    for m in ("sshell", "tshell", "fuchs"):
        d = out[m]
        print(f"{m:<8} {pk(d['vacuum_eps_H']):>11} {pk(d['source_eps_H']):>11} "
              f"{pk(d['source_consistency_rel']):>11} {pk(d['vacuum_eps_M']):>11} "
              f"{pk(d['source_eps_M']):>11}")


if __name__ == "__main__":
    main()
