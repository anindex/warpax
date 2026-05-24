"""Hamiltonian K-term decomposition and angular EC sampling diagnostics.

Default T-shell: R_1=10, R_2=20, rho_0=1e-4, v_0=0.1.

K-term decomposition: across radial probes through the shell, decompose the
Hamiltonian constraint H = R + (K^2 - K_ij K^ij) - 16 pi E and quantify the
extrinsic-curvature part |K^2 - K_ij K^ij| against |16 pi E|, |R|, and the
absolute Hamiltonian numerator |H|. Expectation: K-terms subdominant.

Angular EC sampling: sample worst min(NEC, WEC, DEC) margin over the 2-sphere
of spatial directions (observer angular sweep) AND over spatial probe POINTS
at multiple polar angles at fixed radius (off the x-axis), at the binding
radii r ~ R_1 = 10 and r ~ R_2 = 20. Compare to the x-axis radial-line
value.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from warpax.metrics.tshell import tshell_default
from warpax.geometry import adm_split, compute_curvature_chain
from warpax.constraints.residuals import (
    _spatial_ricci_scalar,
    normalized_residuals,
)
from warpax.energy_conditions import verify_point

OUTPUT = Path(__file__).resolve().parents[1] / "results" / "tshell_kterm_angular.json"

R_1 = 10.0
R_2 = 20.0
V_0 = 0.1


# Eulerian energy density E = T_{ab} n^a n^b
def eulerian_E(metric, coords):
    """Eulerian energy density E = T_{ab} n^a n^b from the autodiff T_{ab}."""
    curv = compute_curvature_chain(metric, coords)
    T = curv.stress_energy
    g_inv = curv.metric_inv
    # ADM unit normal n^a = (1/alpha, -beta^i/alpha); alpha = (-g^{00})^{-1/2}
    alpha = 1.0 / jnp.sqrt(-g_inv[0, 0])
    n_up = jnp.array([
        1.0 / alpha,
        -g_inv[0, 1] * alpha,
        -g_inv[0, 2] * alpha,
        -g_inv[0, 3] * alpha,
    ])
    return jnp.einsum("a,ab,b->", n_up, T, n_up)


# Hamiltonian K-term decomposition along radial (x-axis) probes
def kterm_decomposition(metric):
    margin = 0.02 * (R_2 - R_1)
    r_probes = jnp.linspace(R_1 + margin, R_2 - margin, 25)

    rows = []
    for r in r_probes:
        coords = jnp.array([0.0, float(r), 0.0, 0.0], dtype=jnp.float64)
        adm = adm_split(metric, coords)
        gamma = adm.spatial_metric
        gamma_inv = adm.spatial_metric_inv
        K = adm.extrinsic_curvature

        K_trace = jnp.einsum("ij,ij->", gamma_inv, K)
        K_sq = jnp.einsum("ij,kl,ik,jl", gamma_inv, gamma_inv, K, K)
        kterm = K_trace**2 - K_sq          # the extrinsic-curvature part of H
        R = _spatial_ricci_scalar(metric, coords)
        E = eulerian_E(metric, coords)
        sixteen_pi_E = 16.0 * jnp.pi * E

        # Source-aware Hamiltonian: H_source = R + (K^2 - K_ijK^ij) - 16piE.
        # (Near-zero since R was built from E; not a useful denominator.)
        H_source = R + kterm - sixteen_pi_E
        # The PUBLISHED residual drops E (calls normalized_residuals w/o E),
        # so its numerator is H_pub = R + (K^2 - K_ijK^ij). This is what
        # eps_H ~5e-3 actually measures -> the meaningful |H| denominator.
        res = normalized_residuals(metric, coords)
        H_pub = R + kterm

        rows.append({
            "r": float(r),
            "abs_kterm": float(jnp.abs(kterm)),
            "abs_R": float(jnp.abs(R)),
            "abs_16piE": float(jnp.abs(sixteen_pi_E)),
            "abs_H_source": float(jnp.abs(H_source)),
            "abs_H_pub": float(jnp.abs(H_pub)),
            "epsilon_H_published": float(res["epsilon_H"]),
            "K_trace": float(K_trace),
            "K_sq": float(K_sq),
            "beta_x": float(metric.shift(coords)[0]),
            "ratio_kterm_over_16piE": float(jnp.abs(kterm) / (jnp.abs(sixteen_pi_E) + 1e-300)),
            "ratio_kterm_over_Hpub": float(jnp.abs(kterm) / (jnp.abs(H_pub) + 1e-300)),
            "ratio_kterm_over_R": float(jnp.abs(kterm) / (jnp.abs(R) + 1e-300)),
        })

    def peak(key):
        return max(row[key] for row in rows)

    def mean(key):
        return sum(row[key] for row in rows) / len(rows)

    return {
        "n_probes": len(rows),
        "r_range": [float(r_probes[0]), float(r_probes[-1])],
        "peak_ratio_kterm_over_16piE": peak("ratio_kterm_over_16piE"),
        "mean_ratio_kterm_over_16piE": mean("ratio_kterm_over_16piE"),
        "peak_ratio_kterm_over_Hpub": peak("ratio_kterm_over_Hpub"),
        "mean_ratio_kterm_over_Hpub": mean("ratio_kterm_over_Hpub"),
        "peak_ratio_kterm_over_R": peak("ratio_kterm_over_R"),
        "mean_ratio_kterm_over_R": mean("ratio_kterm_over_R"),
        "peak_abs_kterm": peak("abs_kterm"),
        "peak_abs_16piE": peak("abs_16piE"),
        "peak_abs_R": peak("abs_R"),
        "peak_epsilon_H_published": peak("epsilon_H_published"),
        "max_abs_beta_x": peak("abs_R") * 0.0 + max(abs(row["beta_x"]) for row in rows),
        "per_probe": rows,
    }


# Angular EC sampling
def worst_min_margin_at_point(metric, coords, n_starts=24):
    """Worst (most negative) of min(NEC, WEC, DEC) at a point via full optimization."""
    curv = compute_curvature_chain(metric, coords)
    T, g, gi = curv.stress_energy, curv.metric, curv.metric_inv
    ec = verify_point(T, g, gi, n_starts=n_starts)
    nec = float(ec.nec_margin)
    wec = float(ec.wec_margin)
    dec = float(ec.dec_margin)
    worst = min(nec, wec, dec)
    return {"nec": nec, "wec": wec, "dec": dec, "min_ndw": worst}


def point_at(r, theta, phi):
    """Spatial probe point at radius r, polar theta (from x-axis), azimuth phi.

    The shift is along x (the axisymmetry axis). We take x = r cos(theta) as
    the symmetry axis so that theta=0 is the x-axis (radial-line) probe.
    """
    x = r * jnp.cos(theta)
    rho = r * jnp.sin(theta)
    y = rho * jnp.cos(phi)
    z = rho * jnp.sin(phi)
    return jnp.array([0.0, float(x), float(y), float(z)], dtype=jnp.float64)


def angular_ec_at_radius(metric, r, label, n_theta=8, n_phi=16, n_starts=24):
    # x-axis radial-line probe (theta=0): the published probe.
    x_axis_coords = jnp.array([0.0, float(r), 0.0, 0.0], dtype=jnp.float64)
    radial = worst_min_margin_at_point(metric, x_axis_coords, n_starts=n_starts)

    # Sweep over spatial probe POINTS on the 2-sphere at fixed |r|.
    # theta in (0, pi] off the x-axis (theta=0 is the radial-line, already done).
    thetas = jnp.linspace(0.0, jnp.pi, n_theta)
    phis = jnp.linspace(0.0, 2 * jnp.pi, n_phi, endpoint=False)

    worst_angular = None
    worst_loc = None
    samples = []
    for th in thetas:
        for ph in phis:
            coords = point_at(r, float(th), float(ph))
            res = worst_min_margin_at_point(metric, coords, n_starts=n_starts)
            rec = {
                "theta": float(th), "phi": float(ph),
                "x": float(coords[1]), "y": float(coords[2]), "z": float(coords[3]),
                **res,
            }
            samples.append(rec)
            if worst_angular is None or rec["min_ndw"] < worst_angular["min_ndw"]:
                worst_angular = rec
                worst_loc = (float(th), float(ph))

    return {
        "label": label,
        "r": float(r),
        "n_theta": n_theta,
        "n_phi": n_phi,
        "n_starts": n_starts,
        "x_axis_radial": {
            "coords_xyz": [float(r), 0.0, 0.0],
            **radial,
        },
        "worst_angular": worst_angular,
        "worst_angular_theta_phi": worst_loc,
        "ratio_worst_angular_over_worst_radial": (
            worst_angular["min_ndw"] / radial["min_ndw"]
            if abs(radial["min_ndw"]) > 1e-300 else float("nan")
        ),
        "radial_captures_binding": worst_angular["min_ndw"] >= radial["min_ndw"] - 1e-9,
        "samples": samples,
    }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    metric = tshell_default(v_0=V_0)

    print("Hamiltonian K-term decomposition ...", flush=True)
    kterm = kterm_decomposition(metric)
    print(
        f"  peak |K-term|/|16piE| = {kterm['peak_ratio_kterm_over_16piE']:.3e}  "
        f"mean = {kterm['mean_ratio_kterm_over_16piE']:.3e}",
        flush=True,
    )

    print("Angular EC sampling at r~R_1=10 ...", flush=True)
    ang_r1 = angular_ec_at_radius(metric, R_1, "inner R_1")
    print(
        f"  x-axis min(NEC,WEC,DEC) = {ang_r1['x_axis_radial']['min_ndw']:+.4e}  "
        f"worst angular = {ang_r1['worst_angular']['min_ndw']:+.4e}",
        flush=True,
    )

    print("Angular EC sampling at r~R_2=20 ...", flush=True)
    ang_r2 = angular_ec_at_radius(metric, R_2, "outer R_2")
    print(
        f"  x-axis min(NEC,WEC,DEC) = {ang_r2['x_axis_radial']['min_ndw']:+.4e}  "
        f"worst angular = {ang_r2['worst_angular']['min_ndw']:+.4e}",
        flush=True,
    )

    out = {
        "config": {"R_1": R_1, "R_2": R_2, "rho_0": 1e-4, "v_0": V_0},
        "kterm": kterm,
        "angular_ec": {"r_R1": ang_r1, "r_R2": ang_r2},
    }
    OUTPUT.write_text(json.dumps(out, indent=2))
    print(f"\n  -> {OUTPUT}")


if __name__ == "__main__":
    main()
