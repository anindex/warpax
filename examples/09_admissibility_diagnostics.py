"""Admissibility diagnostics for the Fuchs warp shell.

Runs the warp-shell diagnostic stack on the Fuchs et al. (CQG 2024)
constant-velocity subluminal shell:

1. 3+1 ADM decomposition (lapse, shift, extrinsic curvature).
2. Vacuum Hamiltonian and momentum constraint residuals in the cavity and the
   exterior. Inside the shell the vacuum-form constraint equals ``16 pi E``
   of the matter and is reported as the Eulerian energy density, not as a
   residual. ``normalized_residuals`` floors its denominator at 1, so for
   curvature scales below 1 ``eps_H`` and ``eps_M`` are absolute residuals
   and are read against the printed curvature scale.
3. ADM mass with Richardson extrapolation against the construction mass.
4. Junction surface stress-energy at the inner boundary ``R_1``.
5. Geodesic-deviation tidal diagnostic.
6. Frame-free energy-condition verdict across the shell, with the Type-I
   slack normalized by the stress-energy scale.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from warpax.adm import adm_mass_richardson, falloff_check
from warpax.constraints import normalized_residuals
from warpax.geometry import adm_split, compute_curvature_chain
from warpax.junction import surface_stress_energy
from warpax.metrics import fuchs_default
from warpax.optimization.ec_constraints import ec_feasibility_frame_free
from warpax.transport import geodesic_deviation_diagnostic

metric = fuchs_default()
R_1, R_2 = float(metric.R_1), float(metric.R_2)
r_mid = 0.5 * (R_1 + R_2)
print(f"Metric: {metric.name()}")
print(f"  v_s = {metric.v_s},  R_1 = {R_1},  R_2 = {R_2},  M = {metric.total_mass:.6f}")
print()

print("1. ADM decomposition")
print("-" * 50)
for label, r in [("cavity", 1.0), ("shell", r_mid), ("exterior", 1.5 * R_2)]:
    adm = adm_split(metric, jnp.array([0.0, r, 0.0, 0.0]))
    K_max = float(jnp.max(jnp.abs(adm.extrinsic_curvature)))
    print(
        f"  r={r:5.1f}  ({label:8s})  alpha={float(adm.lapse):.6f}  "
        f"beta^x={float(adm.shift_upper[0]):+.6f}  max|K|={K_max:.2e}"
    )
print()

print("2. Constraint residuals (vacuum sources)")
print("-" * 50)
for label, r in [("cavity", 1.0), ("shell", r_mid), ("exterior", 1.5 * R_2)]:
    res = normalized_residuals(metric, jnp.array([0.0, r, 0.0, 0.0]))
    scale = float(abs(res["R_spatial"]) + abs(res["K_trace"]) ** 2 + abs(res["K_sq"]))
    if label == "shell":
        E = float(res["H_raw"]) / (16.0 * jnp.pi)
        print(f"  r={r:5.1f}  ({label:8s})  H/16pi = {E:+.4e}  (Eulerian energy density)")
        continue
    print(
        f"  r={r:5.1f}  ({label:8s})  eps_H={float(res['epsilon_H']):.2e}  "
        f"eps_M={float(res['epsilon_M']):.2e}  curvature scale={scale:.2e}"
    )
print()

print("3. ADM mass")
print("-" * 50)
mass = adm_mass_richardson(metric, radii=[50.0, 100.0, 200.0, 400.0])
for r_surf, M in zip(mass["radii"], mass["M_values"], strict=True):
    print(f"  r_surface={r_surf:6.1f}  M_ADM={M:+.6e}")
print(
    f"  Extrapolated M_ADM={mass['M_extrapolated']:+.6e}  "
    f"(order {mass['convergence_order']:.2f}, valid={mass['valid']}); "
    f"construction mass {metric.total_mass:.6e}"
)
falloff = falloff_check(metric, r_test=200.0)
print(f"  Falloff O(1/r) in every component: {all(falloff.values())}")
print()

print(f"4. Junction surface stress-energy at R_1 = {R_1}")
print("-" * 50)
S_ab = surface_stress_energy(
    metric,
    lambda c: jnp.sqrt(c[1] ** 2 + c[2] ** 2 + c[3] ** 2) - R_1,
    jnp.array([0.0, R_1 - 0.1, 0.0, 0.0]),
    jnp.array([0.0, R_1 + 0.1, 0.0, 0.0]),
)
print(f"  S_tt = {float(S_ab[0, 0]):+.4e}")
print(f"  S_xx = {float(S_ab[1, 1]):+.4e}")
print(f"  max|S| = {float(jnp.max(jnp.abs(S_ab))):.4e}")
print()

print("5. Geodesic deviation (tidal acceleration)")
print("-" * 50)
for r in [1.0, r_mid, 1.5 * R_2]:
    A = geodesic_deviation_diagnostic(metric, jnp.array([0.0, r, 0.0, 0.0]))
    print(f"  r={r:5.1f}  A_geo={float(A):.4e}")
print()

print("6. Frame-free energy conditions across the shell")
print("-" * 50)
r_probes = jnp.linspace(R_1, R_2, 15)
feas = ec_feasibility_frame_free(metric, r_probes)
T_probes = jax.vmap(
    lambda r: compute_curvature_chain(metric, jnp.array([0.0, r, 0.0, 0.0])).stress_energy
)(r_probes)
T_scale = float(jnp.max(jnp.abs(T_probes)))
print(f"  Probes: {len(r_probes)} on [{R_1}, {R_2}],  Type I: {feas.n_type_i}/{feas.n_total}")
print(f"  Feasible (NEC, WEC, DEC): {feas.feasible}")
print(f"  Worst Type-I slack: {feas.worst_margin:+.4e}  (max|T_ab| = {T_scale:.4e})")
print(f"  Slack / max|T_ab|:  {feas.worst_margin / T_scale:+.4e}")
print()
