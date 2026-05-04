
"""Admissibility diagnostics for the Fuchs warp shell.

Demonstrates the v0.3.0 infrastructure: 3+1 ADM decomposition, constraint
residuals, ADM mass, source-consistency, junction conditions, and tidal
diagnostics, all evaluated on the Fuchs et al. (CQG 2024) constant-velocity
subluminal warp shell.

Outputs a structured admissibility report covering:
  1. ADM decomposition (lapse, shift, K_{ij})
  2. Hamiltonian / momentum constraint residuals
  3. ADM mass via surface integral
  4. Source-consistency check (DeltaT = T_input - G/(8pi))
  5. Junction surface stress-energy
  6. Tidal deviation (geodesic hazard)
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from warpax.adm import adm_mass, falloff_check
from warpax.constraints import normalized_residuals, stress_energy_residual
from warpax.geometry import adm_split
from warpax.junction import surface_stress_energy
from warpax.metrics import fuchs_default
from warpax.transport import geodesic_deviation_diagnostic

# -- Metric ---------------------------------------------------------------
metric = fuchs_default()
print(f"Metric: {metric.name()}")
print(f"  v_s = {metric.v_s},  R_1 = {metric.R_1},  R_2 = {metric.R_2}")
print()

# -- 1. ADM decomposition at three radii ----------------------------------
print("1. ADM decomposition")
print("-" * 50)
for label, r in [("interior", 1.0), ("shell", 15.0), ("exterior", 30.0)]:
    coords = jnp.array([0.0, r, 0.0, 0.0])
    adm = adm_split(metric, coords)
    K_max = float(jnp.max(jnp.abs(adm.extrinsic_curvature)))
    print(f"  r={r:5.1f}  ({label:8s})  alpha={float(adm.lapse):.6f}  "
          f"beta^x={float(adm.shift_upper[0]):+.6f}  max|K|={K_max:.2e}")
print()

# -- 2. Constraint residuals -----------------------------------------------
print("2. Constraint residuals")
print("-" * 50)
for r in [1.0, 15.0, 30.0]:
    coords = jnp.array([0.0, r, 0.0, 0.0])
    res = normalized_residuals(metric, coords)
    print(f"  r={r:5.1f}  eps_H={float(res['epsilon_H']):.4e}  "
          f"eps_M={float(res['epsilon_M']):.4e}  "
          f"R={float(res['R_spatial']):+.4e}")
print()

# -- 3. ADM mass -----------------------------------------------------------
print("3. ADM mass")
print("-" * 50)
for r_surf in [50.0, 100.0, 200.0]:
    M = adm_mass(metric, r_surface=r_surf, n_theta=16, n_phi=32)
    print(f"  r_surface={r_surf:6.1f}  M_ADM={float(M):+.6e}")

falloff = falloff_check(metric, r_test=200.0)
print(f"  Falloff check: {falloff}")
print()

# -- 4. Source consistency -------------------------------------------------
print("4. Source consistency (DeltaT = T_input - G/8pi)")
print("-" * 50)
for r in [1.0, 15.0, 30.0]:
    coords = jnp.array([0.0, r, 0.0, 0.0])
    sc = stress_energy_residual(metric, coords, T_input=None)
    print(f"  r={r:5.1f}  max|DeltaT|={float(sc['max_residual']):.4e}")
print()

# -- 5. Junction surface stress-energy ------------------------------------
print("5. Junction surface stress-energy at R_1 boundary")
print("-" * 50)
boundary_fn = lambda c: jnp.sqrt(c[1]**2 + c[2]**2 + c[3]**2) - 10.0
inside = jnp.array([0.0, 9.9, 0.0, 0.0])
outside = jnp.array([0.0, 10.1, 0.0, 0.0])
S_ab = surface_stress_energy(metric, boundary_fn, inside, outside)
print(f"  S_{{tt}} = {float(S_ab[0, 0]):+.4e}")
print(f"  S_{{xx}} = {float(S_ab[1, 1]):+.4e}")
print(f"  max|S| = {float(jnp.max(jnp.abs(S_ab))):.4e}")
print()

# -- 6. Tidal deviation ----------------------------------------------------
print("6. Geodesic deviation (tidal acceleration)")
print("-" * 50)
for r in [1.0, 15.0, 30.0]:
    coords = jnp.array([0.0, r, 0.0, 0.0])
    A = geodesic_deviation_diagnostic(metric, coords)
    print(f"  r={r:5.1f}  A_geo={float(A):.4e}")
print()
