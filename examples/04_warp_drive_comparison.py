
"""Warp drive metric comparison with Hawking–Ellis classification.

Surveys all five warp drive metrics in warpax at a representative bubble-wall
point, computing curvature, Hawking–Ellis stress-energy type, and
observer-robust energy condition margins.  Shows how violations scale with
warp bubble velocity.

Demonstrates:
- Constructing every warp drive metric (Lentz, Natário, Rodal, VDB, WarpShell)
- Hawking–Ellis Type I/II/III/IV classification
- Observer-robust NEC/WEC margins vs Eulerian-frame margins
- Velocity scaling of exotic matter requirements
"""

import jax.numpy as jnp

from warpax.metrics import (
    LentzMetric,
    NatarioMetric,
    RodalMetric,
    VanDenBroeckMetric,
    WarpShellMetric,
)
from warpax.geometry import compute_curvature_chain, kretschner_scalar
from warpax.energy_conditions import (
    classify_hawking_ellis,
    verify_point,
    compute_eulerian_ec,
)

# Build metrics at v_s = 0.5
metrics = {
    "Alcubierre": None,  # filled below (from benchmarks for analytical comparison)
    "Lentz":      LentzMetric(v_s=0.5, R=100.0, sigma=8.0),
    "Natário":    NatarioMetric(v_s=0.5, R=1.0, sigma=8.0),
    "Rodal":      RodalMetric(v_s=0.5, R=1.0, sigma=8.0),
    "VDB":        VanDenBroeckMetric(v_s=0.5, R=1.0, sigma=8.0),
    "WarpShell":  WarpShellMetric(v_s=0.5),
}

# Include Alcubierre from benchmarks for a complete picture
from warpax.benchmarks import AlcubierreMetric
metrics["Alcubierre"] = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)

# Probe each metric at a bubble-wall point
print("Warp Drive Metric Comparison (v_s = 0.5)")
print("=" * 72)
print(f"{'Metric':<12s} {'HE Type':>8s} {'Euler NEC':>12s} {'Robust NEC':>12s} "
      f"{'Robust WEC':>12s} {'Kretschner':>12s}")
print("-" * 72)

for name, metric in metrics.items():
    # Choose probe point near the bubble wall for each metric
    if name == "WarpShell":
        # WarpShell has R1=10, R2=20; probe at the outer shell wall
        coords = jnp.array([0.0, 15.0, 0.0, 0.0])
    elif name == "Lentz":
        # Lentz uses L1 distance with R=100; probe at wall d ≈ R
        coords = jnp.array([0.0, 99.0, 0.5, 0.0])
    elif name == "Rodal":
        # Rodal has R=1, sigma=8; probe at wall with slight offset
        coords = jnp.array([0.0, 1.0, 0.3, 0.0])
    else:
        # Standard bubble: probe at r ≈ R
        coords = jnp.array([0.0, 1.0, 0.0, 0.0])

    # Curvature chain
    result = compute_curvature_chain(metric, coords)
    K = kretschner_scalar(result.riemann, result.metric, result.metric_inv)

    # Hawking–Ellis classification of stress-energy
    g_inv = result.metric_inv
    T_mixed = jnp.einsum("ab,bc->ac", g_inv, result.stress_energy)
    he = classify_hawking_ellis(T_mixed, result.metric)

    # Observer-robust EC verification
    ec = verify_point(result.stress_energy, result.metric, g_inv)

    # Eulerian-frame EC for comparison
    eul = compute_eulerian_ec(result.stress_energy, result.metric, g_inv)

    print(f"{name:<12s} {'Type ' + str(int(he.he_type)):>8s} {eul['nec']:>12.4e} "
          f"{ec.nec_margin:>12.4e} {ec.wec_margin:>12.4e} {float(K):>12.4e}")

# Velocity scaling for Alcubierre metric
print("\n")
print("Alcubierre NEC Violation vs Velocity")
print("=" * 50)
print(f"{'v_s':>6s} {'Eulerian NEC':>14s} {'Robust NEC':>14s} {'Ratio':>8s}")
print("-" * 50)

for v_s in [0.1, 0.3, 0.5, 0.7, 0.9]:
    metric = AlcubierreMetric(v_s=v_s, R=1.0, sigma=8.0)
    coords = jnp.array([0.0, 1.0, 0.0, 0.0])

    result = compute_curvature_chain(metric, coords)
    g_inv = result.metric_inv

    ec = verify_point(result.stress_energy, result.metric, g_inv)
    eul = compute_eulerian_ec(result.stress_energy, result.metric, g_inv)

    eul_nec = eul['nec']
    rob_nec = ec.nec_margin
    ratio = rob_nec / eul_nec if abs(eul_nec) > 1e-20 else float('nan')

    print(f"{v_s:>6.1f} {eul_nec:>14.6e} {rob_nec:>14.6e} {ratio:>8.2f}")

print("\nRatio > 1 means the robust analysis finds stronger violations than Eulerian.")
print("This demonstrates why observer-robust verification is essential.")
