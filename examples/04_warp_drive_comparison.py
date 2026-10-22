"""Warp drive metric comparison with Hawking-Ellis classification.

Surveys six warp drive metrics at representative bubble-wall points,
computing curvature, the Hawking-Ellis stress-energy type, the Eulerian
margins and the observer-robust margins, then shows how the Alcubierre
energy density scales with bubble velocity.

The Eulerian NEC is the exact minimum over the null sphere at Eulerian
normalization; the Eulerian WEC is the energy density seen by the normal
observer. The robust columns search boosted observers up to rapidity
``zeta_max = 5`` (Type I points use exact eigenvalue slacks instead), so a
robust WEC far below the Eulerian one is the signature of a Type IV wall.

Probe points sit at each metric's wall: WarpShell (R1 = 10, R2 = 20) at the
outer shell wall x = 15; Lentz (L1 distance, R = 100) at d ~ R; Rodal
(R = 1, sigma = 8) at r = 1 with a slight y offset; the standard bubbles at
r ~ R = 1. The velocity scan uses the off-axis wall point of example 03,
where the closed-form density ``-(v_s^2 / 32 pi) (df/dr)^2 (y^2 + z^2) / r^2``
is non-zero. The wall there turns Type I at ``v_s = 0.9``, where the robust
WEC becomes the exact eigenvalue slack instead of a value at the rapidity cap.
"""

import jax.numpy as jnp

from warpax.benchmarks import AlcubierreMetric
from warpax.benchmarks.alcubierre import eulerian_energy_density
from warpax.energy_conditions import compute_eulerian_ec, verify_point
from warpax.geometry import compute_curvature_chain, kretschmann_scalar
from warpax.metrics import (
    LentzMetric,
    NatarioMetric,
    RodalMetric,
    VanDenBroeckMetric,
    WarpShellMetric,
)

V_S = 0.5
WALL = jnp.array([0.0, 1.0, 0.0, 0.0])

cases = {
    "Alcubierre": (AlcubierreMetric(v_s=V_S, R=1.0, sigma=8.0), WALL),
    "Lentz": (LentzMetric(v_s=V_S, R=100.0, sigma=8.0), jnp.array([0.0, 99.0, 0.5, 0.0])),
    "Natário": (NatarioMetric(v_s=V_S, R=1.0, sigma=8.0), WALL),
    "Rodal": (RodalMetric(v_s=V_S, R=1.0, sigma=8.0), jnp.array([0.0, 1.0, 0.3, 0.0])),
    "Van den Broeck": (VanDenBroeckMetric(v_s=V_S, R=1.0, sigma=8.0), WALL),
    "WarpShell": (WarpShellMetric(v_s=V_S), jnp.array([0.0, 15.0, 0.0, 0.0])),
}


def margins_at(metric, coords):
    """Curvature, Eulerian and observer-robust margins at one point."""
    result = compute_curvature_chain(metric, coords)
    T, g, g_inv = result.stress_energy, result.metric, result.metric_inv
    K = kretschmann_scalar(result.riemann, g, g_inv)
    return verify_point(T, g, g_inv), compute_eulerian_ec(T, g, g_inv), K


print(f"Warp Drive Metric Comparison (v_s = {V_S})")
print("=" * 88)
print(
    f"{'Metric':<15s} {'HE Type':>8s} {'Euler NEC':>12s} {'Euler WEC':>12s} "
    f"{'Robust NEC':>12s} {'Robust WEC':>12s} {'Kretschmann':>12s}"
)
print("-" * 88)

for name, (metric, coords) in cases.items():
    ec, eul, K = margins_at(metric, coords)
    print(
        f"{name:<15s} {'Type ' + str(int(ec.he_type)):>8s} {eul['nec']:>12.4e} "
        f"{eul['wec']:>12.4e} {ec.nec_margin:>12.4e} {ec.wec_margin:>12.4e} {float(K):>12.4e}"
    )
    assert eul["nec"] < 0 and ec.nec_margin < 0, f"{name}: NEC should be violated at the wall"

print("\n")
print("Alcubierre Eulerian energy density vs velocity")
print("=" * 81)
print(
    f"{'v_s':>6s} {'HE Type':>8s} {'Euler NEC':>12s} {'Euler WEC':>12s} "
    f"{'Closed form':>12s} {'rho / v_s^2':>12s} {'Robust WEC':>12s}"
)
print("-" * 81)

scan_point = jnp.array([0.0, 1.0, 0.5, 0.0])
for v_s in [0.1, 0.3, 0.5, 0.7, 0.9]:
    ec, eul, _ = margins_at(AlcubierreMetric(v_s=v_s, R=1.0, sigma=8.0), scan_point)
    rho = eulerian_energy_density(*scan_point[1:], v_s=v_s, R=1.0, sigma=8.0)
    assert jnp.allclose(eul["wec"], rho, rtol=1e-8), "Eulerian density mismatch"
    print(
        f"{v_s:>6.1f} {'Type ' + str(int(ec.he_type)):>8s} {eul['nec']:>12.4e} "
        f"{eul['wec']:>12.4e} {rho:>12.4e} {rho / v_s**2:>12.4e} {ec.wec_margin:>12.4e}"
    )

print("\nThe Eulerian density matches the closed form and scales exactly as v_s^2.")
print("Boosted observers see a far larger WEC violation than the Eulerian one,")
print("which is why observer-robust verification is essential.")
