"""Calibration probe (scratch): wall-cell count of the graded grid vs (a, N).

Counts clustered x-nodes whose normalized shape function lies in [0.05, 0.95]
(same band as analysis.construction_adapter.wall_cells) applied to the ACTUAL
sinh-clustered nodes. 1-D only (uses _cosh_stretch directly, bypassing the 3-D
volume-weight build), so it is fast: shape-function evals only, no curvature.
Picks a* and an N-ladder giving ~8/12/16 wall cells at sigma=8, R_b=1, box +-3.
"""
from __future__ import annotations
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.benchmarks import AlcubierreMetric
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric
from warpax.grids._clustered import _cosh_stretch, _infer_wall_radius

BOUNDS = [(-3.0, 3.0)] * 3
V_S = 0.5


def _metrics():
    return {
        "Alcubierre": AlcubierreMetric(v_s=V_S, R=1.0, sigma=8.0),
        "Natario": NatarioMetric(v_s=V_S, R=1.0, sigma=8.0),
        "VanDenBroeck": VanDenBroeckMetric(
            v_s=V_S, R=1.0, sigma=8.0, R_tilde=1.0, alpha_vdb=0.5, sigma_B=8.0
        ),
        "Rodal": RodalMetric(v_s=V_S, R=1.0, sigma=8.0),
    }


def _x_nodes(metric, N, a):
    lo, hi = BOUNDS[0]
    wall_r = _infer_wall_radius(metric, tuple(BOUNDS))
    u_wall = float(jnp.clip((wall_r - lo) / (hi - lo), 0.05, 0.95))
    u = jnp.linspace(0.0, 1.0, N)
    return lo + (hi - lo) * np.asarray(_cosh_stretch(u, u_wall, a)), wall_r


def _shape_on(metric, xs):
    coords = jnp.stack([jnp.zeros_like(xs), jnp.asarray(xs), jnp.zeros_like(xs),
                        jnp.zeros_like(xs)], axis=1)
    return np.asarray(jax.vmap(metric.shape_function_value)(coords))


def wall_cells(metric, N, a):
    xs, wall_r = _x_nodes(metric, N, a)
    f = _shape_on(metric, jnp.asarray(xs))
    fn = (f - f.min()) / (np.ptp(f) + 1e-30)
    band = (fn > 0.05) & (fn < 0.95)
    bx = np.sort(xs[band])
    dxw = float(np.min(np.diff(bx))) if bx.size >= 2 else float("nan")
    return int(band.sum()), dxw


def main():
    A_LIST = [1.2, 2.0, 2.5, 3.0, 3.5, 4.0]
    N_LIST = [40, 50, 60, 70, 80, 100]
    for name, m in _metrics().items():
        print(f"\n=== {name} (sigma=8, R=1, box +-3) ===")
        print("        " + "".join(f"N={n:<9}" for n in N_LIST))
        for a in A_LIST:
            row = []
            for n in N_LIST:
                c, dxw = wall_cells(m, n, a)
                row.append(f"{c}c/{dxw:.3f}")
            print(f"a={a:<4}  " + "".join(f"{r:<11}" for r in row))


if __name__ == "__main__":
    main()
