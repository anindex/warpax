"""Algebraic momentum-discriminant cross-check of the Type-IV map.

Independent, eig-free confirmation of the Hawking-Ellis Type-IV classification.
At each wall grid point (v_s=0.5) we decompose ``T`` in the Eulerian frame into
energy density ``rho``, momentum density ``j^i`` and spatial stress ``S``, then
form the *algebraic* momentum discriminant

    Delta = (rho + S_par)^2 - 4 |j|^2 ,   S_par = S(jhat, jhat), jhat = j/|j|,

and label a point "Delta-TypeIV" where ``Delta < 0``. This is the closed-form
momentum-plane witness (``T_ab k^a k^b = rho + S_par - 2|j| < 0`` for the null
``k = n +/- jhat`` iff ``Delta < 0``). We compare it point-by-point to the
eig-based ``classify_hawking_ellis`` Type-IV label and report the agreement rate.

PHYSICS EXPECTATION: the momentum-plane drives (Alcubierre, Natario) agree ~100%
-- their complex eigenpair is sourced entirely by the ``j`` plane. Rodal is 100%
Type I (no Type-IV content at all, so the two labels agree trivially with zero
active points). Van den Broeck is the KNOWN EXCEPTION: its conformal spatial
factor opens a *transverse* complex pair that the full 4x4 eig detects (Type IV)
but that leaves ``Delta >= 0`` (the momentum witness is non-negative). We
quantify that conformal-channel exception count.

Outputs
-------
- results/delta_crosscheck.json
"""

from __future__ import annotations

import os

import jax
from _json_io import dump_json

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions import certify_grid_frame_free
from warpax.energy_conditions.filtering import shape_function_mask
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.grids import wall_clustered
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")

ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]
FAMILY = {
    "Alcubierre": (AlcubierreMetric, {"R": 1.0, "sigma": 8.0}, [(-3.0, 3.0)] * 3),
    "Natário": (NatarioMetric, {"R": 1.0, "sigma": 8.0}, [(-3.0, 3.0)] * 3),
    "Van den Broeck": (
        VanDenBroeckMetric,
        {"R": 1.0, "sigma": 8.0, "R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0},
        [(-3.0, 3.0)] * 3,
    ),
    "Rodal": (RodalMetric, {"R": 100.0, "sigma": 0.03}, [(-300.0, 300.0)] * 3),
}
V_S = 0.5
N = 60  # wall-clustered resolution


def _instantiate(name):
    cls, kw, _ = FAMILY[name]
    return cls(v_s=V_S, **kw)


def _eulerian_delta(T_ab, g_ab, g_inv):
    """Momentum discriminant Delta = (rho+S_par)^2 - 4|j|^2 (mirrors the witness)."""
    n_low = jnp.array([-1.0, 0.0, 0.0, 0.0])
    n_up = g_inv @ n_low
    n_up = n_up / jnp.sqrt(jnp.abs(n_low @ n_up))
    n_low2 = g_ab @ n_up
    proj = jnp.eye(4) + jnp.outer(n_up, n_low2)  # h^a_b = delta + n^a n_b
    T_mixed = g_inv @ T_ab
    rho = n_up @ (T_ab @ n_up)
    j_up = -(proj @ (T_mixed @ n_up))
    j2 = j_up @ (g_ab @ j_up)
    jmag = jnp.sqrt(jnp.clip(j2, min=0.0))
    jhat = j_up / jnp.where(jmag > 1e-30, jnp.sqrt(jnp.clip(j2, min=1e-300)), 1.0)
    S_par = jhat @ (g_ab @ ((proj @ (T_mixed @ proj)) @ jhat))
    delta = (rho + S_par) ** 2 - 4.0 * j2
    return delta, jmag


_delta_grid = jax.jit(jax.vmap(_eulerian_delta))


def _analyze(name):
    metric = _instantiate(name)
    bounds = FAMILY[name][2]
    grid = wall_clustered(metric, bounds, (N, N, N), a=1.2)
    curv = evaluate_curvature_grid(metric, grid, batch_size=4096)
    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)

    delta, jmag = _delta_grid(
        curv.stress_energy.reshape(-1, 4, 4),
        curv.metric.reshape(-1, 4, 4),
        curv.metric_inv.reshape(-1, 4, 4),
    )
    delta = np.asarray(delta)
    he = np.asarray(ff.he_types).ravel()
    imag = np.max(np.abs(np.asarray(ff.eigenvalues_imag).reshape(-1, 4)), axis=-1)

    coords = build_coord_batch(grid, t=0.0)
    wall = np.asarray(shape_function_mask(metric, coords, grid.shape)).ravel().astype(bool)

    eig_iv = he == 4.0
    delta_iv = delta < 0.0

    def _stats(sel):
        e = eig_iv[sel]
        d = delta_iv[sel]
        im = imag[sel]
        n = int(np.sum(sel))
        agree = int(np.sum(e == d))
        # eig says IV but Delta>=0: complex pair NOT sourced by the momentum
        # plane (VdB conformal channel; Natario vortical/transverse channel).
        exc = e & ~d
        n_exc = int(np.sum(exc))
        # Delta<0 but eig not Type IV (e.g. Type-I NEC-violating far field).
        delta_only = int(np.sum(~e & d))
        active = e | d
        n_active = int(np.sum(active))
        agree_active = int(np.sum((e == d) & active))
        return {
            "n_points": n,
            "n_eig_typeIV": int(np.sum(e)),
            "n_delta_typeIV": int(np.sum(d)),
            "agreement_rate": (agree / n) if n else float("nan"),
            "n_active": n_active,
            "agreement_rate_active": (agree_active / n_active) if n_active else 1.0,
            "eig_iv_delta_nonneg": n_exc,  # eig-IV & Delta>=0 (off-momentum channel)
            "delta_only": delta_only,  # Delta<0 & not eig-IV
            # median |Im lambda| of the off-momentum exception points: confirms
            # they are genuine complex pairs, not classifier noise.
            "exc_median_abs_imag": float(np.median(im[exc])) if n_exc else 0.0,
        }

    return {"full_grid": _stats(np.ones_like(eig_iv, bool)), "wall": _stats(wall)}


def main():
    print("=" * 72)
    print(f"ALGEBRAIC Delta vs eig Type-IV CROSS-CHECK  (v_s={V_S}, N={N})")
    print("=" * 72)
    results = {}
    for name in ORDER:
        r = _analyze(name)
        results[name] = r
        w = r["wall"]
        g = r["full_grid"]
        print(f"\n{name}:")
        for region, s in (("wall", w), ("full", g)):
            ar = s["agreement_rate"]
            ara = s["agreement_rate_active"]
            print(
                f"  [{region:4s}] n={s['n_points']:>7d}  eig-IV={s['n_eig_typeIV']:>6d}  "
                f"Delta-IV={s['n_delta_typeIV']:>6d}  agree={ar * 100:6.2f}%  "
                f"agree(active)={ara * 100:6.2f}%  "
                f"eig-IV&Delta>=0={s['eig_iv_delta_nonneg']:>5d}"
                f"(med|Im|={s['exc_median_abs_imag']:.1e})  "
                f"delta_only={s['delta_only']:>5d}"
            )

    out = {
        "params": {
            "v_s": V_S,
            "N": N,
            "note": "matched family R_b=1 sigma=8; Rodal native R=100 sigma=0.03. "
            "eig_iv_delta_nonneg = eig Type-IV but Delta>=0 (off-momentum-plane "
            "channel: VdB conformal, Natario vortical). Wall = shape function in "
            "[0.1,0.9]; full-grid delta_only is dominated by Type-I NEC-violating "
            "far field (Delta<0 is a NEC-violation witness, broader than Type IV).",
        },
        "order": ORDER,
        "results": results,
    }
    dump_json(out, os.path.join(RESULTS_DIR, "delta_crosscheck.json"))
    print(f"\nWrote {os.path.join(RESULTS_DIR, 'delta_crosscheck.json')}")


if __name__ == "__main__":
    main()
