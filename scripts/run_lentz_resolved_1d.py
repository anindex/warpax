"""Resolved 1D characterization of the Lentz wall.

The Lentz wall cannot be sampled on a practical 3D grid: at R=100 on a domain
that contains it, N=50 gives 0.02 cells across the wall. It is therefore excluded
from every 3D fraction in this paper. The wall itself is smooth away from the L1
(diamond) edges, so a fine 1D radial cut resolves it. On that cut the wall is
Type IV and violates the NEC and DEC throughout, exotic like the other bubble
walls and consistent with Celmaster-Rubin.

Outputs
- results/lentz_resolved_1d.json
"""

from __future__ import annotations

import os

import jax
from _json_io import dump_json

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.energy_conditions.frame_free import certify_point_frame_free
from warpax.geometry import compute_curvature_chain
from warpax.metrics import LentzMetric

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")


def main():
    metric = LentzMetric(v_s=0.5, R=100.0, sigma=8.0)
    xs = np.linspace(99.0, 101.0, 1201)  # dr ~ 0.0017 << wall width 0.27
    types = {1: 0, 2: 0, 3: 0, 4: 0}
    nec, dec = [], []
    for x in xs:
        res = compute_curvature_chain(metric, jnp.array([0.0, float(x), 0.05, 0.0]))
        if float(jnp.max(jnp.abs(res.stress_energy))) < 1e-9:
            continue
        out = certify_point_frame_free(res.stress_energy, res.metric, res.metric_inv)
        types[int(out["he_type"])] += 1
        nec.append(float(out["nec"]))
        dec.append(float(out["dec"]))
    tot = sum(types.values())
    result = {
        "v_s": 0.5,
        "R": 100.0,
        "sigma": 8.0,
        "n_wall_points": tot,
        "dr": float(xs[1] - xs[0]),
        "type_iv_frac": types[4] / tot,
        "nec_min": min(nec),
        "nec_violated_frac": sum(v < -1e-10 for v in nec) / tot,
        "dec_min": min(dec),
        "dec_violated_frac": sum(v < -1e-10 for v in dec) / tot,
    }
    print(
        f"  resolved 1D wall points: {tot}, Type-IV frac={result['type_iv_frac']:.2f}, "
        f"NEC min={result['nec_min']:.3f} (violated {result['nec_violated_frac']:.0%})"
    )
    dump_json(result, os.path.join(RESULTS_DIR, "lentz_resolved_1d.json"))


if __name__ == "__main__":
    main()
