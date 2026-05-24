"""Null round-trip asymmetry delta_tau across a few T-shell configurations.

Gauge-invariant counterpart to the coordinate shift magnitude
max|beta^x|. Useful as a sanity check that the shift produces a
physical transport signal rather than a coordinate artifact.
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

from warpax.metrics.tshell import tshell_from_profiles
from warpax.metrics.tshell_profiles import constant_velocity_profiles
from warpax.transport.diagnostics import null_round_trip_asymmetry


OUTPUT = Path(__file__).resolve().parents[1] / "results" / "delta_tau.json"

CONFIGS = [
    {"label": "default low-mid", "R_1": 10.0, "R_2": 20.0, "rho_0": 1e-4, "v_0": 0.1},
    {"label": "low-C thin",      "R_1": 16.0, "R_2": 20.0, "rho_0": 1e-5, "v_0": 0.1},
    {"label": "low-C thick",     "R_1":  4.0, "R_2": 20.0, "rho_0": 1e-5, "v_0": 0.1},
    {"label": "high-C thick",    "R_1":  4.0, "R_2": 20.0, "rho_0": 1e-3, "v_0": 0.1},
    {"label": "high-v_0",        "R_1": 10.0, "R_2": 20.0, "rho_0": 1e-4, "v_0": 0.2},
]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    emitter = jnp.array([0.0, -25.0, 0.0, 0.0])
    receiver = jnp.array([0.0,  25.0, 0.0, 0.0])

    results = []
    for cfg in CONFIGS:
        print(f"\n{cfg['label']}: R_1={cfg['R_1']}, R_2={cfg['R_2']}, "
              f"rho_0={cfg['rho_0']:.0e}, v_0={cfg['v_0']}")
        t0 = time.time()
        profiles = constant_velocity_profiles(
            R_1=cfg["R_1"], R_2=cfg["R_2"],
            rho_0=cfg["rho_0"], v_0=cfg["v_0"],
        )
        m = tshell_from_profiles(profiles)
        dt = null_round_trip_asymmetry(m, emitter, receiver,
                                       tau_max=80.0, num_points=600)
        rec = {**cfg, "delta_tau": float(dt), "elapsed_s": time.time() - t0}
        print(f"  delta_tau = {float(dt):+.5e}  ({rec['elapsed_s']:.1f}s)")
        results.append(rec)

    OUTPUT.write_text(json.dumps({"results": results}, indent=2))
    print(f"\n  -> {OUTPUT}")


if __name__ == "__main__":
    main()
