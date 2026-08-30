"""Box-size sensitivity of the full-grid Type-I fraction.

A full-grid Hawking-Ellis type fraction is dominated by the near-vacuum exterior,
so it is set by the size of the computational box, not by the wall. Sweeping the
box at fixed resolution shows the full-grid Type-I fraction drift toward 100% as
the box grows, while the wall-restricted fraction stays fixed. Only the
wall-restricted fraction is a physical statement.

Outputs
- results/box_sensitivity.json
"""
from __future__ import annotations

import os

import jax
from _json_io import dump_json

jax.config.update("jax_enable_x64", True)

from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions import certify_grid_frame_free, type_fractions
from warpax.energy_conditions.filtering import shape_function_mask
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import GridSpec, build_coord_batch

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")


def main():
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    N = 50
    rows = []
    for half in (3.0, 5.0, 8.0, 12.0):
        grid = GridSpec(bounds=[(-half, half)] * 3, shape=(N, N, N))
        curv = evaluate_curvature_grid(metric, grid, batch_size=4096)
        ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)
        full = type_fractions(ff)
        coords = build_coord_batch(grid, t=0.0)
        wall_mask = shape_function_mask(metric, coords, grid.shape)
        wall = type_fractions(ff, mask=wall_mask)
        rows.append({
            "half_box": half,
            "full_type_i": full["frac_type_i"],
            "wall_type_i": wall["frac_type_i"],
            "wall_n": wall["n_selected"],
        })
        print(f"  box=+/-{half:<4}: full Type-I={full['frac_type_i']*100:.1f}%  "
              f"wall Type-I={wall['frac_type_i']*100:.1f}%  wall_n={wall['n_selected']}")
    dump_json({"metric": "Alcubierre", "v_s": 0.5, "N": N, "rows": rows},
              os.path.join(RESULTS_DIR, "box_sensitivity.json"))


if __name__ == "__main__":
    main()
