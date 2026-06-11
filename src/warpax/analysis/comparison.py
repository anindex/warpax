"""Eulerian vs observer-robust EC margin comparison.

Per-point ``missed`` flag and severity ratio quantifying violations that
the Eulerian frame misses relative to a boost-optimized observer
(Santiago, Schuster, Visser, *Generic warp drives violate the null
energy condition*, Phys. Rev. D 105, 064038 (2022),
[arXiv:2105.03079](https://arxiv.org/abs/2105.03079)).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from warpax.energy_conditions.verifier import _eulerian_ec_point, verify_grid

_CONDITIONS = ("nec", "wec", "sec", "dec")


class ComparisonResult(NamedTuple):
    """Per-point comparison between Eulerian and robust EC margins.

    Dict-valued fields are keyed by ``"nec"|"wec"|"sec"|"dec"``. ``missed``
    flags ``(eul >= 0) & (rob < -1e-10)``; ``severity`` is ``eul - rob``
    at missed points (zero elsewhere); ``conditional_miss_rate`` is
    ``pct_missed / pct_violated_robust`` (zero when nothing violated).
    """

    eulerian_margins: dict[str, Float[Array, "..."]]
    robust_margins: dict[str, Float[Array, "..."]]
    missed: dict[str, Float[Array, "..."]]
    severity: dict[str, Float[Array, "..."]]
    pct_missed: dict[str, float]
    pct_violated_robust: dict[str, float]
    conditional_miss_rate: dict[str, float]
    classification_stats: dict[str, int | float]
    opt_margins: dict[str, Float[Array, "..."]]
    he_types: Float[Array, "..."]


def compare_eulerian_vs_robust(
    T_field: Float[Array, "... 4 4"],
    g_field: Float[Array, "... 4 4"],
    g_inv_field: Float[Array, "... 4 4"],
    grid_shape: tuple[int, ...],
    n_starts: int = 16,
    zeta_max: float = 5.0,
    batch_size: int = 64,
    key=None,
) -> ComparisonResult:
    """Run both Eulerian and robust EC analysis and compare per-point.

    Eulerian and robust paths must be computed independently. Do not pass
    ``compute_eulerian=True`` to ``verify_grid`` here; that merges the two
    analyses and invalidates the comparison.

    Parameters
    ----------
    T_field : Float[Array, "... 4 4"]
        Stress-energy tensor field, shape ``(*grid_shape, 4, 4)``.
    g_field : Float[Array, "... 4 4"]
        Metric tensor field, shape ``(*grid_shape, 4, 4)``.
    g_inv_field : Float[Array, "... 4 4"]
        Inverse metric field, shape ``(*grid_shape, 4, 4)``.
    grid_shape : tuple[int, ...]
        Shape of the spatial grid (e.g. ``(50, 50, 50)``).
    n_starts : int
        Multi-start count for optimization.
    zeta_max : float
        Maximum rapidity for observer search.
    batch_size : int
        Batch size for memory-safe grid processing.
    key : PRNGKey or None
        Random key for optimization.

    Returns
    -------
    ComparisonResult
        Per-point comparison with missed flags and severity ratios.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    # Pure-host product of static shape dims; avoids a device round-trip.
    n_points = int(np.prod(grid_shape))
    flat_T = T_field.reshape(n_points, 4, 4)
    flat_g = g_field.reshape(n_points, 4, 4)
    flat_g_inv = g_inv_field.reshape(n_points, 4, 4)

    # 1. Eulerian EC: vectorized, fast
    eul_results = jax.vmap(_eulerian_ec_point)(flat_T, flat_g, flat_g_inv)

    # 2. Robust EC: optimizer per point (compute_eulerian=False!)
    robust = verify_grid(
        T_field,
        g_field,
        g_inv_field,
        n_starts=n_starts,
        zeta_max=zeta_max,
        batch_size=batch_size,
        key=key,
        compute_eulerian=False,
    )

    # 3. Build per-condition comparison
    eulerian_margins = {}
    robust_margins = {}
    missed = {}
    severity = {}
    pct_missed = {}
    pct_violated_robust = {}
    conditional_miss_rate = {}
    opt_margins_dict = {}

    # Map condition names to robust ECGridResult fields
    robust_field_map = {
        "nec": robust.nec_margins,
        "wec": robust.wec_margins,
        "sec": robust.sec_margins,
        "dec": robust.dec_margins,
    }

    opt_field_map = {
        "nec": robust.nec_opt_margins,
        "wec": robust.wec_opt_margins,
        "sec": robust.sec_opt_margins,
        "dec": robust.dec_opt_margins,
    }

    for cond in _CONDITIONS:
        eul_margin = eul_results[cond].reshape(grid_shape)
        rob_margin = robust_field_map[cond]

        # Missed: Eulerian says satisfied, robust says violated
        missed_mask = (eul_margin >= 0.0) & (rob_margin < -1e-10)
        sev = jnp.where(missed_mask, eul_margin - rob_margin, 0.0)

        # Statistics
        n_total = float(rob_margin.size)
        pct_m = float(jnp.sum(missed_mask.astype(jnp.float64))) / n_total * 100.0
        pct_v = (
            float(jnp.sum((rob_margin < -1e-10).astype(jnp.float64)))
            / n_total
            * 100.0
        )

        # Conditional miss rate: f_miss|viol = missed / violated
        cond_miss = pct_m / pct_v * 100.0 if pct_v > 0 else 0.0

        eulerian_margins[cond] = eul_margin
        robust_margins[cond] = rob_margin
        missed[cond] = missed_mask
        severity[cond] = sev
        pct_missed[cond] = pct_m
        pct_violated_robust[cond] = pct_v
        conditional_miss_rate[cond] = cond_miss
        opt_margins_dict[cond] = opt_field_map[cond]

    # 4. Classification statistics
    classification_stats = {
        "n_type_i": robust.n_type_i,
        "n_type_ii": robust.n_type_ii,
        "n_type_iii": robust.n_type_iii,
        "n_type_iv": robust.n_type_iv,
        "max_imag_eigenvalue": robust.max_imag_eigenvalue,
    }

    return ComparisonResult(
        eulerian_margins=eulerian_margins,
        robust_margins=robust_margins,
        missed=missed,
        severity=severity,
        pct_missed=pct_missed,
        pct_violated_robust=pct_violated_robust,
        conditional_miss_rate=conditional_miss_rate,
        classification_stats=classification_stats,
        opt_margins=opt_margins_dict,
        he_types=robust.he_types.reshape(grid_shape),
    )


def build_comparison_table(
    results_dir: str,
    metrics: list[str],
    v_s_values: list[float],
) -> list[dict]:
    """Build a comparison table from cached .npz result files.

    Loads cached analysis results and assembles a per-metric-per-velocity
    comparison table suitable for the paper.

    Parameters
    ----------
    results_dir : str
        Directory containing ``{metric}_vs{v_s}.npz`` files.
    metrics : list[str]
        List of metric names (e.g. ``["alcubierre", "lentz", ...]``).
    v_s_values : list[float]
        Warp velocities to include.

    Returns
    -------
    list[dict]
        Rows of the comparison table. Also saved to
        ``{results_dir}/comparison_table.json``.
    """
    rows: list[dict] = []
    results_path = Path(results_dir)

    for v_s in v_s_values:
        for name in metrics:
            cache_path = results_path / f"{name}_vs{v_s}.npz"
            if not cache_path.exists():
                continue

            data = np.load(str(cache_path))
            row: dict = {
                "metric": name,
                "v_s": v_s,
            }

            for cond in _CONDITIONS:
                eul_key = f"{cond}_eulerian"
                rob_key = f"{cond}_robust"
                if eul_key not in data or rob_key not in data:
                    continue

                eul = data[eul_key]
                rob = data[rob_key]
                missed_mask = (eul >= 0) & (rob < -1e-10)

                n_violated = int(np.sum(rob < -1e-10))
                n_missed = int(np.sum(missed_mask))

                row[f"{cond}_eulerian_min"] = float(np.nanmin(eul))
                row[f"{cond}_robust_min"] = float(np.nanmin(rob))
                row[f"{cond}_pct_violated_robust"] = float(
                    n_violated / rob.size * 100
                )
                row[f"{cond}_pct_missed"] = float(
                    n_missed / missed_mask.size * 100
                )
                # Conditional miss rate: f_miss|viol
                row[f"{cond}_conditional_miss_rate"] = (
                    float(n_missed / n_violated * 100) if n_violated > 0 else 0.0
                )

            rows.append(row)

    output_path = results_path / "comparison_table.json"
    os.makedirs(str(results_path), exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(rows, f, indent=2)

    return rows
