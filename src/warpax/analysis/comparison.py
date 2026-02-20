"""Eulerian vs robust energy condition comparison logic.

Compares Eulerian-frame EC margins against observer-robust (optimized)
margins at every grid point, identifying where Eulerian analysis fails
to detect violations that non-Eulerian observers can see.

The key insight from Santiago-Schuster-Visser (2022): energy conditions
are observer-dependent, so checking only the Eulerian frame is
insufficient.  This module computes the per-point "missed" flag and
severity ratio that quantifies how much worse the optimized observer
sees compared to the Eulerian one.
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

    All dict values are keyed by condition name: "nec", "wec", "sec", "dec".

    Attributes
    ----------
    eulerian_margins : dict[str, Array]
        Eulerian-frame margins with shape ``(*grid_shape,)``.
    robust_margins : dict[str, Array]
        Observer-robust (optimized) margins with shape ``(*grid_shape,)``.
    missed : dict[str, Array]
        Boolean mask: True where Eulerian says satisfied but robust says
        violated.  ``(eul >= 0) & (rob < -1e-10)``.
    severity : dict[str, Array]
        Severity ratio: ``eul_margin - rob_margin`` at missed points,
        zero elsewhere.
    pct_missed : dict[str, float]
        Percentage of grid points missed by Eulerian analysis.
    pct_violated_robust : dict[str, float]
        Percentage of grid points violated under robust analysis.
    conditional_miss_rate : dict[str, float]
        Conditional miss rate: missed / violated per condition
        (f_miss|viol = pct_missed / pct_violated_robust when violated > 0).
    classification_stats : dict[str, int | float]
        Classification breakdown: n_type_i..n_type_iv, max_imag_eigenvalue.
    opt_margins : dict[str, Float[Array, "..."]]
        Raw optimizer margins (before merge with algebraic).
    he_types : Array
        Per-point Hawking-Ellis type (1-4) with shape ``(*grid_shape,)``.
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

    CRITICAL: Eulerian and robust results are computed SEPARATELY.
    Do NOT use ``compute_eulerian=True`` in ``verify_grid`` that merges
    results, destroying the comparison (Eulerian and robust must be
    computed independently).

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

    n_points = int(jnp.prod(jnp.array(grid_shape)))
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
        Rows of the comparison table.  Also saved to
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

    # Save to JSON
    output_path = results_path / "comparison_table.json"
    os.makedirs(str(results_path), exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(rows, f, indent=2)

    return rows
