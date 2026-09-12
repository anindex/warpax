"""Kinematically standardized construction diagnostics and reference parameter sets.

Each construction has an independent three-grid ladder. The JSON records all
parameters, units and provenance. Garattini-Zatrimaylov's matched-motion metric
has exact j=0 and Type I; its fractions use that identity. Numerical labels are
retained separately for debugging. No matter-content or stress-severity ordering
is inferred from this panel.
"""

from __future__ import annotations

import argparse
import os

import jax
from _json_io import dump_json
from _json_io import write_table as write_tex_table

jax.config.update("jax_enable_x64", True)

import numpy as np

from warpax.analysis.construction_adapter import (
    MATCHED_R_C,
    MATCHED_SIGMA,
    MATCHED_V_S,
    MATCHED_WIDTH,
    MATCHING_CAVEAT,
    MIN_WALL_CELLS,
    construction_registry,
    matched_registry,
)
from warpax.analysis.invariant_verification import single_frame_miss
from warpax.energy_conditions.frame_free import certify_grid_frame_free
from warpax.energy_conditions.verifier import _eulerian_ec_point
from warpax.geometry import evaluate_curvature_points
from warpax.grids import axisymmetric_grid, proper_volume_weights, wall_cells_on_axis

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

ORDER = ["Alcubierre", "Rodal", "Fuchs", "Garattini"]
LADDER = ((32, 32), (48, 48), (64, 64))
F_LOW, F_HIGH = 0.1, 0.9


def verify_one(spec, n_r: int, n_mu: int) -> dict:
    metric = spec.metric()
    center = spec.center_of(metric)
    grid = axisymmetric_grid(
        spec.r_max,
        n_r,
        n_mu,
        wall_radius=spec.wall_radius,
        a=spec.cluster_a,
        center=center,
    )
    # Measure the resolution witness on the axis the grid samples, the shifted one for
    # an off-origin bubble: on grid.r alone the Garattini wall reads 1.5 cells against
    # 4.5 through the bubble centre. grid.r may start at 0, which the mirror duplicates.
    r_pos = grid.r[grid.r > 0.0]
    axis = center + np.concatenate([-r_pos[::-1], grid.r])
    res = wall_cells_on_axis(metric, axis)
    row = {
        "metric": spec.name,
        "n_r": n_r,
        "n_mu": n_mu,
        "speed": spec.default_speed,
        "speed_param": spec.speed_param,
        "wall_radius": spec.wall_radius,
        "r_max": spec.r_max,
        "grid_center": center,
        "wall_cells": res.cells,
        "wall_width": res.width,
        "resolved": bool(res.cells >= MIN_WALL_CELLS),
        "params": spec.params,
        "claim": spec.claim,
    }
    if not row["resolved"]:
        row["note"] = f"wall spans {res.cells:.2f} cells (< {MIN_WALL_CELLS})"
        return row

    curv = evaluate_curvature_points(metric, grid.coords, batch_size=256)
    T, g, gi = curv.stress_energy, curv.metric, curv.metric_inv

    f = np.asarray(jax.vmap(metric.shape_function_value)(grid.coords))
    wall = (f >= F_LOW) & (f <= F_HIGH)
    w = np.asarray(proper_volume_weights(grid.weights, g))

    ff = certify_grid_frame_free(T, g, gi, solver="auto")
    he = np.asarray(ff.he_types).ravel()
    sel = wall & np.isfinite(he)
    w_wall = w[sel].sum()

    nec = np.asarray(ff.nec_margins).ravel()
    if spec.name == "Garattini":
        # The matched-motion shift is a gradient. Use its exact Type-I reduction
        # through the Eulerian normal, avoiding defective eigensolver labels.
        normal = -np.asarray(gi)[:, :, 0]
        rho = np.einsum("ni,nij,nj->n", normal, np.asarray(T), normal)
        pressures = np.linalg.eigvalsh(np.asarray(T)[:, 1:, 1:])
        nec = rho + pressures[:, 0]
        row["numerical_type_labels"] = {
            str(kind): float(w[sel & (he == kind)].sum() / w_wall) for kind in (1, 2, 3, 4)
        }
        he = np.ones_like(he)
        sel = wall & np.isfinite(rho) & np.all(np.isfinite(pressures), axis=1)
        w_wall = w[sel].sum()
        row["type_basis"] = (
            "exact j=0; Eulerian normal timelike eigenvector; symmetric spatial stress"
        )
        assert float(metric.H) > 0
        assert np.isclose(spec.params["r_0"] * float(metric.H), spec.default_speed)
        row["v_equals_H_r0"] = spec.params["r_0"] * float(metric.H)
        eulerian = jax.vmap(_eulerian_ec_point)(T, g, gi)
        margins = {
            "nec": nec,
            "wec": np.minimum(rho, nec),
            "dec": rho - np.max(np.abs(pressures), axis=1),
        }
        miss = {}
        for cond, margin in margins.items():
            violated = sel & (margin < -1e-10)
            missed = violated & (np.asarray(eulerian[cond]) >= 0.0)
            denominator = float(w[violated].sum())
            miss[cond] = {
                "miss_rate": float(w[missed].sum()) / denominator if denominator else None,
                "n_violated": int(violated.sum()),
            }
    else:
        miss = single_frame_miss(T, g, gi, mask=wall, volume_weights=w)

    typeI_wall = sel & (he == 1.0) & np.isfinite(nec)
    nec_min = float(np.min(nec[typeI_wall])) if typeI_wall.any() else float("nan")

    # The ADM normal is unit timelike at every speed.
    eulerian_valid = True

    row.update(
        {
            "n_wall_points": int(sel.sum()),
            "frac_type_i": float(w[sel & (he == 1.0)].sum() / w_wall) if w_wall else float("nan"),
            "frac_type_iv": float(w[sel & (he == 4.0)].sum() / w_wall) if w_wall else float("nan"),
            "frac_type_ii": float(w[sel & (he == 2.0)].sum() / w_wall) if w_wall else float("nan"),
            "invariant_nec_min": nec_min,
            # Curvature carries 1/L^2, so this is the comparable quantity.
            "nec_min_dimensionless": nec_min * spec.wall_radius**2,
            "eulerian_valid": eulerian_valid,
        }
    )
    # miss_rate is None when nothing violates at all (empty denominator), which
    # is a meaningful outcome here, not an error: it is what a construction that
    # clears the all-observer check on its wall looks like.
    for cond in ("wec", "nec", "dec"):
        rate = miss[cond]["miss_rate"] if miss else None
        row[f"miss_{cond}_pct"] = 100.0 * rate if rate is not None else None
        row[f"n_violated_{cond}"] = miss[cond]["n_violated"] if miss else None
    return row


def _fmt(x, nd=1):
    return f"{x:.{nd}f}" if (x is not None and np.isfinite(x)) else "--"


def _fmt_margin(x):
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.1e}" if abs(x) < 5e-3 else f"{x:.3f}"


def write_table(rows_by_metric: dict, out_path: str, *, show_speed: bool) -> None:
    speed_col = " c" if show_speed else ""
    lines = [
        r"\begin{tabular}{@{}l" + speed_col + r" cc cc c cc@{}}",
        r"  \toprule",
        (
            r"  & $v_s$ & \multicolumn{2}{c}{Wall cells}"
            if show_speed
            else r"  & \multicolumn{2}{c}{Wall cells}"
        )
        + r" & Type~I & Type~IV & $R_c^2\min(\rho+p_i)$ & WEC & NEC \\",
        (r"  Metric & & coarse & fine" if show_speed else r"  Metric & coarse & fine")
        + r" & (\%) & (\%) & (Type~I) & \multicolumn{2}{c}{miss (\%)} \\",
        r"  \midrule",
    ]
    for name in ORDER:
        rows = rows_by_metric.get(name)
        if not rows:
            continue
        fine, coarse = rows[-1], rows[0]
        lead = f"  {name}"
        if show_speed:
            lead += f" & {fine['speed']:g}"
        if not fine.get("resolved", False):
            lines.append(
                lead + f" & {_fmt(coarse['wall_cells'])} & {_fmt(fine['wall_cells'])}"
                r" & \multicolumn{5}{c}{\emph{wall unresolved}} \\"
            )
            continue
        lines.append(
            lead + f" & {_fmt(coarse['wall_cells'])} & {_fmt(fine['wall_cells'])}"
            f" & {_fmt(fine['frac_type_i'] * 100)} & {_fmt(fine['frac_type_iv'] * 100)}"
            f" & {_fmt_margin(fine['nec_min_dimensionless'])}"
            f" & {_fmt(fine.get('miss_wec_pct'))} & {_fmt(fine.get('miss_nec_pct'))} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_tex_table(
        out_path,
        lines,
        script="scripts/run_construction_verification.py",
        sources="results/construction_verification.json",
    )
    print(f"  Wrote {out_path}")


def run_mode(registry, ladder, label: str) -> dict:
    print(f"\n--- {label} ---")
    out: dict = {}
    for name in ORDER:
        spec = registry[name]
        rows = []
        for n_r, n_mu in ladder:
            rows.append(verify_one(spec, n_r, n_mu))
        out[name] = rows
        fine = rows[-1]
        if fine.get("resolved", False):
            spread_i = (
                abs(rows[-1]["frac_type_i"] - rows[-2]["frac_type_i"]) * 100
                if len(rows) > 1
                else float("nan")
            )
            print(
                f"  {name:>11s}  cells {rows[0]['wall_cells']:5.2f}->{fine['wall_cells']:5.2f}"
                f"  TypeI={fine['frac_type_i'] * 100:6.2f}%  TypeIV={fine['frac_type_iv'] * 100:6.2f}%"
                f"  Rc^2 minNEC={fine['nec_min_dimensionless']:+.4g}"
                f"  missW/N={_fmt(fine.get('miss_wec_pct'))}/{_fmt(fine.get('miss_nec_pct'))}"
                f"  [finest-two TypeI spread {spread_i:.2f} pp]"
            )
        else:
            print(f"  {name:>11s}  {fine.get('note')}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["matched", "native", "both"], default="both")
    p.add_argument("--smoke", action="store_true", help="coarsest level only")
    args = p.parse_args()

    ladder = LADDER[:1] if args.smoke else LADDER

    print("=" * 78)
    print("CROSS-CONSTRUCTION ALL-OBSERVER VERIFICATION")
    print("=" * 78)
    print(
        f"  matched point: v_s={MATCHED_V_S}  R_c={MATCHED_R_C}  "
        f"sigma={MATCHED_SIGMA:.9f}  W={MATCHED_WIDTH}"
    )
    print(f"  ladder: {list(ladder)}   wall band f in [{F_LOW}, {F_HIGH}]")
    print(f"  NOT matched: {MATCHING_CAVEAT}")

    payload = {
        "matched_point": {
            "v_s": MATCHED_V_S,
            "R_c": MATCHED_R_C,
            "sigma": MATCHED_SIGMA,
            "wall_width": MATCHED_WIDTH,
            "width_over_Rc": MATCHED_WIDTH / MATCHED_R_C,
        },
        "matching_caveat": MATCHING_CAVEAT,
        "ladder": [list(x) for x in ladder],
        "wall_band": [F_LOW, F_HIGH],
        "order": ORDER,
    }

    if args.mode in ("matched", "both"):
        rows = run_mode(matched_registry(), ladder, "MATCHED shift kinematics")
        payload["matched"] = rows
        if not args.smoke:
            write_table(
                rows, os.path.join(TABLES_DIR, "construction_matched.tex"), show_speed=False
            )
    if args.mode in ("native", "both"):
        rows = run_mode(construction_registry(), ladder, "REFERENCE parameter choices")
        payload["native"] = rows
        if not args.smoke:
            write_table(rows, os.path.join(TABLES_DIR, "construction_native.tex"), show_speed=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "construction_verification.json")
    dump_json(payload, out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
