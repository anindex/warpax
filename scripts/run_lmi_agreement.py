"""Compare routed energy-condition margins with direct numerical LMI tests.

Reports Type-I sign agreement, contradictions between Type-III/IV labels
and LMI verdicts, and wall coverage. Both routes use finite precision;
a disagreement does not by itself identify which route failed.

The normalized NEC deficit is twice the LMI margin, so comparisons use
its doubled noise floor. Values within either route's floor are inconclusive.
Grid fractions and tolerance studies are numerical diagnostics.

Outputs: results/lmi_agreement.json and the manuscript's lmi_typefree.tex table.
"""

from __future__ import annotations

import argparse
import os

import jax
from _benchmark_grid import benchmark_grid
from _json_io import dump_json
from _json_io import write_table as write_tex_table
from _paper_metrics import METRIC_ORDER, METRICS

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpax.energy_conditions.filtering import shape_function_mask
from warpax.energy_conditions.frame_free import certify_grid_frame_free
from warpax.energy_conditions.slemma import certify_point as certify_point_lmi
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

CONDITIONS = ("nec", "wec", "sec", "dec")
F_LOW, F_HIGH = 0.1, 0.9


def _verdict(margin, floor):
    """-1 violated, +1 satisfied, 0 inconclusive. The floor is what makes it three-valued."""
    return np.where(margin > floor, 1, np.where(margin < -floor, -1, 0))


def compare_one(name, v_s, N, batch_size=2048):
    cls, extra = METRICS[name]
    metric = cls(v_s=v_s, R=1.0, sigma=8.0, **extra)
    grid = benchmark_grid(metric, N)
    curv = evaluate_curvature_grid(metric, grid, batch_size=batch_size)
    ff = certify_grid_frame_free(
        curv.stress_energy, curv.metric, curv.metric_inv, solver="standard"
    )

    flat_T = jnp.reshape(curv.stress_energy, (-1, 4, 4))
    flat_g = jnp.reshape(curv.metric, (-1, 4, 4))
    lmi = jax.vmap(certify_point_lmi)(flat_T, flat_g)

    he = np.asarray(ff.he_types).reshape(-1)
    floor = np.asarray(ff.nec_noise_floor).reshape(-1) / 2  # Underlying LMI scale.
    vac = np.asarray(ff.is_vacuum).reshape(-1) > 0.5
    coords = build_coord_batch(grid, t=0.0)
    wall = np.asarray(
        jnp.reshape(
            shape_function_mask(metric, coords, (N, N, N), f_low=F_LOW, f_high=F_HIGH),
            (-1,),
        )
    ).astype(bool)

    routed = {c: np.asarray(getattr(ff, f"{c}_margins")).reshape(-1) for c in CONDITIONS}

    out = {
        "metric": name,
        "v_s": v_s,
        "N": N,
        "n_points": int(he.size),
        "n_wall": int(wall.sum()),
        "n_vacuum": int(vac.sum()),
        "type_counts": {str(t): int((he == t).sum()) for t in (1, 2, 3, 4)},
    }

    # Compare only points decisive under both numerical routes.
    isI = (he == 1) & ~vac
    for c in CONDITIONS:
        routed_floor = 2 * floor if c == "nec" else floor
        a = _verdict(routed[c][isI], routed_floor[isI])
        b = _verdict(np.asarray(lmi[c])[isI], floor[isI])
        both = (a != 0) & (b != 0)
        out[f"typeI_{c}_n_decisive"] = int(both.sum())
        out[f"typeI_{c}_agree"] = float((a[both] == b[both]).mean()) if both.any() else None

    # A positive LMI verdict contradicts an exact Type-III/IV label.
    # Count numerical contradictions; neither route is independently certified.
    bad_label = (he >= 3) & ~vac
    lmi_says_ok = np.zeros_like(bad_label)
    for c in CONDITIONS:
        lmi_says_ok |= _verdict(np.asarray(lmi[c]), floor) > 0
    nec_says_ok = _verdict(np.asarray(lmi["nec"]), floor) > 0
    out["n_type_iii_iv"] = int(bad_label.sum())
    out["n_detected_label_contradictions"] = int((bad_label & lmi_says_ok).sum())
    out["detected_label_contradiction_rate"] = (
        float((bad_label & lmi_says_ok).sum() / bad_label.sum()) if bad_label.any() else None
    )
    # The NEC-only count is quoted separately in the appendix text, so it is recorded
    # rather than recomputed from the disjunction.
    out["n_detected_label_contradictions_nec"] = int((bad_label & nec_says_ok).sum())

    # (c) Coverage. The eigenvalue route has nothing to say outside Type I; the LMI
    #     decides there too. This is the fraction of the wall it recovers.
    outside = wall & (he != 1) & ~vac
    decided = outside & (_verdict(np.asarray(lmi["nec"]), floor) != 0)
    out["n_wall_non_type_i"] = int(outside.sum())
    out["n_wall_non_type_i_decided_by_lmi"] = int(decided.sum())
    return out


def write_lmi_table(rows, out_path, table_vels=(0.5, 1.0, 2.0)):
    def get(name, v):
        for r in rows:
            if r["metric"] == name and abs(r["v_s"] - v) < 1e-9:
                return r
        return None

    lines = [
        r"\begin{tabular}{@{}l ccc ccc@{}}",
        r"  \toprule",
        r"  & \multicolumn{3}{c}{Type-I verdicts agreeing (\%)} "
        r"& \multicolumn{3}{c}{Numerically detected label contradictions (\%)} \\",
        r"  \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"  Metric & $v_s=0.5$ & $1.0$ & $2.0$ & $v_s=0.5$ & $1.0$ & $2.0$ \\",
        r"  \midrule",
    ]
    for name in METRIC_ORDER:
        agree, err = [], []
        for v in table_vels:
            r = get(name, v)
            a = None if r is None else r.get("typeI_nec_agree")
            e = None if r is None else r.get("detected_label_contradiction_rate")
            agree.append("--" if a is None else f"{100.0 * a:.2f}")
            err.append("--" if e is None else f"{100.0 * e:.2f}")
        lines.append(f"  {name} & " + " & ".join(agree + err) + r" \\")
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_tex_table(
        out_path, lines, script="scripts/run_lmi_agreement.py", sources="results/lmi_agreement.json"
    )
    print(f"  Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--velocities", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--N", type=int, default=48)
    p.add_argument("--metrics", type=str, nargs="+", default=METRIC_ORDER)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.N, args.velocities, args.metrics = 20, [0.5], ["Alcubierre", "Rodal"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print(f"LMI COMPARISON (R=1, sigma=8, N={args.N}, wall-clustered)")
    print("=" * 70)
    rows = []
    for name in args.metrics:
        for v_s in args.velocities:
            r = compare_one(name, v_s, args.N)
            rows.append(r)
            agree = r["typeI_nec_agree"]
            rate = r["detected_label_contradiction_rate"]
            print(
                f"  {len(rows):2d}/{len(args.metrics) * len(args.velocities)} "
                f"{name:>15s} v_s={v_s:.2f}  "
                f"TypeI agree={'n/a' if agree is None else f'{100 * agree:6.2f}%'}  "
                f"III/IV={r['n_type_iii_iv']:7d}  "
                f"numerically detected label contradictions="
                f"{'n/a' if rate is None else f'{100 * rate:.3f}%'}",
                flush=True,
            )

    name = "lmi_agreement_smoke.json" if args.smoke else "lmi_agreement.json"
    dump_json({"config": vars(args), "rows": rows}, os.path.join(RESULTS_DIR, name))
    print(f"\nWrote {os.path.join(RESULTS_DIR, name)}")
    if not args.smoke:
        write_lmi_table(rows, os.path.join(TABLES_DIR, "lmi_typefree.tex"))


if __name__ == "__main__":
    main()
