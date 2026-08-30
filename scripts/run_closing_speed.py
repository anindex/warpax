"""The Type-IV closing speed: one speed-independent field bounds the whole sweep.

For a flat-slice unit-lapse drive the Eulerian decomposition scales exactly:
``rho_n, S_par ~ v_s^2`` and ``|j| ~ v_s``.
Writing ``rho_n + S_par = a v_s^2`` and ``|j| = d v_s``, the momentum
discriminant is

    Delta = v_s^2 (a^2 v_s^2 - 4 d^2),

so ``a`` and ``d`` do not depend on the speed and each wall point carries an
exact closing speed

    v_*(x) = 2 d(x) / |a(x)|,

Type IV in the momentum channel below it and Type I above. Computed once, this
field fixes a curve for every speed at once:

    F_mom(v_s) = vol{ v_* > v_s } / vol(wall).

Delta < 0 is sufficient but not necessary for Type IV, so F_mom is a LOWER BOUND
on the wall Type-IV fraction, and the gap to the measured fraction is the share
carried by the transverse (conformal) channel, which Delta does not govern. That
gap is what this script measures; it is not otherwise quantified anywhere.

The premise is the exact v_s scaling, which needs flat slices. Van den Broeck has
gamma_ij = B^2 delta_ij and fails it, its ``a`` moves by O(1) between speeds
while the flat-slice drives hold theirs to 1e-15, so it is reported and
excluded, by the same hypothesis that excludes it from the integrated E_- law.

Outputs
-------
- results/closing_speed.json
- ../warpax_arxiv/tables/closing_speed.tex
"""

from __future__ import annotations

import argparse
import json
import os

import jax
from _benchmark_grid import benchmark_grid
from _json_io import dump_json
from _json_io import write_table as write_tex_table
from _paper_metrics import METRIC_ORDER, instantiate

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpax.energy_conditions.filtering import shape_function_mask
from warpax.energy_conditions.frame_free import eulerian_momentum_frame
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.grids import proper_volume_weights

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

F_LOW, F_HIGH = 0.1, 0.9
# a = 0 means the pair never closes; guarded relative to the tensor scale.
A_FLOOR_REL = 1e-12


def coefficients(name, v_s, N):
    """Wall values of (a, d, weight), with a, d stripped of their v_s scaling."""
    metric = instantiate(name, v_s)
    grid = benchmark_grid(metric, N)
    curv = evaluate_curvature_grid(metric, grid, batch_size=2048)
    coords = build_coord_batch(grid, t=0.0)
    mask = np.asarray(
        jnp.reshape(
            shape_function_mask(metric, coords, (N, N, N), f_low=F_LOW, f_high=F_HIGH),
            (-1,),
        )
    ).astype(bool)

    T = jnp.reshape(curv.stress_energy, (-1, 4, 4))
    g = jnp.reshape(curv.metric, (-1, 4, 4))
    gi = jnp.reshape(curv.metric_inv, (-1, 4, 4))
    rho, S_par, jmag = jax.vmap(eulerian_momentum_frame)(T, g, gi)

    w = proper_volume_weights(grid.volume_weights_array, curv.metric)
    w = np.asarray(jnp.reshape(w, (-1,)))

    a = np.asarray(rho + S_par)[mask] / v_s**2
    d = np.asarray(jmag)[mask] / v_s
    return a, d, w[mask]


def closing_speeds(a, d):
    """v_* = 2 d / |a|, with +inf where a is below the scale floor."""
    scale = max(float(np.max(np.abs(a))), 1e-300)
    open_forever = np.abs(a) <= A_FLOOR_REL * scale
    v_star = np.where(open_forever, np.inf, 2.0 * d / np.maximum(np.abs(a), 1e-300))
    # d = 0 (irrotational) closes at zero speed: never Type IV in this channel.
    return np.where(d <= 0.0, 0.0, v_star)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--reference-v", type=float, default=1.0)
    # Second speed, used only to check that a and d really are speed-independent.
    p.add_argument("--check-v", type=float, default=0.5)
    p.add_argument("--metrics", nargs="+", default=METRIC_ORDER)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.N, args.metrics = 24, ["Alcubierre", "Rodal"]

    with open(os.path.join(RESULTS_DIR, "velocity_sweep.json")) as fh:
        sweep = json.load(fh)
    measured = {(r["metric"], r["v_s"]): r["wall_frac_type_iv"] for r in sweep["rows"]}
    speeds = sorted({v for (_, v) in measured})

    rows = []
    print("=" * 74)
    print(f"CLOSING SPEED v_* = 2d/|a|  (N={args.N}, reference v_s={args.reference_v})")
    print("=" * 74)
    for name in args.metrics:
        a, d, w = coefficients(name, args.reference_v, args.N)
        a2, d2, _ = coefficients(name, args.check_v, args.N)
        # Speed-independence of the coefficients is the whole premise; measure it.
        # S_par = S(jhat, jhat) needs a momentum direction, so `a` is only defined
        # where d > 0. On an irrotational drive that set is empty and the momentum
        # channel is identically closed, which is not a premise failure.
        scale_a = max(float(np.max(np.abs(a))), 1e-300)
        scale_d = max(float(np.max(np.abs(d))), 1e-300)
        # Against the tensor scale, not max(d): on an irrotational drive max(d)
        # is itself roundoff, and a relative floor would select the noise.
        active = d > 1e-9 * scale_a * args.reference_v
        a_dev = float(np.max(np.abs(a - a2)[active])) / scale_a if active.any() else 0.0
        d_dev = float(np.max(np.abs(d - d2))) / scale_d

        v_star = closing_speeds(a, d)
        wt = float(np.sum(w))
        pred = {v: float(np.sum(w * (v_star > v)) / wt) for v in speeds}
        # Signed, not absolute: the claim is a bound, and its direction is the
        # content. gap = measured - predicted = the transverse channel's share.
        gaps = [measured[(name, v)] - pred[v] for v in speeds if (name, v) in measured]
        # Flat-slice drives must satisfy the bound; a negative gap there would
        # falsify it. Van den Broeck is exempt and is flagged, not excluded.
        flat_slice = a_dev < 1e-9
        rows.append(
            {
                "metric": name,
                "reference_v_s": args.reference_v,
                "check_v_s": args.check_v,
                "N": args.N,
                "n_wall": int(a.size),
                "coeff_rel_dev_a": a_dev,
                "n_wall_with_momentum": int(active.sum()),
                "coeff_rel_dev_d": d_dev,
                "v_star_median": float(np.median(v_star[np.isfinite(v_star)]))
                if np.any(np.isfinite(v_star))
                else None,
                "frac_never_closing": float(np.sum(w * ~np.isfinite(v_star)) / wt),
                "speeds": speeds,
                "predicted_frac_type_iv": [pred[v] for v in speeds],
                "measured_frac_type_iv": [measured.get((name, v)) for v in speeds],
                "flat_slice_premise_holds": bool(flat_slice),
                "transverse_gap_pp": [100.0 * g for g in gaps],
                "min_gap_pp": 100.0 * min(gaps) if gaps else None,
                "max_gap_pp": 100.0 * max(gaps) if gaps else None,
            }
        )
        if flat_slice and gaps and min(gaps) < -1e-9:
            raise RuntimeError(
                f"{name}: momentum-channel fraction exceeds the measured Type-IV "
                f"fraction by {-100.0 * min(gaps):.2f} pp; the bound is claimed "
                f"for flat-slice drives and this would falsify it"
            )
        print(
            f"  {name:>15s}  a,d speed-dev {a_dev:.1e}, {d_dev:.1e}  "
            f"{'flat-slice' if flat_slice else 'PREMISE FAILS'}  "
            f"transverse gap {100.0 * min(gaps):+5.1f} to "
            f"{100.0 * max(gaps):+5.1f} pp",
            flush=True,
        )

    name_out = "closing_speed_smoke.json" if args.smoke else "closing_speed.json"
    dump_json({"config": vars(args), "rows": rows}, os.path.join(RESULTS_DIR, name_out))
    print(f"\nWrote {os.path.join(RESULTS_DIR, name_out)}")
    if not args.smoke:
        write_table(rows, os.path.join(TABLES_DIR, "closing_speed.tex"))


def write_table(rows, out_path, table_vels=(0.1, 0.5, 1.0, 2.5)):
    lines = [
        r"\begin{tabular}{@{}l cc cc cc cc cc@{}}",
        r"  \toprule",
        r"  & & & \multicolumn{2}{c}{$v_s=0.1$} & \multicolumn{2}{c}{$0.5$}"
        r" & \multicolumn{2}{c}{$1.0$} & \multicolumn{2}{c}{$2.5$} \\",
        r"  \cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
        r"  Metric & $\delta a$ & $\tilde v_\star$"
        r" & $F_{\rm mom}$ & gap & $F_{\rm mom}$ & gap"
        r" & $F_{\rm mom}$ & gap & $F_{\rm mom}$ & gap \\",
        r"  \midrule",
    ]
    for r in rows:
        cells = []
        for v in table_vels:
            try:
                i = r["speeds"].index(v)
            except ValueError:
                cells += ["--", "--"]
                continue
            p = r["predicted_frac_type_iv"][i]
            g = r["transverse_gap_pp"][i]
            cells.append("--" if p is None else f"{100.0 * p:.1f}")
            cells.append("--" if g is None else f"{g:+.1f}")
        med = r["v_star_median"]
        lines.append(
            f"  {r['metric']} & "
            + f"{r['coeff_rel_dev_a']:.0e}"
            + " & "
            + ("--" if med is None else f"{med:.2f}")
            + " & "
            + " & ".join(cells)
            + r" \\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_tex_table(
        out_path, lines, script="scripts/run_closing_speed.py", sources="results/closing_speed.json"
    )
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
