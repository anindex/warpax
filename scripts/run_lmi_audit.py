"""Audit the type-based certification against the type-free LMI, at every grid point.

Type-IV identification is tolerance dependent, and a
finite grid missing the Type-II/III loci is not a continuum proof of their absence. Both
are true of the *classification*. Neither is true of Theorem 1: the linear matrix
inequality forms no eigendecomposition, consults no tolerance and never asks what the
algebraic type is, so it decides every point of every grid on the same footing.

That makes the LMI an auditor rather than a competitor. The certification routing is
unchanged -- Type-I points are still decided by the eigenvalue inequalities, which are
exact there -- and this script simply runs the LMI everywhere as well and reports how
often the two agree. What the agreement rate measures is the classification, not the
energy conditions:

  * At a Type-I point the two are provably equivalent, so a disagreement is a *label*
    error: the point was called Type I and is not, or the eigen-decomposition of a
    near-defective tensor moved the eigenvalues.
  * At a Type-III or Type-IV point every standard energy condition fails (Martin-Moruno
    & Visser 2017). An LMI margin that *certifies satisfaction* there is therefore a
    certified misclassification, and this script counts those separately: they are the
    honest error rate of the float64 classifier, reported rather than assumed away.

Both comparisons are made against ``nec_noise_floor`` and not against zero. The LMI
margin contract is one-sided, so a value inside the floor is inconclusive rather than a
verdict, and thresholding at zero would report the float64 residual as classifier error.

Outputs
-------
- results/lmi_audit.json
- ../warpax_arxiv/tables/lmi_typefree.tex
"""
from __future__ import annotations

import argparse
import os

from _benchmark_grid import benchmark_grid
from _json_io import dump_json, write_table as write_tex_table

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions.filtering import shape_function_mask
from warpax.energy_conditions.frame_free import certify_grid_frame_free
from warpax.energy_conditions.slemma import certify_point as certify_point_lmi
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

METRICS = {
    "Alcubierre": (AlcubierreMetric, {}),
    "Natário": (NatarioMetric, {}),
    "Van den Broeck": (VanDenBroeckMetric,
                       {"R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0}),
    "Rodal": (RodalMetric, {}),
}
METRIC_ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]
CONDITIONS = ("nec", "wec", "sec", "dec")
F_LOW, F_HIGH = 0.1, 0.9


def _verdict(margin, floor):
    """-1 violated, +1 satisfied, 0 inconclusive. The floor is what makes it three-valued."""
    return np.where(margin > floor, 1, np.where(margin < -floor, -1, 0))


def audit_one(name, v_s, N, batch_size=2048):
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
    floor = np.asarray(ff.nec_noise_floor).reshape(-1)
    vac = np.asarray(ff.is_vacuum).reshape(-1) > 0.5
    coords = build_coord_batch(grid, t=0.0)
    wall = np.asarray(
        jnp.reshape(
            shape_function_mask(metric, coords, (N, N, N), f_low=F_LOW, f_high=F_HIGH),
            (-1,),
        )
    ).astype(bool)

    routed = {c: np.asarray(getattr(ff, f"{c}_margins")).reshape(-1) for c in CONDITIONS}

    out = {"metric": name, "v_s": v_s, "N": N, "n_points": int(he.size),
           "n_wall": int(wall.sum()), "n_vacuum": int(vac.sum()),
           "type_counts": {str(t): int((he == t).sum()) for t in (1, 2, 3, 4)}}

    # (a) Type-I agreement. Both routes are exact there, so any disagreement is a label
    #     error rather than a disagreement about physics.
    isI = (he == 1) & ~vac
    for c in CONDITIONS:
        a = _verdict(routed[c][isI], floor[isI])
        b = _verdict(np.asarray(lmi[c])[isI], floor[isI])
        both = (a != 0) & (b != 0)
        out[f"typeI_{c}_n_decisive"] = int(both.sum())
        out[f"typeI_{c}_agree"] = (
            float((a[both] == b[both]).mean()) if both.any() else None
        )

    # (b) Certified misclassification. Types III and IV violate every standard energy
    #     condition, so an LMI margin that certifies satisfaction there cannot be
    #     reconciled with the label. Counting them is the classifier's measured error
    #     rate -- the quantity that would otherwise be unquantified.
    #     Types III and IV violate *every* condition, so certifying *any one* of them is
    #     already a contradiction: the predicate is a disjunction, not a conjunction. An
    #     earlier version initialised True and ANDed, which counted a point only when all
    #     four were certified satisfied -- strictly weaker than the claim it reports, and
    #     an undercount by construction.
    bad_label = (he >= 3) & ~vac
    lmi_says_ok = np.zeros_like(bad_label)
    for c in CONDITIONS:
        lmi_says_ok |= _verdict(np.asarray(lmi[c]), floor) > 0
    nec_says_ok = _verdict(np.asarray(lmi["nec"]), floor) > 0
    out["n_type_iii_iv"] = int(bad_label.sum())
    out["n_certified_misclassified"] = int((bad_label & lmi_says_ok).sum())
    out["certified_misclassification_rate"] = (
        float((bad_label & lmi_says_ok).sum() / bad_label.sum())
        if bad_label.any() else None
    )
    # The NEC-only count is quoted separately in the appendix text, so it is recorded
    # rather than recomputed from the disjunction.
    out["n_certified_misclassified_nec"] = int((bad_label & nec_says_ok).sum())

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
        r"& \multicolumn{3}{c}{Certified label errors (\%)} \\",
        r"  \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"  Metric & $v_s=0.5$ & $1.0$ & $2.0$ & $v_s=0.5$ & $1.0$ & $2.0$ \\",
        r"  \midrule",
    ]
    for name in METRIC_ORDER:
        agree, err = [], []
        for v in table_vels:
            r = get(name, v)
            a = None if r is None else r.get("typeI_nec_agree")
            e = None if r is None else r.get("certified_misclassification_rate")
            agree.append("--" if a is None else f"{100.0 * a:.2f}")
            err.append("--" if e is None else f"{100.0 * e:.2f}")
        lines.append(f"  {name} & " + " & ".join(agree + err) + r" \\")
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_tex_table(out_path, lines, script="scripts/run_lmi_audit.py",
                    sources="results/lmi_audit.json")
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
    print(f"LMI AUDIT (R=1, sigma=8, N={args.N}, wall-clustered)")
    print("=" * 70)
    rows = []
    for name in args.metrics:
        for v_s in args.velocities:
            r = audit_one(name, v_s, args.N)
            rows.append(r)
            agree = r["typeI_nec_agree"]
            rate = r["certified_misclassification_rate"]
            print(f"  {len(rows):2d}/{len(args.metrics) * len(args.velocities)} "
                  f"{name:>15s} v_s={v_s:.2f}  "
                  f"TypeI agree={'n/a' if agree is None else f'{100 * agree:6.2f}%'}  "
                  f"III/IV={r['n_type_iii_iv']:7d}  "
                  f"certified label errors="
                  f"{'n/a' if rate is None else f'{100 * rate:.3f}%'}", flush=True)

    name = "lmi_audit_smoke.json" if args.smoke else "lmi_audit.json"
    dump_json({"config": vars(args), "rows": rows}, os.path.join(RESULTS_DIR, name))
    print(f"\nWrote {os.path.join(RESULTS_DIR, name)}")
    if not args.smoke:
        write_lmi_table(rows, os.path.join(TABLES_DIR, "lmi_typefree.tex"))


if __name__ == "__main__":
    main()
