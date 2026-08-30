"""Velocity scaling of the wall NEC deficit.

Santiago, Schuster and Visser proved that any physically reasonable warp drive
must violate the null energy condition somewhere. That result is an existence and
sign statement: the deficit is negative. It does not fix a speed law. Here we
compute the speed dependence for the matched-parameter family. For a unit-lapse,
spatially flat drive the shift is linear in the warp speed, so the leading wall
NEC deficit is quadratic,

    min(rho + p_i)  =  - C  v_s^2 ,

with a per-drive coefficient ``C``. We read the frame-independent wall NEC deficit
from the velocity sweep and fit it for each drive whose Type-I wall branch admits a
single power law: a fixed-exponent fit has ``R^2`` ~ 1 and the free exponent recovers
``q ~ 2``. The coefficient ``C`` is the per-drive fingerprint.

Two deviation figures are reported, and the difference between them is the point.
On an irrotational drive the whole wall is Type I at every speed and the law is
exact. On a vortical drive the Type-I set is a *residual* of a Type-IV-dominated
wall, and that residual is sparse at low speed -- tens of nodes at ``v_s = 0.1``
against thousands at ``v_s = 1``. The worst-case deviation is therefore dominated by
the sparsest speed and measures sampling, not physics; the deviation restricted to
``v_s >= 0.5``, where the residual is populated, measures the law. We print the
sparsest Type-I node count alongside both so the reader can tell which is which.

Outputs
-------
- results/ssv_bound.json
- ../warpax_arxiv/tables/ssv_bound.tex
"""
from __future__ import annotations

import json
import os

from _json_io import dump_json

import numpy as np

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]


def _load(rel):
    with open(os.path.join(RESULTS_DIR, rel)) as f:
        return json.load(f)


def _subluminal_deficits(rows, metric):
    """Return (v_s, |min(rho+p_i)|, n_type_i) for violating Type-I wall points."""
    vs, def_, n_i = [], [], []
    for r in rows:
        if r["metric"] != metric or r["v_s"] >= 1.0:
            continue
        nec = r.get("typeI_nec_min")
        if nec is None or not np.isfinite(nec) or nec >= 0.0:
            continue
        if r.get("n_type_i_wall", 0) < 1:
            continue
        vs.append(r["v_s"])
        def_.append(abs(nec))
        n_i.append(int(r["n_type_i_wall"]))
    order = np.argsort(vs)
    return np.array(vs)[order], np.array(def_)[order], np.array(n_i)[order]


# Speed above which the Type-I residual of a vortical wall is populated enough for
# the deviation to be a property of the law rather than of the sampling.
DENSE_VS = 0.5


def fit_bound(vs, deficits, n_type_i=None):
    """Fixed-exponent (q=2) velocity fit + free-exponent check.

    Fixed model: deficit = C v_s^2, least squares through the origin in v_s^2,
    so C = sum(deficit * v_s^2) / sum(v_s^4). Also report the free log-log
    exponent q and both R^2, plus the worst relative deviation of the data from
    the fixed-exponent fit.
    """
    if len(vs) < 3:
        return {"C": None, "r_squared_fixed": None, "q_free": None,
                "r_squared_free": None, "max_rel_dev": None,
                "max_rel_dev_dense": None, "n_type_i_min": None,
                "n": int(len(vs))}
    v2 = vs ** 2
    C = float(np.sum(deficits * v2) / np.sum(v2 ** 2))
    pred = C * v2
    ss_res = float(np.sum((deficits - pred) ** 2))
    # Through-origin fit (deficit = C v_s^2, no intercept): use the uncentered
    # total sum of squares. The mean-centered form is only valid for fits with
    # an intercept and here produces a spurious negative R^2 for poorly-fit
    # metrics (e.g. the Type-IV-dominated Van den Broeck branch).
    ss_tot = float(np.sum(deficits ** 2))
    r2_fixed = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rel_dev = np.abs(deficits - pred) / np.abs(deficits)
    max_rel_dev = float(np.max(rel_dev))
    dense = vs >= DENSE_VS
    max_rel_dev_dense = float(np.max(rel_dev[dense])) if dense.any() else None
    # Free exponent (log-log) for an independent check that q ~ 2.
    lv, ld = np.log(vs), np.log(deficits)
    q, logA = np.polyfit(lv, ld, 1)
    pred_l = q * lv + logA
    ss_res_l = float(np.sum((ld - pred_l) ** 2))
    ss_tot_l = float(np.sum((ld - np.mean(ld)) ** 2))
    r2_free = 1.0 - ss_res_l / ss_tot_l if ss_tot_l > 0 else 1.0
    # Two-term fit deficit = C2 v_s^2 + D v_s. D is the momentum correction:
    # zero for an irrotational shift, nonzero for vortical drives.
    A = np.vstack([v2, vs]).T
    (C2, D), *_ = np.linalg.lstsq(A, deficits, rcond=None)
    pred2 = A @ np.array([C2, D])
    r2_two = 1.0 - float(np.sum((deficits - pred2) ** 2)) / ss_tot if ss_tot > 0 else 1.0
    return {"C": C, "r_squared_fixed": float(r2_fixed), "q_free": float(q),
            "r_squared_free": float(r2_free), "max_rel_dev": max_rel_dev,
            "max_rel_dev_dense": max_rel_dev_dense,
            "dense_v_s_min": DENSE_VS,
            "n_type_i_min": int(np.min(n_type_i)) if n_type_i is not None else None,
            "n_type_i_max": int(np.max(n_type_i)) if n_type_i is not None else None,
            "C_two": float(C2), "D_two": float(D), "r_squared_two": float(r2_two),
            "n": int(len(vs))}


def _f(x, nd=3):
    return f"{x:.{nd}f}" if (x is not None and np.isfinite(x)) else "--"


def write_table(fits, out_path):
    lines = [
        r"% Generated by scripts/run_ssv_bound.py from results/velocity_sweep.json;"
        r" do not edit.",
        r"\begin{tabular}{@{}l ccccc@{}}",
        r"  \toprule",
        r"  Metric & $C$ & dev.\ (all $v_s$) & dev.\ ($v_s\ge0.5$) & $R^2$"
        r" & NEC $\forall\,v_s$ \\",
        r"  \midrule",
    ]

    def _pct(x):
        return f"{x*100:.2f}\\%" if (x is not None and np.isfinite(x)) else "--"

    for name in ORDER:
        fit = fits[name]
        r2 = fit.get("r_squared_fixed")
        if r2 is not None and np.isfinite(r2) and r2 >= 0.99 and fit.get("C", 0):
            lines.append(
                f"  {name} & {_f(fit.get('C'))} & {_pct(fit.get('max_rel_dev'))} & "
                f"{_pct(fit.get('max_rel_dev_dense'))} & "
                f"{_f(fit.get('r_squared_fixed'),4)} & violated \\\\"
            )
        else:
            # The gate is R^2 >= 0.99, not the existence of Type-I points: Van den
            # Broeck has a Type-I branch, it just admits no single power law.
            lines.append(
                rf"  {name} & \multicolumn{{4}}{{c}}{{no single power law "
                rf"($R^2={_f(r2, 2)}$)}} & violated \\"
            )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main():
    sweep = _load("velocity_sweep.json")
    rows = sweep["rows"]

    print("=" * 70)
    print("WALL NEC DEFICIT SPEED SCALING  min(rho+p_i) = -C v_s^2")
    print("=" * 70)
    fits = {}
    for name in ORDER:
        vs, deficits, n_type_i = _subluminal_deficits(rows, name)
        fit = fit_bound(vs, deficits, n_type_i)
        fits[name] = fit
        print(f"  {name:16s} C={_f(fit['C'])}  q_free={_f(fit['q_free'],2)}  "
              f"R^2={_f(fit['r_squared_fixed'],4)}  "
              f"maxdev={_f((fit['max_rel_dev'] or float('nan'))*100,2)}%  "
              f"dev(v_s>={DENSE_VS})="
              f"{_f((fit['max_rel_dev_dense'] or float('nan'))*100,2)}%  "
              f"n_typeI={fit['n_type_i_min']}..{fit['n_type_i_max']}  "
              f"n={fit['n']}")

    out = {
        "model": "min(rho+p_i) = -C v_s^2 (unit-lapse flat-slice velocity scaling)",
        # Provenance: this fit reads the velocity sweep, so it is only as current as
        # the sweep is. Recording the sweep's own config here makes a stale rerun
        # visible in the artifact instead of only in a file mtime.
        "source": {"file": "velocity_sweep.json", "config": sweep.get("config")},
        "fits": fits,
    }
    out_path = os.path.join(RESULTS_DIR, "ssv_bound.json")
    dump_json(out, out_path)
    print(f"\nWrote {out_path}")
    write_table(fits, os.path.join(TABLES_DIR, "ssv_bound.tex"))


if __name__ == "__main__":
    main()
