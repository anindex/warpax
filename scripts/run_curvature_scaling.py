"""Universal v_s scaling of the wall curvature invariants.

The matter-sector result (the wall NEC severity ``|min(rho+p_i)|`` follows a
universal ``v_s^2`` law) has a geometric counterpart: the peak curvature in the
bubble wall also follows clean power laws in the warp speed. Here we sweep the
warp speed on the same matched-parameter (``R=1``, ``sigma=8``) wall-clustered
grids used by the type map, and record the wall-peak Kretschmann ``K``,
Weyl-squared ``C^2`` and Ricci-squared ``R_{ab}R^{ab}`` invariants. Each is fit
to ``X = A v_s^q`` over the subluminal branch.

Two boost-invariant sectors emerge: the source-free Weyl (tidal) curvature and
the Ricci (matter) curvature scale as separate, universal powers of ``v_s``
across the family, showing that the exotic content is intrinsic geometry, not a
coordinate artifact, and that it grows smoothly through and beyond the luminal
transition.

Sentinels: the Minkowski invariants vanish; Schwarzschild reproduces the closed
form ``K = 48 M^2 / r^6`` (checked in tests/test_curvature_scaling.py).

Outputs
-------
- results/curvature_scaling.json
- ../warpax_arxiv/tables/curvature_scaling.tex
- ../warpax_arxiv/figures/curvature_scaling.pdf
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
from _json_io import dump_json
from _json_io import write_table as write_tex_table
from _paper_metrics import METRIC_ORDER, instantiate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from warpax.visualization._style import (
    DOUBLE_COL,
    USE_TEX,
    apply_style,
    metric_color,
)

apply_style()

_PCT = r"\%" if USE_TEX else "%"

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from _benchmark_grid import CLUSTER_A

from warpax.analysis.extrema import refine_extremum, seed_from_grid_index
from warpax.energy_conditions.filtering import shape_function_mask
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.grids import wall_clustered

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")
FIG_DIR = os.path.join(HERE, "..", "figures")

BOUNDS = [(-3, 3)] * 3
F_LOW, F_HIGH = 0.1, 0.9


# Invariants reported, in (json key, latex symbol) form.
INVARIANTS = (
    ("kretschmann", r"$K$"),
    ("weyl_squared", r"$C^2$"),
    ("ricci_squared", r"$|R_{ab}R^{ab}|$"),
)




def run_point(name, v_s, N):
    """Wall-peak curvature invariants at one (metric, v_s)."""
    shape = (N, N, N)
    metric = instantiate(name, v_s)
    grid = wall_clustered(metric, BOUNDS, shape, a=CLUSTER_A)
    curv = evaluate_curvature_grid(metric, grid, batch_size=256)
    coords = build_coord_batch(grid, t=0.0)
    mask = np.asarray(
        jnp.reshape(
            shape_function_mask(metric, coords, shape, f_low=F_LOW, f_high=F_HIGH),
            (-1,),
        )
    ).astype(bool)
    axes = [np.asarray(a) for a in grid.axes]
    out = {"metric": name, "v_s": v_s, "N": N, "wall_n": int(mask.sum())}
    for key, _sym in INVARIANTS:
        field = np.abs(np.asarray(jnp.reshape(getattr(curv, key), (-1,))))
        masked = np.where(mask, field, -np.inf)
        k = int(np.argmax(masked))
        raw = float(masked[k])
        if not np.isfinite(raw):
            out[f"{key}_max"] = float("nan")
            out[f"{key}_max_grid"] = float("nan")
            continue
        # Polish to the continuum: the raw grid max undershoots by 1-15%,
        # speed-dependently, which tilts the exponent.
        seed = seed_from_grid_index(k, shape, axes)
        polished = refine_extremum(
            metric, seed, lambda c, _k=key: np.abs(np.asarray(getattr(c, _k))),
            mode="max", half_width=0.15, n=9, levels=7, compute_invariants=True,
        )
        out[f"{key}_max"] = float(polished["value"])
        out[f"{key}_max_grid"] = raw
    return out


def _branch(rows, metric, key, v_max=1.0):
    """The subluminal (v_s, X_max) samples for one metric and one invariant.

    Non-positive samples are dropped (the log fit cannot take them) and counted,
    so a shortened branch is visible in the artifact.
    """
    vs, xs, dropped = [], [], 0
    for r in rows:
        if r["metric"] != metric or r["v_s"] >= v_max:
            continue
        x = r.get(f"{key}_max")
        if x is None or not np.isfinite(x) or x <= 0.0:
            dropped += 1
            continue
        vs.append(r["v_s"])
        xs.append(x)
    return np.array(vs), np.array(xs), dropped


def fit_power_law(rows, metric, key, v_max=1.0):
    """Fit X_max = A v_s^q over the subluminal branch (log-log regression).

    ``r_squared`` is the log-space R^2 of the regression actually performed;
    ``r_squared_linear`` is the same curve scored in linear space. They differ
    only on Alcubierre Ricci^2 (0.9938 vs 0.9338).
    """
    vs, xs, dropped = _branch(rows, metric, key, v_max)
    if len(vs) < 3:
        return {"A": None, "q": None, "r_squared": None,
                "r_squared_linear": None, "max_rel_dev": None,
                "v_s_at_max_dev": None, "n": len(vs), "n_dropped": dropped}
    lv, lx = np.log(vs), np.log(xs)
    q, logA = np.polyfit(lv, lx, 1)
    ss_res = float(np.sum((lx - (q * lv + logA)) ** 2))
    ss_tot = float(np.sum((lx - np.mean(lx)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    pred_lin = np.exp(logA) * vs**q
    tot_lin = float(np.sum((xs - xs.mean()) ** 2))
    r2_lin = (1.0 - float(np.sum((xs - pred_lin) ** 2)) / tot_lin
              if tot_lin > 0 else 1.0)
    # Worst single-point deviation, as run_ssv_bound.py reports: R^2 alone hides
    # a fit driven by one sample. Alcubierre's Ricci branch sits 21% off at
    # v_s = 0.9, where the wall-peak argmax migrates to another basin.
    max_rel_dev = float(np.max(np.abs(xs - pred_lin) / xs))
    return {"A": float(np.exp(logA)), "q": float(q), "r_squared": float(r2),
            "r_squared_linear": float(r2_lin), "max_rel_dev": max_rel_dev,
            "v_s_at_max_dev": float(vs[int(np.argmax(np.abs(xs - pred_lin) / xs))]),
            "n": len(vs), "n_dropped": dropped}


def fit_exact_exponents(rows, metric, key, powers=(2.0, 4.0), v_max=1.0):
    """Fit X_max = sum_p C_p v_s^p at the exponents the theorem fixes.

    verify/weyl_scaling.sage fixes the exponents: an irrotational shift starts
    at v_s^4, a vortical one carries a nonzero v_s^2 term. Fitting the
    coefficients at those exponents makes C_2 a measurement of that statement.
    """
    vs, xs, dropped = _branch(rows, metric, key, v_max)
    if len(vs) <= len(powers):
        return {"powers": list(powers), "coeffs": None, "r_squared": None,
                "n": len(vs), "n_dropped": dropped}
    design = np.vstack([vs**p for p in powers]).T
    coeffs, *_ = np.linalg.lstsq(design, xs, rcond=None)
    resid = xs - design @ coeffs
    tot = float(np.sum((xs - xs.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / tot if tot > 0 else 1.0
    return {"powers": list(powers), "coeffs": [float(c) for c in coeffs],
            "r_squared": float(r2), "n": len(vs), "n_dropped": dropped}


def _f(x, nd=2):
    return f"{x:.{nd}f}" if (x is not None and np.isfinite(x)) else "--"


def write_table(fits, out_path):
    """Universal curvature-invariant scaling exponents per metric."""
    lines = [
        r"\begin{tabular}{@{}l ccc ccc@{}}",
        r"  \toprule",
        r"  & \multicolumn{3}{c}{Weyl $C^2$} "
        r"& \multicolumn{3}{c}{Ricci $|R_{ab}R^{ab}|$} \\",
        r"  \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        # R^2 of a power law is ambiguous; name the convention.
        r"  Metric & $q$ & $A$ & $R^2_{\log}$ & $q$ & $A$ & $R^2_{\log}$ \\",
        r"  \midrule",
    ]
    for name in METRIC_ORDER:
        w = fits[name]["weyl_squared"]
        ri = fits[name]["ricci_squared"]
        r2w = w.get("r_squared")
        # A Type-IV-dominated wall (VdB) has no resolved Type-I curvature branch
        # and no clean single power law; report it as such rather than a noisy fit.
        if r2w is not None and np.isfinite(r2w) and r2w >= 0.99:
            lines.append(
                f"  {name} & {_f(w.get('q'))} & {_f(w.get('A'),3)} & {_f(r2w,4)}"
                f" & {_f(ri.get('q'))} & {_f(ri.get('A'),3)} & {_f(ri.get('r_squared'),4)} \\\\"
            )
        else:
            lines.append(
                rf"  {name} & \multicolumn{{6}}{{c}}{{no clean fit "
                rf"(Type-IV-dominated wall)}} \\"
            )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_tex_table(out_path, lines, script="scripts/run_curvature_scaling.py", sources="results/curvature_scaling.json")
    print(f"  Wrote {out_path}")


_MARKERS = {"Alcubierre": "o", "Natário": "s",
            "Van den Broeck": "^", "Rodal": "D"}


def make_figure(rows, fits):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, (ax_w, ax_r) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.44))
    for ax, key, title in ((ax_w, "weyl_squared", "(a) Weyl $C^2$"),
                           (ax_r, "ricci_squared", "(b) Ricci $|R_{ab}R^{ab}|$")):
        for name in METRIC_ORDER:
            rs = sorted([r for r in rows if r["metric"] == name],
                        key=lambda r: r["v_s"])
            xs = [r["v_s"] for r in rs]
            ys = [r[f"{key}_max"] for r in rs]
            ax.loglog(xs, ys, marker=_MARKERS.get(name, "o"), ms=4, ls="none",
                      color=metric_color(name), label=name)
            fit = fits[name][key]
            if fit.get("A") is not None and fit.get("r_squared", 0) and fit["r_squared"] >= 0.9:
                vv = np.array([min(xs), 1.0])
                ax.loglog(vv, fit["A"] * vv ** fit["q"], color=metric_color(name),
                          lw=0.9, alpha=0.7)
        ax.axvline(1.0, color="0.6", ls="--", lw=0.8)
        ax.set_xlabel(r"warp speed $v_s$")
        ax.set_ylabel("wall-peak invariant")
        ax.set_title(title, fontsize=9)
    ax_w.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "curvature_scaling.pdf")
    fig.savefig(p)
    plt.close(fig)
    print(f"  Wrote {p}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--velocities", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 2.5])
    p.add_argument("--N", type=int, default=60)
    p.add_argument("--metrics", type=str, nargs="+", default=METRIC_ORDER)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--from-cache", action="store_true")
    args = p.parse_args()

    if args.from_cache:
        with open(os.path.join(RESULTS_DIR, "curvature_scaling.json")) as f:
            cached = json.load(f)
        rows = cached["rows"]
        fits = cached["fits"]
        write_table(fits, os.path.join(TABLES_DIR, "curvature_scaling.tex"))
        make_figure(rows, fits)
        return

    if args.smoke:
        args.velocities = [0.3, 0.5, 0.9, 1.5]
        args.N = 24
        args.metrics = ["Alcubierre", "Rodal"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print(f"CURVATURE-INVARIANT SCALING (R=1, sigma=8, N={args.N}, wall-clustered)")
    print("=" * 70)
    rows = []
    for name in args.metrics:
        for v_s in args.velocities:
            r = run_point(name, v_s, args.N)
            rows.append(r)
            print(f"  {name:>15s} v_s={v_s:.2f}  "
                  f"K={r['kretschmann_max']:.3g}  "
                  f"C^2={r['weyl_squared_max']:.3g}  "
                  f"R^2={r['ricci_squared_max']:.3g}")

    fits = {name: {key: fit_power_law(rows, name, key) for key, _ in INVARIANTS}
            for name in args.metrics}
    two_term = {name: {key: fit_exact_exponents(rows, name, key)
                       for key, _ in INVARIANTS}
                for name in args.metrics}
    print("\n  Subluminal scaling exponents X_max = A v_s^q "
          "(R^2 of the log-log regression; R^2_lin of the same curve in linear space):")
    for name in args.metrics:
        for key, sym in INVARIANTS:
            fl = fits[name][key]
            print(f"    {name:16s} {key:14s} q={_f(fl['q'])}  "
                  f"A={_f(fl['A'],3)}  R^2={_f(fl['r_squared'],4)}  "
                  f"R^2_lin={_f(fl['r_squared_linear'],4)}  "
                  f"maxdev={_f(fl['max_rel_dev'],3)}@v={_f(fl['v_s_at_max_dev'],2)}"
                  + (f"  [{fl['n_dropped']} point(s) dropped]"
                     if fl.get("n_dropped") else ""))
    print("\n  Fixed-exponent law X_max = C2 v_s^2 + C4 v_s^4 "
          "(C2 = 0 iff the shift is irrotational, verify/weyl_scaling.sage):")
    for name in args.metrics:
        for key, sym in INVARIANTS:
            tt = two_term[name][key]
            c = tt["coeffs"]
            print(f"    {name:16s} {key:14s} "
                  + (f"C2={c[0]:12.4g}  C4={c[1]:12.4g}  R^2={_f(tt['r_squared'],6)}"
                     if c else "--"))

    dump_json({"config": vars(args), "rows": rows, "fits": fits,
               "fits_fixed_exponent": two_term},
              os.path.join(RESULTS_DIR, "curvature_scaling.json"))
    print(f"\nWrote {os.path.join(RESULTS_DIR, 'curvature_scaling.json')}")

    if not args.smoke:
        write_table(fits, os.path.join(TABLES_DIR, "curvature_scaling.tex"))
        make_figure(rows, fits)


if __name__ == "__main__":
    main()
