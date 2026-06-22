"""Boost-invariant exoticity ranking and v_s scaling laws.

Aggregates the frame-independent invariants already computed by the velocity
sweep and the rigorous ANEC into a single citable ranking across
all retained drives, and extracts universal ``v_s`` scaling laws.

Axes (all boost-invariant):
  - NEC severity ``|min(rho+p_i)|`` over wall Type-I points;
  - Hawking-Ellis Type-IV volume fraction (no-rest-frame content);
  - rigorous ANEC minimum (most-negative geodesic-integrated line integral).

Each axis is mapped to a [0,1] sub-score relative to the Alcubierre baseline and
combined by a geometric mean, so a construction that is clean on any axis (e.g.
the irrotational Rodal drive) scores low. The full per-axis vector is reported
alongside the scalar index -- never the scalar alone.

Scaling laws: for each metric the wall NEC severity is fit to ``|min(rho+p_i)|
= A v_s^p`` over the subluminal branch (log-log regression). Rodal's clean
``~0.688 v_s^2`` is recovered as the special case; the exponent is universal
(~2) and the coefficient is the per-drive fingerprint.

Outputs
-------
- results/exoticity_ranking.json
- ../warpax_arxiv/tables/scaling_laws.tex
- ../warpax_arxiv/tables/exoticity_ranking.tex
"""
from __future__ import annotations

import json
import math
import os

from _json_io import dump_json

import numpy as np

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]
REF_V_S = 0.5


def _load(path):
    with open(path) as f:
        return json.load(f)


def scaling_law_fit(rows, metric):
    """Fit |min(rho+p_i)| = A v_s^p over the subluminal Type-I branch."""
    vs, sev = [], []
    for r in rows:
        if r["metric"] != metric or r["v_s"] >= 1.0:
            continue
        nec = r.get("typeI_nec_min")
        if nec is None or not np.isfinite(nec) or nec >= 0.0:
            continue
        if r.get("n_type_i_wall", 0) < 1:
            continue
        vs.append(r["v_s"])
        sev.append(abs(nec))
    if len(vs) < 3:
        return {"A": None, "p": None, "r_squared": None, "n": len(vs)}
    lv, ls = np.log(np.array(vs)), np.log(np.array(sev))
    p, logA = np.polyfit(lv, ls, 1)
    pred = p * lv + logA
    ss_res = float(np.sum((ls - pred) ** 2))
    ss_tot = float(np.sum((ls - np.mean(ls)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"A": float(np.exp(logA)), "p": float(p), "r_squared": float(r2),
            "n": len(vs)}


def axis_values(rows, anec, metric, ref_v_s):
    """Raw boost-invariant axes for ``metric`` at the reference speed."""
    nec_sev = float("nan")
    type_iv = float("nan")
    for r in rows:
        if r["metric"] == metric and abs(r["v_s"] - ref_v_s) < 1e-9:
            nec = r.get("typeI_nec_min")
            nec_sev = abs(nec) if (nec is not None and np.isfinite(nec)) else 0.0
            type_iv = float(r.get("wall_frac_type_iv", float("nan")))
            break
    anec_min = float("nan")
    if anec is not None and metric in anec.get("metrics", {}):
        anec_min = abs(float(anec["metrics"][metric]["min_line_integral"]))
    return {"nec_severity": nec_sev, "type_iv_frac": type_iv, "anec_min_abs": anec_min}


def _safe_ratio(x, ref):
    if not np.isfinite(x) or not np.isfinite(ref) or ref <= 0:
        return float("nan")
    return min(x / ref, 1.0)


def exoticity_index(axes, baseline_axes):
    """Geometric mean of [0,1] sub-scores relative to the Alcubierre baseline."""
    s_nec = _safe_ratio(axes["nec_severity"], baseline_axes["nec_severity"])
    s_iv = axes["type_iv_frac"] if np.isfinite(axes["type_iv_frac"]) else float("nan")
    s_anec = _safe_ratio(axes["anec_min_abs"], baseline_axes["anec_min_abs"])
    subs = [s for s in (s_nec, s_iv, s_anec) if np.isfinite(s)]
    if not subs:
        return {"index": float("nan"), "s_nec": s_nec, "s_type_iv": s_iv,
                "s_anec": s_anec}
    # Geometric mean with a small floor so a zero axis does not annihilate it.
    floored = [max(s, 1e-4) for s in subs]
    idx = math.exp(sum(math.log(s) for s in floored) / len(floored))
    return {"index": idx, "s_nec": s_nec, "s_type_iv": s_iv, "s_anec": s_anec}


def _f(x, nd=3):
    return f"{x:.{nd}f}" if (x is not None and np.isfinite(x)) else "--"


def write_scaling_table(fits, out_path):
    lines = [
        r"\begin{tabular}{@{}l ccc@{}}",
        r"  \toprule",
        r"  Metric & exponent $p$ & coefficient $A$ & $R^2$ \\",
        r"  \midrule",
    ]
    for name in ORDER:
        fit = fits.get(name, {})
        r2 = fit.get("r_squared")
        # Only report a power law where the fit is clean (R^2 >= 0.99); a noisy
        # Type-IV-dominated wall (e.g. VdB) has no resolved Type-I branch.
        if r2 is not None and np.isfinite(r2) and r2 >= 0.99:
            lines.append(
                f"  {name} & {_f(fit.get('p'),2)} & {_f(fit.get('A'),3)} & "
                f"{_f(r2,4)} \\\\"
            )
        else:
            lines.append(
                rf"  {name} & \multicolumn{{3}}{{c}}{{no clean fit "
                rf"($R^2 < 0.99$)}} \\"
            )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def write_ranking_table(scores, out_path):
    lines = [
        r"\begin{tabular}{@{}l cccc@{}}",
        r"  \toprule",
        r"  Metric & NEC & Type~IV & ANEC & Exoticity \\",
        r"  & severity & fraction & $|\min|$ & index \\",
        r"  \midrule",
    ]
    for name in ORDER:
        s = scores.get(name, {})
        lines.append(
            f"  {name} & {_f(s.get('s_nec'),3)} & {_f(s.get('s_type_iv'),3)} & "
            f"{_f(s.get('s_anec'),3)} & {_f(s.get('index'),3)} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main():
    sweep = _load(os.path.join(RESULTS_DIR, "velocity_sweep.json"))
    rows = sweep["rows"]
    anec_path = os.path.join(RESULTS_DIR, "anec", "retained_symplectic.json")
    anec = _load(anec_path) if os.path.exists(anec_path) else None

    print("=" * 70)
    print(f"EXOTICITY RANKING + SCALING LAWS  (reference v_s={REF_V_S})")
    print("=" * 70)

    fits = {name: scaling_law_fit(rows, name) for name in ORDER}
    print("  NEC severity scaling  |min(rho+p_i)| = A v_s^p:")
    for name in ORDER:
        fl = fits[name]
        print(f"    {name:16s} p={_f(fl['p'],2)}  A={_f(fl['A'],3)}  R^2={_f(fl['r_squared'],4)}")

    raw = {name: axis_values(rows, anec, name, REF_V_S) for name in ORDER}
    baseline = raw["Alcubierre"]
    scores = {name: exoticity_index(raw[name], baseline) for name in ORDER}
    print("  Exoticity index (lower = less exotic):")
    for name in ORDER:
        print(f"    {name:16s} index={_f(scores[name]['index'],3)}  "
              f"[NEC={_f(scores[name]['s_nec'],2)} "
              f"IV={_f(scores[name]['s_type_iv'],2)} "
              f"ANEC={_f(scores[name]['s_anec'],2)}]")

    out = {
        "reference_v_s": REF_V_S,
        "scaling_laws": fits,
        "raw_axes": raw,
        "scores": scores,
    }
    out_path = os.path.join(RESULTS_DIR, "exoticity_ranking.json")
    dump_json(out, out_path)
    print(f"\nWrote {out_path}")
    write_scaling_table(fits, os.path.join(TABLES_DIR, "scaling_laws.tex"))
    write_ranking_table(scores, os.path.join(TABLES_DIR, "exoticity_ranking.tex"))


if __name__ == "__main__":
    main()
