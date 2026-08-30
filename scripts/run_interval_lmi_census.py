"""Every wall point decided from the metric: a census, not a spot check.

``run_interval_lmi_spotcheck.py`` certifies three points per construction, which
supports a qualitative claim -- that the type-free decision survives being taken
back to the metric -- and nothing more. This sweeps the whole active wall band
instead, so the resulting statement is a count: of the sampled wall points, how
many had each of the null, weak, strong and dominant conditions *certified*
violated, how many certified satisfied, and how many the interval arithmetic
declined to decide.

Why it is worth the compute. The verdicts here consult no Hawking--Ellis type, no
eigendecomposition, no classification tolerance and no rapidity cap; the whole
chain ``g -> Gamma -> Riem -> Ric -> G -> T`` is enclosed in interval arithmetic
and the acceptance test is an interval ``LDL^T``. So the count is immune to the
two standing objections to a type-based census -- that Type IV identification is
tolerance-dependent, and that a float64 eigensolver cannot separate the Type II
and Type III strata. Whatever the algebraic type at a point is, the verdict does
not ask.

What it is NOT. Each entry counts *sampled points*, not wall measure. A rigorous
statement about the continuum needs the branch-and-bound of ``run_enclosures.py``,
which brackets the infimum over the whole band rather than over a finite set. The
two are complements: this one is broad and pointwise, that one is narrow and
global. Do not quote a count here as a fraction of the wall.

Cost. About 20 ms per point at 80 bits -- the seconds-per-point figure in this
codebase belongs to the branch-and-bound, which evaluates thousands of boxes per
tree, not to a single point evaluation. The default budget is ~6 min serial and
under a minute across the machine, so per-metric checkpointing is enough
insurance and per-point checkpointing would cost more in JSON churn than the
compute it protects.

Outputs
-------
- results/interval_lmi_census.json
- ../warpax_arxiv/tables/interval_lmi_census.tex
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import mpmath
from mpmath import iv

from _json_io import dump_json, write_table as write_tex_table

from warpax.energy_conditions.enclosure import (
    _hi,
    _lo,
    alcubierre_metric,
    natario_metric,
    rodal_metric,
    shape_interval,
    van_den_broeck_metric,
)
from warpax.energy_conditions.interval_lmi import certify_point_from_metric

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

V_S, R_B, SIGMA = 0.5, 1.0, 8.0
VDB_KW = dict(R_tilde=1.0, alpha_vdb=0.5, sigma_B=8.0)
F_LOW, F_HIGH = 0.1, 0.9

BUILDERS = {
    "Alcubierre": lambda: alcubierre_metric(V_S, R_B, SIGMA),
    "Natário": lambda: natario_metric(V_S, R_B, SIGMA),
    "Van den Broeck": lambda: van_den_broeck_metric(V_S, R_B, SIGMA, **VDB_KW),
    "Rodal": lambda: rodal_metric(V_S, R_B, SIGMA),
}
ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]
CONDITIONS = ["nec", "wec", "sec", "dec"]
COND_LABEL = {"nec": "NEC", "wec": "WEC", "sec": "SEC", "dec": "DEC"}


def band_radii(R: float = R_B, sigma: float = SIGMA) -> tuple[float, float]:
    """Radii where the tanh top-hat sits in ``[F_LOW, F_HIGH]``.

    ``f(r) = [tanh s(r+R) - tanh s(r-R)] / (2 tanh sR)``, monotone decreasing on
    ``r > 0``, so the band is the interval between the two solutions of ``f = F_HIGH``
    and ``f = F_LOW``. Bisected rather than inverted: the closed form is ugly and
    this runs once.
    """
    def f(r):
        return ((math.tanh(sigma * (r + R)) - math.tanh(sigma * (r - R)))
                / (2.0 * math.tanh(sigma * R)))

    def solve(target):
        lo, hi = 0.0, 10.0 * R
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    return solve(F_HIGH), solve(F_LOW)


def sample_points(n_r: int, n_theta: int) -> list[tuple[float, float]]:
    """Product grid in ``(r, theta)`` on the axisymmetry-reduced half-plane.

    Radial nodes are INTERIOR to the band. At an endpoint the interval enclosure of
    ``f`` straddles the band boundary, so the point cannot be certified to be a wall
    point at all and would have to be dropped -- which is the honest outcome, but a
    pointless one when moving half a step inward avoids it entirely.
    """
    import numpy as np

    r_lo, r_hi = band_radii()
    radii = np.linspace(r_lo, r_hi, n_r + 2)[1:-1]
    thetas = np.linspace(0.0, math.pi, n_theta)
    return [(float(r * math.cos(t)), float(abs(r * math.sin(t))))
            for r in radii for t in thetas]


def in_band(shape_fn, x: float, s: float) -> tuple[float, float] | None:
    """Certify ``F_LOW <= f <= F_HIGH`` at ``(x, s)``, in interval arithmetic.

    The float radius that placed the node is not evidence that the node IS a wall
    point: that is a claim about the spacetime and is certified like any other.
    Returns the enclosure of ``f``, or ``None`` when band membership cannot be
    certified.
    """
    pt = iv.mpf([x, x]), iv.mpf([s, s])
    f_iv = shape_fn(*pt)
    f_lo, f_hi = _lo(f_iv), _hi(f_iv)
    if f_lo >= F_LOW and f_hi <= F_HIGH:
        return f_lo, f_hi
    return None


_WORKER: dict = {}


def _init_worker(name: str, prec: int) -> None:
    """Rebuild the metric inside the worker.

    The interval metric constructors return a closure over a nested function and do
    not pickle across ``spawn``. Never put the metric object in a job tuple; build
    it here, exactly as ``run_enclosures.certify_one`` does.
    """
    mpmath.mp.prec = prec
    iv.prec = prec
    _WORKER["metric"] = BUILDERS[name]()
    _WORKER["shape"] = shape_interval(R_B, SIGMA)
    _WORKER["prec"] = prec


def _certify_one_point(xs: tuple[float, float]) -> dict:
    x, s = xs
    band = in_band(_WORKER["shape"], x, s)
    if band is None:
        return {"x": x, "s": s, "status": "out_of_band"}
    try:
        r = certify_point_from_metric(_WORKER["metric"], (0.0, x, s, 0.0),
                                      prec=_WORKER["prec"])
    except (ZeroDivisionError, ValueError) as exc:
        # A chain that could not be EVALUATED and a chain that evaluated and could
        # not DECIDE are different failures. Never fold the first into "refused".
        return {"x": x, "s": s, "status": "chain_failed",
                "error": f"{type(exc).__name__}: {exc}"}
    r["x"], r["s"], r["f"] = x, s, list(band)
    r["status"] = "ok"
    return r


def _tally(rows: list[dict]) -> dict:
    """Counts per condition, plus the deepest certified violation as an exemplar."""
    ok = [r for r in rows if r["status"] == "ok"]
    out = {
        "n_sampled": len(rows),
        "n_certified_points": len(ok),
        "n_out_of_band": sum(1 for r in rows if r["status"] == "out_of_band"),
        "n_chain_failed": sum(1 for r in rows if r["status"] == "chain_failed"),
        "conditions": {},
    }
    for cond in CONDITIONS:
        viol = [r for r in ok if r[cond] == "violated"]
        deepest = min(viol, key=lambda r: r[f"{cond}_upper"], default=None)
        out["conditions"][cond] = {
            "violated": len(viol),
            "satisfied": sum(1 for r in ok if r[cond] == "satisfied"),
            "inconclusive": sum(1 for r in ok if r[cond] == "inconclusive"),
            "deepest_upper": None if deepest is None else deepest[f"{cond}_upper"],
            "deepest_at": None if deepest is None else [deepest["x"], deepest["s"]],
        }
    out["errors"] = [
        {"x": r["x"], "s": r["s"], "error": r["error"]}
        for r in rows if r["status"] == "chain_failed"
    ][:32]
    return out


def census_one(name: str, points, prec: int, jobs: int) -> dict:
    t0 = time.time()
    if jobs > 1:
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
        with ctx.Pool(jobs, initializer=_init_worker, initargs=(name, prec)) as pool:
            rows = list(pool.imap_unordered(_certify_one_point, points, chunksize=32))
    else:
        _init_worker(name, prec)
        rows = [_certify_one_point(p) for p in points]
    tally = _tally(rows)
    tally["seconds"] = round(time.time() - t0, 1)
    return tally


def write_table(results: dict, sampling: dict, out_path: str) -> None:
    lines = [
        r"\begin{tabular}{@{}l l rrr r@{}}",
        r"  \toprule",
        r"  Construction & condition & certified & certified & refused"
        r" & deepest \\",
        r"   &  & violated & satisfied &  & upper \\",
        r"  \midrule",
    ]
    for k, name in enumerate(ORDER):
        r = results.get(name)
        if r is None:
            continue
        for i, cond in enumerate(CONDITIONS):
            c = r["conditions"][cond]
            deep = ("--" if c["deepest_upper"] is None
                    else f"${c['deepest_upper']:+.4f}$")
            lines.append(
                f"  {name if i == 0 else ''} & {COND_LABEL[cond]} & "
                f"{c['violated']} & {c['satisfied']} & {c['inconclusive']} & "
                f"{deep} \\\\"
            )
        if k < len(ORDER) - 1:
            lines.append(r"  \cmidrule(l){1-6}")
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_tex_table(out_path, lines, script="scripts/run_interval_lmi_census.py",
                    sources="results/interval_lmi_census.json")
    print(f"  Wrote {out_path}")


NOTE = (
    "Each entry counts sampled wall points, not wall measure. The verdict at each "
    "point is a rigorous statement about the spacetime at that point, obtained "
    "without any Hawking-Ellis type, eigendecomposition, classification tolerance "
    "or rapidity cap; the table is a statement about this finite set of points and "
    "not about the continuum. 'Refused' is the interval method declining to decide "
    "and is not evidence that the condition holds. The continuum statement for the "
    "null condition is the branch-and-bound bracket of results/enclosures.json."
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prec", type=int, default=80)
    ap.add_argument("--n-r", type=int, default=33)
    ap.add_argument("--n-theta", type=int, default=129)
    ap.add_argument("--metrics", nargs="+", default=ORDER)
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes; 0 = cpu_count - 2, 1 = serial")
    ap.add_argument("--smoke", action="store_true",
                    help="5 x 9 nodes per construction, for a wiring check")
    ap.add_argument("--table-only", action="store_true",
                    help="rewrite the table from the existing JSON; no compute")
    args = ap.parse_args()

    out_json = os.path.join(RESULTS_DIR, "interval_lmi_census.json")
    if args.table_only:
        with open(out_json) as fh:
            payload = json.load(fh)
        if payload.get("partial", False):
            raise SystemExit(f"{out_json} is a partial checkpoint; not writing a table")
        write_table(payload["results"], payload["sampling"],
                    os.path.join(TABLES_DIR, "interval_lmi_census.tex"))
        return payload["results"]

    if args.smoke:
        args.n_r, args.n_theta = 5, 9
    if args.jobs <= 0:
        args.jobs = max(1, (os.cpu_count() or 2) - 2)

    r_lo, r_hi = band_radii()
    points = sample_points(args.n_r, args.n_theta)
    sampling = {
        "rule": "product grid in (r, theta) on the reduced half-plane; interior "
                "radial nodes of the f-band; theta over the closed [0, pi]",
        "n_r": args.n_r, "n_theta": args.n_theta, "n_points": len(points),
        "f_band": [F_LOW, F_HIGH], "r_band": [r_lo, r_hi],
        "dr": (r_hi - r_lo) / (args.n_r + 1),
        "dtheta": math.pi / max(1, args.n_theta - 1),
    }

    print("=" * 72)
    print(f"INTERVAL LMI CENSUS  (v_s={V_S}, R_b={R_B}, sigma={SIGMA}, "
          f"prec={args.prec} bits)")
    print(f"  band r in [{r_lo:.4f}, {r_hi:.4f}], {len(points)} points per "
          f"construction, {args.jobs} workers")
    print("=" * 72)

    results: dict = {}
    for name in args.metrics:
        results[name] = census_one(name, points, args.prec, args.jobs)
        c = results[name]["conditions"]
        print(f"  {name:>15s}  " + "  ".join(
            f"{COND_LABEL[k]} {c[k]['violated']}/{c[k]['satisfied']}/"
            f"{c[k]['inconclusive']}" for k in CONDITIONS)
            + f"   ({results[name]['seconds']}s)", flush=True)
        dump_json({"partial": True, "config": vars(args),
                   "params": {"v_s": V_S, "R_b": R_B, "sigma": SIGMA, **VDB_KW},
                   "sampling": sampling, "note": NOTE, "order": ORDER,
                   "results": results}, out_json)

    dump_json({"partial": False, "config": vars(args),
               "params": {"v_s": V_S, "R_b": R_B, "sigma": SIGMA, **VDB_KW},
               "sampling": sampling, "note": NOTE, "order": ORDER,
               "results": results}, out_json)
    write_table(results, sampling,
                os.path.join(TABLES_DIR, "interval_lmi_census.tex"))
    return results


if __name__ == "__main__":
    main()
