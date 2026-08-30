"""Certified global enclosures of the wall null-energy deficit (paper Appendix H).

Supplies the global-coverage error bound that accompanies the
basin-local continuum polishing of ``run_diagnostic_convergence.py``.

For each metric we run an interval-arithmetic Moore-Skelboe branch-and-bound over the
axisymmetry-reduced domain ``(x, s)``, ``s = sqrt(y^2 + z^2)``, and report

    lower <= inf_{wall} N <= upper,

where ``N(x) = min_{|v|=1} T_ab k^a k^b`` with ``k = n + v`` is the Eulerian-normalized
null deficit, which is defined at every point regardless of Hawking-Ellis type.

The two ends are not equally sharp. ``upper`` is an *achieved* value, evaluated on a
degenerate box with the same rigorous arithmetic, so it is a wall value.
``lower`` is a continuum bound, valid at every point, not merely at sampled ones,
which is the global statement local refinement cannot supply, but it is loose,
because the interval extension of the full curvature chain overestimates. Report the
width; do not suppress it.

For an irrotational drive the Eulerian momentum vanishes identically, so
``N = min_i(rho + p_i)`` and the search reproduces the paper's polished invariant
margin by an independent route. For the vortical walls ``N`` is the all-type deficit
and is deeper than the Type-I-restricted margin, which samples only the residual
real-eigenvalue points.

Outputs
-------
- results/enclosures.json
- ../warpax_arxiv/tables/enclosures.tex
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

from _json_io import dump_json

from warpax.energy_conditions.enclosure import (
    alcubierre_metric,
    certify_nec_deficit,
    natario_metric,
    rodal_metric,
    shape_interval,
    tail_bound,
    van_den_broeck_metric,
)

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

V_S, R_B, SIGMA = 0.5, 1.0, 8.0
BOX_X = (-3.0, 3.0)
BOX_S = (0.0, 3.0)
# Only the inner radius is read: tail_bound bounds f on all of r_s >= r_min by
# monotonicity, so the exterior is covered unbounded.
TAIL_X = (3.0, 30.0)
TAIL_S = (0.0, 30.0)

# Comparison column, loaded from the artifact that produces it with the key path
# recorded beside the value. The statistic differs by drive, so the column carries
# its name too: Rodal's is the polished min_i(rho + p_i), Alcubierre's the all-wall
# NEC minimum. Natario and Van den Broeck have no comparable one and stay blank.
REFERENCE_SOURCES = {
    "Alcubierre": (
        "comparison_table.json",
        ("metric", "alcubierre", "v_s", V_S, "nec_robust_min"),
        # Name the section by what it is, not by a number: the number moves
        # and this string is baked into the artifact and the table.
        r"all-wall NEC min, per-metric section",
    ),
    "Rodal": (
        "diagnostic_convergence.json",
        ("results", "Rodal", "nec_min", "polished"),
        r"polished $\min(\rho+p_i)$",
    ),
    "Natario": None,
    "VanDenBroeck": None,
}


def _load_references():
    """Read the comparison column from the artifacts that produce it.

    Missing or moved is a hard failure, not a silent ``--``: a blank cell means
    "no comparable statistic exists", which is the state for Natario and Van den
    Broeck, and must not double as "the lookup broke".
    """
    out = {}
    for name, spec in REFERENCE_SOURCES.items():
        if spec is None:
            out[name] = None
            continue
        fname, path, label = spec
        with open(os.path.join(RESULTS_DIR, fname)) as fh:
            doc = json.load(fh)
        if fname == "comparison_table.json":
            key_a, val_a, key_b, val_b, field = path
            hits = [
                r for r in doc if r.get(key_a) == val_a and abs(r.get(key_b, 1e9) - val_b) < 1e-12
            ]
            if len(hits) != 1:
                raise RuntimeError(
                    f"{fname}: expected exactly one row with {key_a}={val_a!r}, "
                    f"{key_b}={val_b}; found {len(hits)}"
                )
            value = hits[0][field]
            where = f"{fname}:{key_a}={val_a},{key_b}={val_b}.{field}"
        else:
            node = doc
            for k in path:
                node = node[k]
            value = node
            where = f"{fname}:" + ".".join(str(k) for k in path)
        out[name] = (float(value), label, where)
    return out


# Van den Broeck's conformal parameters. Same tanh top-hat shift as Alcubierre, so
# the wall mask is unchanged; the slice carries B^2 delta_ij. R_tilde MUST stay 1.0
# to match the benchmark, or the bracket certifies a different spacetime.
VDB_KW = dict(R_tilde=1.0, alpha_vdb=0.5, sigma_B=8.0)

BUILDERS = {
    "Alcubierre": alcubierre_metric,
    "Rodal": rodal_metric,
    "Natario": natario_metric,
    "VanDenBroeck": lambda v, R, s: van_den_broeck_metric(v, R, s, **VDB_KW),
}
ORDER = ["Alcubierre", "Rodal", "Natario", "VanDenBroeck"]


def _f(x, nd=4):
    return f"{x:.{nd}f}" if x is not None else "--"


def _endpoint_decimals(width):
    """Decimals that resolve a bracket of this width, two figures into it.

    Endpoints and width share a per-row precision fixed by the bracket. Printing
    them at fixed, different places renders a tight bracket as
    ``-0.632, -0.6311, 0.0``, where the width reads as zero and is not
    recoverable from the endpoints.
    """
    if not math.isfinite(width) or width <= 0.0:
        return 4
    return min(6, max(3, math.ceil(-math.log10(width)) + 2))


def write_table(rows, out_path):
    lines = [
        "% Generated by scripts/run_enclosures.py; do not edit.",
        r"\begin{tabular}{@{}l cc c l@{}}",
        r"  \toprule",
        r"  Metric & $\mathcal{N}_{\rm lower}$ & $\mathcal{N}_{\rm upper}$"
        r" & $\mathcal{N}_{\rm lower}/\mathcal{N}_{\rm upper}$"
        r" & independently reported value \\",
        r"  \midrule",
    ]
    for name in ORDER:
        r = rows.get(name)
        if r is None:
            continue
        label = name.replace("VanDenBroeck", "Van den Broeck").replace("Natario", "Nat\\'ario")
        # Taken from the row, not re-looked-up, so --table-only rewrites exactly
        # what the search recorded.
        ref_val, ref_stat = r.get("reference"), r.get("reference_statistic")
        ref_cell = f"${_f(ref_val, 3)}$ ({ref_stat})" if ref_val is not None else "--"
        if not math.isfinite(r["lower"]):
            # Report non-closure explicitly rather than printing a bare -inf: a
            # bound that did not close is a coverage statement, not a number.
            lines.append(
                f"  {label} & \\multicolumn{{3}}{{c}}{{did not close at the box "
                f"budget}} & {ref_cell} \\\\"
            )
            continue
        nd = _endpoint_decimals(r["width"])
        # The ratio, not the difference: it is scale-free, it is what the text
        # quotes, and it does not invite the reader to subtract two rounded
        # numbers and get a third that disagrees with a printed one.
        ratio = r["lower"] / r["upper"] if r["upper"] != 0.0 else float("inf")
        lines.append(
            f"  {label} & ${_f(r['lower'], nd)}$ & ${_f(r['upper'], nd)}$ & "
            f"${ratio:.5g}$ & {ref_cell} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def certify_one(job):
    """Enclose one construction. Module-level so it survives process spawning."""
    name, tol, max_boxes = job
    t0 = time.time()
    metric = BUILDERS[name](V_S, R_B, SIGMA)
    shape = shape_interval(R_B, SIGMA)
    build_s = time.time() - t0
    t0 = time.time()
    enc = certify_nec_deficit(
        metric,
        shape,
        x_range=BOX_X,
        s_range=BOX_S,
        tol=tol,
        max_boxes=max_boxes,
    )
    search_s = time.time() - t0
    # The exterior is excluded because it holds no wall point, not because its
    # deficit is large: f depends on position only through r, so one interval
    # evaluation covers every direction, x < -3 included.
    tail_f_lo, tail_f_hi, tail_excluded = tail_bound(metric, shape, TAIL_X, TAIL_S)
    ref = _load_references().get(name)
    return name, {
        "lower": enc.lower,
        "upper": enc.upper,
        "width": enc.width,
        "argmin": list(enc.argmin),
        "n_evaluations": enc.n_evaluations,
        "tail_shape_enclosure": [tail_f_lo, tail_f_hi],
        "tail_excluded": tail_excluded,
        "reference": ref[0] if ref else None,
        "reference_statistic": ref[1] if ref else None,
        "reference_source": ref[2] if ref else None,
        "build_seconds": build_s,
        "search_seconds": search_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--max-boxes", type=int, default=120000)
    ap.add_argument("--metrics", nargs="+", default=ORDER)
    ap.add_argument(
        "--jobs", type=int, default=0, help="worker processes; 0 = one per construction, 1 = serial"
    )
    ap.add_argument(
        "--table-only",
        action="store_true",
        help="rewrite the table from results/enclosures.json; no search",
    )
    args = ap.parse_args()

    # Four hours of branch-and-bound is too much to pay to change a format string.
    if args.table_only:
        src = os.path.join(RESULTS_DIR, "enclosures.json")
        with open(src) as fh:
            payload = json.load(fh)
        if payload.get("partial", False):
            raise SystemExit(f"{src} is a partial checkpoint; not writing a table")
        write_table(payload["results"], os.path.join(TABLES_DIR, "enclosures.tex"))
        return payload["results"]

    print("=" * 72)
    print(f"CERTIFIED GLOBAL ENCLOSURES  (R_b={R_B}, sigma={SIGMA}, v_s={V_S})")
    print("=" * 72)

    jobs = [(name, args.tol, args.max_boxes) for name in args.metrics]
    n_workers = args.jobs if args.jobs > 0 else len(jobs)

    # Independent constructions, ~1 h each. spawn, not fork: mpmath/JAX state
    # does not survive a fork cleanly.
    if n_workers > 1:
        import concurrent.futures as _cf
        import multiprocessing as _mp

        ctx = _mp.get_context("spawn")
        results = []
        with _cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            for fut in _cf.as_completed([pool.submit(certify_one, j) for j in jobs]):
                results.append(fut.result())
                # Checkpoint on every completion.
                _dump(args, dict(results), partial=True)
    else:
        results = []
        for j in jobs:
            results.append(certify_one(j))
            _dump(args, dict(results), partial=True)

    rows = dict(results)
    for name in ORDER:
        r = rows.get(name)
        if r is None:
            continue
        print(
            f"  {name:12s} [{r['lower']:+.6f}, {r['upper']:+.6f}]  "
            f"width={r['width']:.2e}  evals={r['n_evaluations']}  "
            f"build={r['build_seconds']:.1f}s  search={r['search_seconds']:.1f}s"
        )
        print(
            f"               argmin=({r['argmin'][0]:+.4f}, {r['argmin'][1]:+.4f})  "
            f"exterior f in [{r['tail_shape_enclosure'][0]:.2e}, "
            f"{r['tail_shape_enclosure'][1]:.2e}]  "
            f"tail_excluded={r['tail_excluded']}  reference={r['reference']}",
            flush=True,
        )

    _dump(args, rows, partial=False)
    return rows


def _dump(args, rows, *, partial):
    out = {
        "partial": partial,
        "params": {
            "v_s": V_S,
            "R_b": R_B,
            "sigma": SIGMA,
            "box_x": list(BOX_X),
            "box_s": list(BOX_S),
            "tail_x": list(TAIL_X),
            "tail_s": list(TAIL_S),
            "tol": args.tol,
            "max_boxes": args.max_boxes,
            "wall_mask": [0.1, 0.9],
            "note": (
                "N = min_{|v|=1} (rho_n + 2 b.v + v^T S v), the Eulerian-normalized "
                "null deficit; defined at every point regardless of Hawking-Ellis type. "
                "Lower end certified on the continuum by interval branch-and-bound."
            ),
        },
        "order": [n for n in ORDER if n in rows],
        "results": rows,
    }
    dump_json(out, os.path.join(RESULTS_DIR, "enclosures.json"))
    print(f"\nWrote {os.path.join(RESULTS_DIR, 'enclosures.json')}")
    if all(n in rows for n in ORDER):
        write_table(rows, os.path.join(TABLES_DIR, "enclosures.tex"))


if __name__ == "__main__":
    main()
