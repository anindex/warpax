"""Emit the diagnostic/ablation LaTeX tables from cached results JSONs.

Regenerates the table bodies that previously lived inline in the manuscript,
so every printed number traces to results/*.json and cannot drift:

- tables/missed_uniform.tex        <- results/comparison_table.json (v_s = 0.5)
- tables/type_breakdown.tex        <- results/wall_restricted_analysis.json
- tables/nstarts.tex               <- results/nstarts_ablation.json
- tables/c1_vs_c2.tex              <- results/c1_vs_c2_comparison.json
- tables/convergence_richardson.tex <- results/convergence_data.json

Read-only on results/: reruns of the upstream analysis scripts refresh the
JSONs, then this script refreshes the tables.

Usage
-----
    python scripts/emit_diagnostic_tables.py
"""
from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
TABLES = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

DISPLAY = {
    "schwarzschild": "Schwarzschild",
    "alcubierre": "Alcubierre",
    "vdb": "Van~Den~Broeck",
    "natario": "Nat\\'ario",
    "rodal": "Rodal",
    "warpshell": "WarpShell$^{\\ddagger}$",
}

MISSED_ORDER = ["schwarzschild", "alcubierre", "vdb", "natario", "rodal", "warpshell"]
BREAKDOWN_ORDER = ["alcubierre", "natario", "vdb", "rodal", "warpshell"]


def _load(rel):
    with open(os.path.join(RESULTS, rel)) as f:
        return json.load(f)


def _sci(x: float, digits: int = 1) -> str:
    """Format as $a.b\\times10^{n}$ (or plain decimal for small exponents)."""
    if x == 0.0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(x))))
    if -3 <= exp <= 3:
        return f"${x:.{digits}f}$" if abs(x) >= 1 else f"${x:.{max(digits, -exp + 1)}f}$"
    mant = x / 10**exp
    return f"${mant:.{digits}f}\\times10^{{{exp}}}$"


def _pct(x: float) -> str:
    return f"{x:.1f}"


def _missed_cell(x: float) -> str:
    s = f"{x:.1f}"
    return f"\\textbf{{{s}}}" if x >= 1.0 else s


def emit_missed_uniform() -> None:
    rows = _load("comparison_table.json")
    sel = {}
    for r in rows:
        m, vs = r["metric"], r["v_s"]
        if m == "schwarzschild" or vs == 0.5:
            sel[m] = r
    lines = [
        "\\begin{tabular}{@{}l cccc cccc@{}}",
        "  \\toprule",
        "  & \\multicolumn{4}{c}{Total violated (\\%)}",
        "  & \\multicolumn{4}{c}{Missed by Eulerian (\\%)} \\\\",
        "  \\cmidrule(lr){2-5} \\cmidrule(lr){6-9}",
        "  Metric & NEC & WEC & SEC & DEC",
        "         & NEC & WEC & SEC & DEC \\\\",
        "  \\midrule",
    ]
    for m in MISSED_ORDER:
        r = sel[m]
        tot = [_pct(r[f"{c}_pct_violated_robust"]) for c in ("nec", "wec", "sec", "dec")]
        mis = [_missed_cell(r[f"{c}_pct_missed"]) for c in ("nec", "wec", "sec", "dec")]
        lines.append(f"  {DISPLAY[m]} & {' & '.join(tot)}")
        lines.append(f"                 & {' & '.join(mis)} \\\\")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    _write("missed_uniform.tex", lines)


def emit_type_breakdown() -> None:
    data = _load("wall_restricted_analysis.json")["metrics"]
    lines = [
        "\\begin{tabular}{@{}l r r r r r r r@{}}",
        "    \\toprule",
        "    Metric & \\% Type~I & \\% Type~II & \\% Type~III & \\% Type~IV"
        " & Wall \\% Type~I & Wall \\% Type~IV & max $|\\mathrm{Im}\\,\\lambda|$ \\\\",
        "    \\midrule",
    ]
    for m in BREAKDOWN_ORDER:
        fg = data[m]["full_grid"]
        wr = data[m]["wall_restricted"]
        cells = [
            _pct(100 * fg["frac_type_i"]),
            _pct(100 * fg["frac_type_ii"]),
            _pct(100 * fg["frac_type_iii"]),
            _pct(100 * fg["frac_type_iv"]),
            _pct(100 * wr["frac_type_i"]),
            _pct(100 * wr["frac_type_iv"]),
            _sci(fg["max_imag_eigenvalue"]),
        ]
        lines.append(f"    {DISPLAY[m]} & {' & '.join(cells)} \\\\")
    lines += ["    \\bottomrule", "\\end{tabular}"]
    _write("type_breakdown.tex", lines)


def emit_nstarts() -> None:
    data = _load("nstarts_ablation.json")
    order = ["alcubierre", "rodal", "warpshell"]
    names = {"alcubierre": "Alcubierre", "rodal": "Rodal", "warpshell": "WarpShell"}
    ns = data["alcubierre"]["n_starts_values"]
    lines = [
        "\\begin{tabular}{@{}l rrrrr@{}}",
        "    \\toprule",
        f"    & \\multicolumn{{{len(ns)}}}{{c}}{{$N_{{\\text{{starts}}}}$}} \\\\",
        f"    \\cmidrule(l){{2-{len(ns) + 1}}}",
        "    Metric & " + " & ".join(str(n) for n in ns) + " \\\\",
        "    \\midrule",
        f"    \\multicolumn{{{len(ns) + 1}}}{{@{{}}l}}"
        "{\\textit{Min robust WEC margin}$^{\\dagger}$} \\\\[2pt]",
    ]
    for m in order:
        vals = data[m]["min_wec_margin"]
        cells = " & ".join(_sci(v) for v in vals)
        lines.append(f"    {names[m]} & {cells} \\\\")
    lines.append("    \\midrule")
    lines.append(
        f"    \\multicolumn{{{len(ns) + 1}}}{{@{{}}l}}"
        "{\\textit{Missed WEC (\\%)}} \\\\[2pt]"
    )
    for m in order:
        vals = data[m]["pct_missed_wec"]
        cells = " & ".join(f"{v:.2f}" for v in vals)
        lines.append(f"    {names[m]} & {cells} \\\\")
    lines += ["    \\bottomrule", "\\end{tabular}"]
    _write("nstarts.tex", lines)


def emit_c1_vs_c2() -> None:
    rows = _load("c1_vs_c2_comparison.json")["rows"]
    lines = [
        "\\begin{tabular}{c cc cc cc cc}",
        "\\hline\\hline",
        "$v_s$ & \\multicolumn{2}{c}{Type I (\\%)} & \\multicolumn{2}{c}{Type IV (\\%)}"
        " & \\multicolumn{2}{c}{$\\min\\;m_{\\mathrm{NEC}}$}"
        " & \\multicolumn{2}{c}{$\\max|d^3\\alpha/dx^3|$} \\\\",
        " & C1 & C2 & C1 & C2 & C1 & C2 & C1 & C2 \\\\",
        "\\hline",
    ]
    for r in rows:
        cells = [
            f"{r['v_s']}",
            _pct(r["c1_pct_type_i"]),
            _pct(r["c2_pct_type_i"]),
            _pct(r["c1_pct_type_iv"]),
            _pct(r["c2_pct_type_iv"]),
            _sci(r["c1_min_nec_robust"], 2),
            _sci(r["c2_min_nec_robust"], 2),
            _sci(r["c1_max_d3_lapse"], 2),
            _sci(r["c2_max_d3_lapse"], 2),
        ]
        lines.append("  " + " & ".join(cells) + " \\\\")
    lines += ["\\hline\\hline", "\\end{tabular}"]
    _write("c1_vs_c2.tex", lines)


def emit_convergence() -> None:
    c = _load("convergence_data.json")
    res = c["resolutions"]

    def _is_fallback(q):
        if "fallback" in q:
            return bool(q["fallback"])
        v = q["values"]
        return (v[0] - v[1]) * (v[1] - v[2]) <= 0  # non-monotone triplet

    def _row(label, key, fmt):
        q = c[key]
        vals = " & ".join(f"${fmt(v)}$" for v in q["values"])
        if _is_fallback(q):
            p_cell = "$(2)^{\\dagger}$"
        else:
            p_cell = f"${q['observed_order']:.1f}$"
        if q["converged"]:
            ext = f"${fmt(q['extrapolated_value'])}$"
            err = _sci(q["error_estimate"])
        else:
            ext, err = "--", "--"
        return f"    {label}\n      & {vals}\n      & {ext} & {p_cell} & {err} \\\\"

    header_cols = " & ".join(f"$N\\!=\\!{n}$" for n in res)
    lines = [
        "\\begin{tabular}{@{}lcccccc@{}}",
        "    \\toprule",
        f"    Quantity & {header_cols}",
        "      & Extrap.\\ & $p$ & Error est.\\ \\\\",
        "    \\midrule",
        _row("Min margin NEC", "min_margin_nec", lambda v: f"{v:.3f}"),
        _row("Integrated viol.\\", "integrated_violation_nec", lambda v: f"{v:.3f}"),
        _row("$L^2$ viol.\\ norm", "l2_violation_nec", lambda v: f"{v:.2f}"),
        "    \\bottomrule",
        "\\end{tabular}",
    ]
    _write("convergence_richardson.tex", lines)


def _write(name: str, lines: list[str]) -> None:
    path = os.path.join(TABLES, name)
    with open(path, "w") as f:
        f.write("% Generated by scripts/emit_diagnostic_tables.py; do not edit.\n")
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {os.path.normpath(path)}")


def main():
    emit_missed_uniform()
    emit_type_breakdown()
    emit_nstarts()
    emit_c1_vs_c2()
    emit_convergence()


if __name__ == "__main__":
    main()
