#!/usr/bin/env python
"""Fail loudly when the manuscript prose disagrees with the generated tables.

Round 2 of review shipped with the same quantity stated two different ways in a
single sentence, and with three prose figures that appear in no table at all.
Every one of those was a value quoted by hand next to a number the pipeline
regenerates. This script pins the quoted copies to the generated ones.

It is deliberately explicit rather than clever: each check names a table cell and
the prose that must agree with it. A generic "find every number in the prose"
scanner would be almost all false positives and would be switched off within a
week. Adding a check is three lines; that is the point.

    python scripts/check_paper_numbers.py [--paper DIR]

Exits non-zero, listing every mismatch, if any check fails.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Artifacts that a table in the manuscript is generated from. Listed explicitly, in
# the same spirit as the checks below: adding one is a line, and a generic scan would
# be mostly false positives.
_TABLE_ARTIFACTS = (
    "velocity_sweep.json",
    "invariant_verification.json",
    "matched_benchmark.json",
    "shift_vorticity.json",
    "exoticity_ranking.json",
    "ssv_bound.json",
    "curvature_scaling.json",
    "construction_verification.json",
    "diagnostic_convergence.json",
    "curvature_convergence.json",
    "extra_convergence.json",
    "rodal_sigma_resolved.json",
    "wall_restricted_analysis.json",
    "classifier_audit.json",
    "enclosures.json",
    "type_transition_audit.json",
    "lmi_audit.json",
)

DEFAULT_PAPER = Path(__file__).resolve().parents[2] / "warpax_arxiv"


def load(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    return path.read_text(encoding="utf-8")


def table_cell(body: str, row: str, col: int) -> str:
    """Cell ``col`` (0-based, after the row label) of the row labelled ``row``."""
    for line in body.splitlines():
        line = line.strip().rstrip(r"\\").strip()
        if not line.startswith(row):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) > col + 1:
            return cells[col + 1]
    raise SystemExit(f"row {row!r} column {col} not found in table")


def table_column(body: str, col: int, rows: list[str]) -> list[str]:
    return [table_cell(body, r, col) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    args = ap.parse_args()

    tex = load(args.paper / "main.tex")
    t = lambda name: load(args.paper / "tables" / f"{name}.tex")  # noqa: E731

    bench = t("invariant_benchmark")
    conv = t("convergence_per_metric")
    missed = t("missed_wall_restricted")
    vort = t("shift_vorticity")
    vel = t("velocity_type_structure")
    exo = t("exoticity_ranking")

    def artifact(name: str):
        path = RESULTS / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    metrics = ["Alcubierre", "Natário", "Van den Broeck"]
    type_i = table_column(bench, 0, metrics)
    type_iv = table_column(bench, 1, metrics)
    rodal_dec_bench = table_cell(bench, "Rodal", 5)
    rodal_dec_missed = table_cell(missed, "Rodal", 4)
    rodal_dec_conv = table_cell(conv, "Rodal", 2)
    vdb_range = table_cell(vort, "Van den Broeck", 3)
    nat_iv_sweep = table_cell(vel, "Natário", 1)
    nat_iv_bench = table_cell(bench, "Natário", 1)
    nat_exo_iv = table_cell(exo, "Natário", 1)

    lo_iv, hi_iv = min(float(v) for v in type_iv), max(float(v) for v in type_iv)
    vdb_lo = vdb_range.split("--")[0]

    failures_pre: list[str] = []

    # Each check: (description, expected group values, regex over main.tex).
    # The regex groups must equal the expected tuple exactly, after stripping
    # whitespace. Ranges in prose are rounded to whole percent, so round rather
    # than truncate when deriving them from the tables.
    checks: list[tuple[str, tuple[str, ...], str]] = [
        (
            "Section 3.1 Type-IV dominance range vs invariant_benchmark",
            (str(round(lo_iv)), str(round(hi_iv))),
            r"Type-IV dominated\}\s*\n\(\$(\d+)\$--\$(\d+)\\%\$ at \$v_s = 0\.5\$\)",
        ),
        (
            "Section 4.3 Type-I fractions vs invariant_benchmark",
            tuple(type_i),
            r"Type-I fractions \$([\d.]+)\\%\$, \$([\d.]+)\\%\$, \$([\d.]+)\\%\$",
        ),
        (
            "Section 3.4 Rodal wall DEC miss range vs the two tables that report it",
            (f"{min(float(rodal_dec_missed), float(rodal_dec_bench)):.1f}",
             f"{max(float(rodal_dec_missed), float(rodal_dec_bench)):.1f}"),
            r"DEC miss \$([\d.]+)\$--\$([\d.]+)\\%\$",
        ),
        (
            "Section 3.2 Van den Broeck low-end wall Type-IV vs shift_vorticity",
            (vdb_lo,),
            r"only \$(\d+)\\%\$ of the Van~den~Broeck wall is Type-IV",
        ),
        (
            "Table 15 caption Natario Type-IV gap vs velocity sweep and benchmark",
            (nat_iv_sweep, nat_iv_bench),
            r"\$([\d.]+)\\%\$ here and \$([\d.]+)\\%\$ in Table~",
        ),
    ]

    # ---- prose numbers pinned directly to the JSON artifacts -----------------
    #
    # The checks above compare prose against generated TABLES. Round 2 shipped a
    # second failure mode the tables cannot catch: prose quoting a number that
    # exists in no table at all, and in several cases in no artifact either
    # (superluminal Type-I fractions, the Rodal minimum at v_s=2.5, a severity
    # ratio, a ray-convergence study the script never ran). These pin the
    # survivors straight to results/*.json.
    sweep = artifact("velocity_sweep.json")
    if sweep is not None:
        rows = {(r["metric"], r["v_s"]): r for r in sweep["rows"]}

        def typeI_pct(metric: str, v: float) -> str:
            return f"{100 * rows[(metric, v)]['wall_frac_type_i']:.1f}"

        checks.append((
            "Section 3.1 superluminal Type-I fractions vs velocity_sweep.json",
            (typeI_pct("Alcubierre", 2.5), typeI_pct("Natário", 2.5),
             typeI_pct("Van den Broeck", 2.5)),
            r"reaching \$([\d.]+)\\%\$\s*\n\(Alcubierre\), \$([\d.]+)\\%\$ "
            r"\(Nat\\'ario\), and \$([\d.]+)\\%\$ \(Van~den~Broeck\)",
        ))
        checks.append((
            "Section 3.1 Rodal margin at v_s=2.5 vs velocity_sweep.json",
            (f"{rows[('Rodal', 2.5)]['typeI_nec_min']:.3f}",),
            r"to \$(-[\d.]+)\$\s*\nat \$v_s=2\.5\$",
        ))
        # The grid the sweep actually ran on must be the grid the captions claim.
        n_run = sweep["config"]["N"]
        for caption_n in re.findall(r"velocity sweep \(\$N=(\d+)\$\)", tex):
            if int(caption_n) != n_run:
                failures_pre.append(
                    "a caption attributes the velocity sweep to $N=" + caption_n
                    + f"$ but velocity_sweep.json ran at N={n_run}"
                )

    exo_json = artifact("exoticity_ranking.json")
    if exo_json is not None:
        axes = exo_json["raw_axes"]
        ratio = axes["Natário"]["nec_severity"] / axes["Alcubierre"]["nec_severity"]
        checks.append((
            "Table 15 caption uncapped NEC severity ratio vs exoticity_ranking.json",
            (f"{ratio:.0f}",),
            r"severity is \$(\d+)\\times\$ the Alcubierre\s*\n?\s*baseline",
        ))

    curv = artifact("curvature_scaling.json")
    if curv is not None:
        fits = curv["fits"]
        a = fits["Alcubierre"]["ricci_squared"]
        r = fits["Rodal"]["ricci_squared"]
        cross = (a["A"] / r["A"]) ** (1.0 / (r["q"] - a["q"]))
        checks.append((
            "Section 3.5 Ricci-axis crossing speed vs curvature_scaling.json",
            (f"{cross:.2f}",),
            r"overtaking the Alcubierre wall at \$v_s=([\d.]+)\$ on the Ricci axis",
        ))

    # Appendix H quotes the type-transition audit in prose rather than only through a
    # table, and those are the numbers that carry the answer to the exhaustiveness
    # objection. Pin them to the artifact.
    tt = artifact("type_transition_audit.json")
    if tt is not None:
        fam = tt["families"]["momentum_aligned"]
        t3 = tt["type_iii_chain"]
        n_iv_at_tight = sum(
            1 for r in t3["rows"] if r["labels"]["tol_1e-10"] == 4)
        checks.append((
            "Appendix H momentum-family sample count vs type_transition_audit.json",
            (str(fam["n"]),),
            r"Across \$(\d+)\$ samples\s+the\s+inequality's margin is Lipschitz",
        ))
        checks.append((
            "Appendix H Type-III arm: every point mislabelled, every point certified",
            (str(t3["n"]), str(n_iv_at_tight), str(
                sum(1 for r in t3["rows"]
                    if r["nec_margin"] < -r["noise_floor"]))),
            r"family, \$(\d+)\$ points log-spaced over[\s\S]{0,400}?"
            r"Type~IV at every one of the\s*\n?\$(\d+)\$[\s\S]{0,600}?"
            r"certifies the null-energy violation at all \$(\d+)\$",
        ))

    failures: list[str] = list(failures_pre)
    for desc, expected, pattern in checks:
        m = re.search(pattern, tex)
        if m is None:
            failures.append(f"{desc}\n    prose pattern not found: {pattern}")
            continue
        found = tuple(g.strip() for g in m.groups())
        if found != tuple(e.strip() for e in expected):
            failures.append(
                f"{desc}\n    tables say {expected}, prose says {found}"
            )

    # The exoticity Type-IV column is the velocity sweep expressed as a fraction.
    if abs(float(nat_exo_iv) * 100.0 - float(nat_iv_sweep)) > 0.05:
        failures.append(
            "exoticity_ranking Type-IV column vs velocity_type_structure\n"
            f"    {nat_exo_iv} (as a fraction) != {nat_iv_sweep}% for Natario"
        )

    # The wall-cell ladder must be quoted with one range everywhere it appears.
    ladder = re.findall(r"\$5\.9\$, \$7\.6\$, \$8\.9\$ cells", tex)
    ranges = set(re.findall(r"wall by \$([\d.]+)\$ to\s*\n?\$?([\d.]+)\$? cells", tex))
    if not ladder:
        failures.append("wall-cell ladder 5.9/7.6/8.9 no longer stated in main.tex")
    for lo, hi in ranges:
        if (lo, hi) != ("5.9", "8.9"):
            failures.append(
                f"wall-cell range ${lo}$ to ${hi}$ contradicts the 5.9/7.6/8.9 ladder"
            )

    # Rodal's DEC miss on the N=120 level must be the value the tables agree on.
    if rodal_dec_conv != rodal_dec_missed:
        failures.append(
            "convergence_per_metric N=120 and missed_wall_restricted disagree for "
            f"Rodal DEC: {rodal_dec_conv} vs {rodal_dec_missed}"
        )

    # Provenance, not arithmetic: no artifact feeding a table may predate the code
    # that produced it. Three fixes landed after most tables had been generated --
    # the near-vacuum gate (which moves Type-I *labels*), the Type-I SEC trace
    # inequality (which moves Type-I *margins*) and the LMI noise floor -- and the
    # resulting mixture put two fits of the same law in the manuscript with
    # different coefficients. A file mtime is a weak check, but it is the one that
    # would have caught that.
    src_dir = RESULTS.parent / "src" / "warpax" / "energy_conditions"
    if src_dir.is_dir():
        newest_code = max((p.stat().st_mtime for p in src_dir.glob("*.py")),
                          default=0.0)
        for name in sorted(_TABLE_ARTIFACTS):
            path = RESULTS / name
            if path.exists() and path.stat().st_mtime < newest_code:
                failures.append(
                    f"results/{name} predates the newest file in "
                    f"src/warpax/energy_conditions/; regenerate before quoting it"
                )

    # Every table must say which script wrote it and which artifact it read. Nine of
    # the twenty-seven did; the rest were indistinguishable from hand-typed values,
    # and "the hardcoded values are entirely fabricated or disconnected from the data"
    # is the charge a table with no provenance invites. tables/wall_resolution.tex was
    # in fact hand-maintained, which is how it came to display a width of 0.27 and a
    # spacing of 0.20 beside a quotient of 1.35: right from the unrounded 0.2747 and
    # 10/49, wrong from the digits shown. It is generated now, and the generated file
    # reproduces every previously published value.
    tables_dir = RESULTS.parent.parent / "warpax_arxiv" / "tables"
    if tables_dir.is_dir():
        for tex in sorted(tables_dir.glob("*.tex")):
            head = tex.read_text(errors="replace").lstrip().split("\n", 1)[0]
            if not head.startswith(("% Generated by", "% Hand-written")):
                failures.append(
                    f"tables/{tex.name} has no provenance header; emit it through "
                    f"_json_io.write_table, or mark it '% Hand-written: <why>'"
                )

    # The cached grids are gitignored, so the manifest is their only integrity
    # record. Keep it in the same gate as the numbers it backs.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import write_manifest

        failures.extend(f"MANIFEST: {p}" for p in write_manifest.check())
    except Exception as exc:                      # pragma: no cover - defensive
        failures.append(f"MANIFEST check could not run: {exc}")

    if failures:
        print(f"{len(failures)} paper-number check(s) FAILED:\n")
        for f in failures:
            print(f"  - {f}\n")
        return 1
    print(f"all {len(checks) + 3} paper-number checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
