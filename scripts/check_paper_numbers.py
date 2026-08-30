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
import collections
import json
import pathlib
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
    "interval_lmi_census.json",
    # Declared by a shipped table's provenance header, so a stale one is a stale
    # published number even though no check below reads it.
    "anec/retained_symplectic.json",
    "anec/retained.json",
    "quantum/ford_roman.json",
    "closing_speed.json",
    "interval_lmi_spotcheck.json",
    "clustered_convergence_alcubierre.json",
    "vorticity_type_analytic.json",
    "wall_resolution.json",
    # Read by a check below, or generating a shipped table, but omitted here until an
    # audit pointed out that a missing one takes its check down with it silently,
    # which is the failure this list exists to prevent.
    "rodal_dec_diagnosis.json",
    "comparison_table.json",
    "nstarts_ablation.json",
    "c1_vs_c2_comparison.json",
    "convergence_data.json",
    "integrated_negative_energy.json",
    "rodal_native_resolution.json",
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
            r"reach(?:ing|es) \$([\d.]+)\\%\$\s*\n\(Alcubierre\), \$([\d.]+)\\%\$ "
            r"\(Nat\\'ario\), and \$([\d.]+)\\%\$ \(Van~den~Broeck\)",
        ))
        checks.append((
            "Section 3.1 Rodal margin at v_s=2.5 vs velocity_sweep.json",
            (f"{rows[('Rodal', 2.5)]['typeI_nec_min']:.3f}",),
            r"to \$(-[\d.]+)\$\s*\nat \$v_s=2\.5\$",
        ))
        # The grid the sweep actually ran on must be the grid the captions claim.
        n_run = sweep["config"]["N"]
        # The caption wording moved from "velocity sweep ($N=...$)" to
        # "benchmark grid ($N=...$)" in an editorial pass, and the old pattern then
        # matched nothing and checked nothing. Match the grid claim itself.
        for caption_n in re.findall(r"benchmark grid\s*\(\$N=(\d+)\$", tex):
            if int(caption_n) != n_run:
                failures_pre.append(
                    "a caption attributes the velocity sweep to $N=" + caption_n
                    + f"$ but velocity_sweep.json ran at N={n_run}"
                )

    # The geodesic ANEC ordering is an argument, not just a number: it reversed this
    # round when the affine window was measured instead of assumed. Pin the ordering
    # and the four values together, so a future window change cannot leave the prose
    # asserting a ranking the data no longer supports.
    anec = artifact("anec/retained_symplectic.json")
    if anec is not None:
        m = anec["metrics"]
        checks.append((
            "Section 3.4 geodesic ANEC minima vs anec/retained_symplectic.json",
            (f"{m['Natário']['min_line_integral']:.2f}",
             f"{m['Alcubierre']['min_line_integral']:.2f}",
             f"{m['Van den Broeck']['min_line_integral']:.3f}",
             f"{m['Rodal']['min_line_integral']:.4f}"),
            r"the impact-parameter scan is \$(-[\d.]+)\$ \(Nat\\'ario\), \$(-[\d.]+)\$ \(Alcubierre\),\s*\n?"
            r"\$(-[\d.]+)\$ \(Van~den~Broeck\), and \$(-[\d.]+)\$ \(Rodal\)",
        ))

    # The coordinate-ray minima of Section 3.4 are a SECOND list, from a different
    # artifact, and they were not pinned. Rodal's drifted from -0.0134 to -0.0136 when
    # the quadrature was fixed, which turns -0.013 into -0.014 at the two significant
    # figures its companions carry, and the prose kept the old digit.
    ray = artifact("anec/retained.json")
    if ray is not None:
        r = ray["metrics"]
        checks.append((
            "Section 3.4 coordinate-ray ANEC minima vs anec/retained.json",
            (f"{r['Alcubierre']['min_line_integral']:.2f}",
             f"{r['Van den Broeck']['min_line_integral']:.3f}",
             f"{r['Rodal']['min_line_integral']:.3f}",
             f"{r['Natário']['min_line_integral']:.4f}"),
            r"the minimum over \$b\$ is\s*\n?"
            r"\$(-[\d.]+)\$ \(Alcubierre, \$b\\!\\approx\\!0\.86\$\), \$(-[\d.]+)\$ \(Van~den~Broeck,\s*\n?"
            r"\$b\\!\\approx\\!0\.82\$\), \$(-[\d.]+)\$ \(Rodal, \$b\\!\\approx\\!1\.27\$\), and \$(-[\d.]+)\$",
        ))
        # Appendix F reads the enclosures as ratios rather than widths, so pin the
        # ratios the prose quotes. The Alcubierre one is the whole A2 claim: a
        # certified bound that agrees with the achieved value to four digits.
        enc = artifact("enclosures.json")
        if enc is not None:
            e = enc["results"]

            def ratio(name: str, sig: int) -> str:
                r = e[name]
                return f"{r['lower'] / r['upper']:.{sig}g}"

            checks.append((
                "Appendix F enclosure ratios vs enclosures.json",
                (f"{e['Alcubierre']['width']:.1e}".replace("e-04", r"\times10^{-4}"),
                 ratio("Alcubierre", 5), ratio("VanDenBroeck", 3),
                 ratio("Natario", 2)),
                r"bracketed to a width of \$(.+?)\$,[\s\S]{0,200}?"
                r"a factor of \$([\d.]+)\$ apart"
                r"[\s\S]{0,140}?within a factor of \$([\d.]+)\$"
                r"[\s\S]{0,300}?finite but \$([\d.]+)\$ times its own achieved value",
            ))

        order = sorted(m, key=lambda k: m[k]["min_line_integral"])
        if order[0] != "Natário":
            failures_pre.append(
                "the deepest geodesic ANEC minimum is now "
                f"{order[0]}, but Section 3.4 names Natário"
            )

    # The Rodal single-frame miss rates are the paper's main observer-dependence
    # number and are quoted in five places, none of them a table. Pin every one.
    inv = artifact("invariant_verification.json")
    if inv is not None:
        rod = next(r for r in inv["rows"] if r["metric"] == "Rodal")
        wec, dec = f"{rod['miss_wec_pct']:.0f}", f"{rod['miss_dec_pct']:.0f}"
        vdb = next(r for r in inv["rows"] if r["metric"].startswith("Van"))
        vdb_wec = f"{vdb['miss_wec_pct']:.0f}"
        vdb_dec = f"{vdb['miss_dec_pct']:.0f}"
        # These patterns run against the wrapped LaTeX source, so every literal
        # space is relaxed to \s+ below. An editorial pass reflows paragraphs, and a
        # pattern that breaks on a moved line break reports a defect that is not one
        # while saying nothing about the number it is supposed to pin.
        for desc, pattern, expected in (
            ("abstract", r"reading of Rodal misses about \$(\d+)\\%\$ of its wall weak-energy",
             (wec,)),
            ("Section 1", r"register \$\{\\approx\}(\d+)\\%\$ of the dominant and "
                          r"\$\{\\approx\}(\d+)\\%\$ of the weak energy-condition", (dec, wec)),
            ("Section 2", r"at \$\{\\approx\}(\d+)\\%\$ of the wall points where a boosted "
                          r"observer sees a\s*\n?weak-energy violation, and at "
                          r"\$\{\\approx\}(\d+)\\%\$ for the dominant energy", (wec, dec)),
            ("Discussion, Van den Broeck wall miss",
             r"Van~den~Broeck is intermediate \(WEC \$(\d+)\\%\$,\s*\n?"
             r"DEC \$(\d+)\\%\$ wall miss", (vdb_wec, vdb_dec)),
            ("Appendix C cross-reference",
             r"matched-parameter benchmark \(\$\{\\approx\}(\d+)\\%\$ DEC, "
             r"\$\{\\approx\}(\d+)\\%\$ WEC;", (dec, wec)),
            # The Conclusion used to quote both fractions again, six lines after the
            # Discussion does, and the Discussion quoted them a third time. Both
            # restatements were cut in the final editorial passes, the Discussion now
            # saying "the majority" and pointing at Section 3.1 for the values. So
            # there is nothing left to pin in either place; the pair is still pinned
            # at Section 1, Section 2 and the Appendix C cross-reference above, which
            # is where the numbers are actually printed.
        ):
            # Literal spaces match any run of whitespace, newlines included.
            pattern = re.sub(r"(?<!\\s)(?<!\\n) ", r"\\s+", pattern)
            checks.append((
                f"Rodal single-frame miss rates ({desc}) vs invariant_verification.json",
                expected, pattern,
            ))

    # The Garattini wall is not uniformly labelled Type I. The balance is Type II and
    # the manuscript now says so; if the fraction moves, the prose must move with it.
    cv = artifact("construction_verification.json")
    if cv is not None:
        def type_ii_pct(block: str) -> str:
            row = cv[block]["Garattini"][-1]
            return f"{100 * (1.0 - row['frac_type_i'] - row['frac_type_iv']):.1f}"
        checks.append((
            "Garattini Type-II wall balance vs construction_verification.json",
            (type_ii_pct("matched"), type_ii_pct("native")),
            r"returns Type~II on \$([\d.]+)\\%\$ of the matched wall volume and "
            r"\$([\d.]+)\\%\$ of the native",
        ))

    exo_json = artifact("exoticity_ranking.json")
    if exo_json is not None:
        axes = exo_json["raw_axes"]
        ratio = axes["Natário"]["nec_severity"] / axes["Alcubierre"]["nec_severity"]
        checks.append((
            "Table 15 caption uncapped NEC severity ratio vs exoticity_ranking.json",
            (f"{ratio:.0f}",),
            # The caption says "~13x" rather than "13x", because the ratio is not an
            # integer and the caption no longer restates the two tabulated values it
            # is formed from. Allow the tilde; the digits are still pinned.
            r"severity is \$(?:\{\\sim\})?(\d+)\\times\$ the Alcubierre\s*\n?\s*baseline",
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
        # The worst single-point departure from each fitted power law. Quoted in
        # Section 3.6 so the log-fit R^2 is not the only thing a reader sees.
        def sci(x: float, sig: int) -> str:
            mant, exp = f"{x:.{sig}e}".split("e")
            mant = mant.rstrip("0").rstrip(".")
            return rf"{mant}\times10^{{{int(exp)}}}"

        def worst(metric: str) -> float:
            return max(f["max_rel_dev"] for f in curv["fits"][metric].values())

        checks.append((
            "Section 3.6 worst power-law deviations vs curvature_scaling.json",
            (sci(curv["fits"]["Alcubierre"]["weyl_squared"]["max_rel_dev"], 1),
             sci(worst("Natário"), 0),
             sci(worst("Rodal"), 0),
             f"{curv['fits']['Alcubierre']['ricci_squared']['max_rel_dev']:.2f}"),
            r"is\s*\n?\$(.+?)\$ \(Alcubierre, Weyl\), \$(.+?)\$ \(Nat\\'ario, all three\) and\s*\n?"
            r"\$(.+?)\$ \(Rodal, all three\), rising to \$([\d.]+)\$ on the one branch",
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
            "Appendix H Type-III branch: every point mislabelled, every point certified",
            (str(t3["n"]), str(n_iv_at_tight), str(
                sum(1 for r in t3["rows"]
                    if r["nec_margin"] < -r["noise_floor"]))),
            r"family, \$(\d+)\$ points log-spaced over[\s\S]{0,400}?"
            r"Type~IV at every one of the\s*\n?\$(\d+)\$[\s\S]{0,600}?"
            r"certifies the null-energy violation at all \$(\d+)\$",
        ))
        # The tolerance sweep is the sharpest number in the appendix and the one most
        # likely to drift, since it is the only place the label is quoted as a
        # function of a knob rather than as a verdict.
        def tol_counts(tol: str) -> tuple[str, str, str]:
            c = collections.Counter(r["labels"][tol] for r in t3["rows"])
            return (str(c[2]), str(c[3]), str(c[4]))

        checks.append((
            "Appendix H Type-III tolerance sweep vs type_transition_audit.json",
            tol_counts("tol_2e-06") + tol_counts("tol_5e-06"),
            r"Type~II at \$(\d+)\$, Type~III at \$(\d+)\$ and Type~IV at \$(\d+)\$, "
            r"and at \$5\\times10\^\{-6\}\$ it\s*\n?returns \$(\d+)\$, \$(\d+)\$ and \$(\d+)\$",
        ))

    # tab:rodal_ablation is the last table in the manuscript whose numbers are typed
    # into an inline tabular rather than \input from a generated file. Its values do
    # exist in the artifact, so the only thing missing was a check; a table that is
    # right today and unpinned is the exact shape of the defect the report caught.
    rodal_abl = artifact("rodal_dec_diagnosis.json")
    if rodal_abl is not None:
        # The sweep is stored as two parallel lists, not as a mapping keyed by N.
        res = rodal_abl.get("sweeps", {}).get("resolution", {})
        by_n = dict(zip(res.get("values", []), res.get("rodal_dec_miss_pct", [])))
        want = [f"{by_n[n]:.2f}" for n in (25, 50, 100)
                if isinstance(by_n.get(n), (int, float))]
        if len(want) == 3:
            checks.append((
                "Table rodal_ablation resolution rows vs rodal_dec_diagnosis.json",
                tuple(want),
                r"& \$25\$\s*& ([\d.]+) & 0\.0 \\\\\s*\n\s*& \$50\$\s*& ([\d.]+) "
                r"& 0\.0 \\\\\s*\n\s*& \$100\$\s*& ([\d.]+) & 0\.0",
            ))
        else:
            failures_pre.append(
                "rodal_dec_diagnosis.json no longer exposes the three resolution "
                "rows that tab:rodal_ablation types by hand; re-point this check"
            )

    failures: list[str] = list(failures_pre)
    for desc, expected, pattern in checks:
        # Every literal space is relaxed to \s+ before matching. These patterns run
        # against the wrapped LaTeX source, so an editorial pass that reflows a
        # paragraph moves a line break into the middle of a pinned phrase and the
        # check reports a defect that is not one, saying nothing about the number it
        # exists to pin. The Rodal miss-rate patterns already did this locally; doing
        # it here covers every check instead of one.
        pattern = re.sub(r"(?<!\\s)(?<!\\n) ", r"\\s+", pattern)
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

    # The wall-cell ladder must be quoted with one range everywhere it appears, and the
    # range has to come from the artifact rather than from three constants typed here.
    # Pinning prose against a hard-coded triple lets a stale paper and a stale gate
    # agree with each other while the data says something else.
    diag = artifact("diagnostic_convergence.json")
    if diag is not None:
        cells = diag["wall_info"]["Alcubierre"]
        rungs = [f"{cells[str(n)]['cells']:.1f}" for n in diag["ladder_N"]]
        if not re.search(
            r"\$" + r"\$, \$".join(re.escape(c) for c in rungs) + r"\$ cells", tex
        ):
            failures.append(
                f"wall-cell ladder {'/'.join(rungs)} (diagnostic_convergence.json) is "
                f"not the ladder main.tex states"
            )
        ranges = set(re.findall(r"wall by \$([\d.]+)\$ to\s*\n?\$?([\d.]+)\$? cells", tex))
        for lo, hi in ranges:
            if (lo, hi) != (rungs[0], rungs[-1]):
                failures.append(
                    f"wall-cell range ${lo}$ to ${hi}$ contradicts the "
                    f"{'/'.join(rungs)} ladder"
                )

    # The census and the branch-and-bound both bound the same infimum from above, by
    # disjoint routes. Appendix H.2 once claimed the census endpoint had to sit at or
    # BELOW the search's achieved upper end; it sits above it, for all four drives, and
    # must, since a 33x129 sample cannot beat a 120,000-box search. Pin the direction
    # that is actually true, so the sentence cannot drift back.
    cen, enc2 = artifact("interval_lmi_census.json"), artifact("enclosures.json")
    if cen is not None and enc2 is not None:
        alias = {"Alcubierre": "Alcubierre", "Natário": "Natario",
                 "Van den Broeck": "VanDenBroeck", "Rodal": "Rodal"}
        for drive, row in cen["results"].items():
            nec = row["conditions"]["nec"] if "conditions" in row else row
            attained, br = nec["deepest_upper"], enc2["results"][alias[drive]]
            if not br["lower"] <= attained:
                failures.append(
                    f"census endpoint {attained:.6f} for {drive} is below the certified "
                    f"lower end {br['lower']:.6f}; one of the two is wrong"
                )
            if attained < br["upper"]:
                failures.append(
                    f"census endpoint {attained:.6f} for {drive} is deeper than the "
                    f"branch-and-bound achieved upper end {br['upper']:.6f}; the sample "
                    f"cannot beat the search, so re-read Appendix H.2"
                )

    # The hand-maintained macros in paper_numbers.tex are quoted numbers that no
    # table carries, and nothing read this file at all: emit_paper_numbers.py rewrites
    # the auto-sourced block and leaves these five alone, so they were the only figures
    # in the submission with no check between them and the artifact they cite.
    pn_path = args.paper / "paper_numbers.tex"
    wall_restricted = artifact("wall_restricted_analysis.json")
    if pn_path.exists() and wall_restricted is not None:
        pn = pn_path.read_text(encoding="utf-8")
        grids = wall_restricted["metrics"]
        for macro, metric, key in (
            ("alcubierreNECmissVSfive", "alcubierre", "nec_pct_missed"),
            ("alcubierreWECmissVSfive", "alcubierre", "wec_pct_missed"),
            ("rodalNECmissVSfive", "rodal", "nec_pct_missed"),
            ("rodalWECmissVSfive", "rodal", "wec_pct_missed"),
            ("rodalDECmissVSfive", "rodal", "dec_pct_missed"),
        ):
            m = re.search(r"\\newcommand\{\\" + macro + r"\}\{([\d.]+)\}", pn)
            want = f"{grids[metric]['full_grid'][key]:.1f}"
            if m is None:
                failures.append(f"paper_numbers.tex no longer defines \\{macro}")
            elif m.group(1) != want:
                failures.append(
                    f"\\{macro} is {m.group(1)}, wall_restricted_analysis.json says "
                    f"{want}"
                )

    # Rodal's DEC miss on the N=120 level must be the value the tables agree on.
    if rodal_dec_conv != rodal_dec_missed:
        failures.append(
            "convergence_per_metric N=120 and missed_wall_restricted disagree for "
            f"Rodal DEC: {rodal_dec_conv} vs {rodal_dec_missed}"
        )

    # Provenance, not arithmetic: no artifact feeding a table may predate the code
    # that produced it. Three fixes landed after most tables had been generated,
    # the near-vacuum gate (which moves Type-I *labels*), the Type-I SEC trace
    # inequality (which moves Type-I *margins*) and the LMI noise floor, and the
    # resulting mixture put two fits of the same law in the manuscript with
    # different coefficients. A file mtime is a weak check, but it is the one that
    # would have caught that.
    #
    # The watch was src/warpax/energy_conditions/ only, which is where the labels
    # and margins live but not where the quadrature weights, the fits or the
    # generators do: a proper-volume fix in src/warpax/grids/ or a polished
    # extremum in scripts/ would have left every artifact "fresh". Watch the whole
    # library and every generator, excluding this file so that editing the gate
    # does not condemn the data it is checking.
    src_dir = RESULTS.parent / "src" / "warpax"
    scripts_dir = pathlib.Path(__file__).resolve().parent
    if src_dir.is_dir():
        watched = list(src_dir.rglob("*.py")) + [
            q for q in scripts_dir.glob("*.py")
            if q.name != pathlib.Path(__file__).name
        ]
        newest_code = max((q.stat().st_mtime for q in watched), default=0.0)
        for name in sorted(_TABLE_ARTIFACTS):
            path = RESULTS / name
            # A MISSING artifact is a failure, not a skip. Every JSON-pinned check in
            # this file already returns early when its input is absent, so on a wiped
            # results/ the gate reported "all checks passed" while checking almost
            # nothing, which is how a broken pipeline survived a round of review.
            # The gate has to be loudest exactly when there is no data.
            if not path.exists():
                failures.append(
                    f"results/{name} is missing; every check that reads it was "
                    f"silently skipped. Run reproduce_all.sh before quoting a number"
                )
                continue
            if path.stat().st_mtime < newest_code:
                failures.append(
                    f"results/{name} predates the newest file in src/warpax/ "
                    f"or scripts/; regenerate before quoting it"
                )
            # A checkpointed artifact is not a finished one.
            try:
                import json as _json

                if _json.loads(path.read_text()).get("partial") is True:
                    failures.append(
                        f"results/{name} is a partial checkpoint; the run that "
                        f"writes it has not finished"
                    )
            except (ValueError, OSError, AttributeError):
                pass

        # The same rule for the cached grids. run_analysis.py skips any .npz that
        # already exists, existence only, no mtime and no hash, so a re-run after a
        # source edit rebuilds every JSON from grids computed before it, and each JSON
        # then carries a fresh mtime and passes the loop above. The staleness was one
        # layer below where anyone was looking. reproduce_all.sh deletes the grids
        # unless --keep-cache is passed, so a clean full run satisfies this; a partial
        # one no longer looks like one.
        stale_npz = sorted(
            q.name for q in RESULTS.glob("*.npz") if q.stat().st_mtime < newest_code
        )
        if stale_npz:
            failures.append(
                f"{len(stale_npz)} cached grid(s) in results/ predate the newest file "
                f"in src/warpax/ or scripts/, starting with {stale_npz[0]}; every JSON "
                f"rebuilt from them inherits pre-edit inputs. Re-run reproduce_all.sh "
                f"WITHOUT --keep-cache"
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
        for tex_path in sorted(tables_dir.glob("*.tex")):
            head = tex_path.read_text(errors="replace").lstrip().split("\n", 1)[0]
            if not head.startswith(("% Generated by", "% Hand-written")):
                failures.append(
                    f"tables/{tex_path.name} has no provenance header; emit it through "
                    f"_json_io.write_table, or mark it '% Hand-written: <why>'"
                )

    # Both ANEC tables sit side by side in the paper, so they must share a window
    # rule. Table 11's columns were generated on the superseded "double until the
    # on-axis integral is stationary" rule long after the geodesic run abandoned it.
    try:
        for fn in ("anec/retained.json", "anec/retained_symplectic.json"):
            note = artifact(fn)["params"]["affine_span_note"]
            if "stationary" in note or "r_s = 3" not in note:
                failures.append(
                    f"{fn}: affine window is not the measured crossing-span rule "
                    f"(note: {note[:60]}...)"
                )
    except Exception as exc:                      # pragma: no cover - defensive
        failures.append(f"ANEC window-rule check could not run: {exc}")

    # The manuscript quotes both integrator witnesses; neither may be asserted.
    try:
        sym = artifact("anec/retained_symplectic.json")["metrics"]
        for name, m in sym.items():
            if not m.get("all_null_preserved"):
                failures.append(f"ANEC {name}: not every ray is symplectically certified")
            drift = m.get("worst_killing_energy_drift")
            if drift is None:
                failures.append(f"ANEC {name}: Killing-energy witness not computed")
            elif drift >= 1e-5:
                failures.append(
                    f"ANEC {name}: Killing drift {drift:.2e} exceeds the quoted 1e-5"
                )
    except Exception as exc:                      # pragma: no cover - defensive
        failures.append(f"ANEC witness check could not run: {exc}")

    # The momentum-channel fraction is claimed as a LOWER bound on the wall Type-IV
    # fraction for flat-slice drives. A negative gap there falsifies it.
    try:
        for row in artifact("closing_speed.json")["rows"]:
            if row["flat_slice_premise_holds"] and row["min_gap_pp"] < -1e-9:
                failures.append(
                    f"closing speed {row['metric']}: momentum-channel fraction "
                    f"exceeds the measured Type-IV fraction by "
                    f"{-row['min_gap_pp']:.2f} pp"
                )
    except Exception as exc:                      # pragma: no cover - defensive
        failures.append(f"closing-speed bound check could not run: {exc}")

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
    # The non-`checks` gates, counted by name rather than by a constant. The previous
    # version claimed to be computed and was a literal 6, against a comment listing ten
    # and a body containing more than that.
    extra_gates = (
        "caption-N pin", "ANEC ordering", "exoticity/velocity cross-table",
        "wall-cell ladder", "Rodal DEC convergence cross-table",
        "artifact existence/staleness/partial", "cached-grid staleness",
        "table provenance headers", "ANEC window rule", "ANEC witnesses",
        "closing-speed bound", "census vs enclosure ordering",
        "paper_numbers.tex macros", "MANIFEST",
    )
    print(f"all {len(checks) + len(extra_gates)} paper-number checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
