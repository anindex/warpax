#!/usr/bin/env python
"""Fail loudly when the manuscript prose disagrees with the generated tables.

Every number quoted by hand is pinned to the generated one it copies. Each check
names a table cell and the prose that must agree with it; a generic "find every
number" scanner would be almost all false positives.

    python scripts/check_paper_numbers.py [--paper DIR]

Exits non-zero, listing every mismatch, if any check fails.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import re
import sys
from itertools import pairwise
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Artifacts a manuscript table is generated from. Listed explicitly: adding one is
# a line, and a generic scan would be mostly false positives.
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
    "exoticity_anec_convergence.json",
    "rodal_sigma_resolved.json",
    "wall_restricted_analysis.json",
    "classifier_error_rate.json",
    "enclosures.json",
    "type_transitions.json",
    "lmi_agreement.json",
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
    # Read by a check below, or generating a shipped table. A missing entry takes
    # its check down with it silently, which is what this list prevents.
    "rodal_dec_diagnosis.json",
    "comparison_table.json",
    "nstarts_ablation.json",
    "convergence_data.json",
    "integrated_negative_energy.json",
    "rodal_native_resolution.json",
    "fibonacci_dec_comparison.json",
)

DEFAULT_PAPER = Path(__file__).resolve().parents[2] / "warpax_arxiv"


def load(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    return path.read_text(encoding="utf-8")


def table_cell(body: str, row: str, col: int) -> str:
    """Cell ``col`` (0-based, after the row label) of the row labelled ``row``."""
    for line in body.splitlines():
        line = line.strip().rstrip("\\").strip()
        if not line.startswith(row):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) > col + 1:
            return cells[col + 1]
    raise SystemExit(f"row {row!r} column {col} not found in table")


def table_column(body: str, col: int, rows: list[str]) -> list[str]:
    return [table_cell(body, r, col) for r in rows]


def ray_scan_failures(row):
    """Check exclusions and selected-ray validity without treating failed rays as data."""
    errors = []
    b = row.get("b_scan", [])
    values = row.get("line_integral_scan", [])
    complete = row.get("scan_completed", [])
    preserved = row.get("scan_null_preserved", [])
    if not (len(b) == len(values) == len(complete) == len(preserved)):
        return ["inconsistent scan status lengths"]
    excluded = [
        i
        for i, (v, c, n) in enumerate(zip(values, complete, preserved, strict=True))
        if not c or not n or v is None or not math.isfinite(v)
    ]
    if row.get("scan_excluded_indices") != excluded or row.get("scan_excluded_count") != len(
        excluded
    ):
        errors.append("wrong excluded scan indices or count")
    if row.get("scan_excluded_b") != [b[i] for i in excluded]:
        errors.append("wrong excluded impact parameters")
    if row.get("all_null_preserved") != all(preserved):
        errors.append("wrong full-fan null-preservation flag")
    for label, ray in row.get("selected_ray_convergence", {}).items():
        if not ray.get("step_stable"):
            errors.append(f"{label}: unstable selected ray")
        for record in ray.get("records", []):
            if not record.get("geodesic_complete") or not record.get("null_preserved"):
                errors.append(f"{label}: incomplete or non-null selected ray")
    if not row.get("all_selected_null_preserved"):
        errors.append("selected-ray null-preservation flag fails")
    return errors


def compatible_artifact(path: Path, sources: dict[str, str], record: dict) -> bool:
    """Accept only the exact artifact/source pair with matching source hashes."""
    entry = record.get("artifacts", {}).get(str(path.relative_to(RESULTS)))
    return bool(
        entry
        and sources == record.get("sources_sha256")
        and entry.get("reason", "").strip()
        and entry.get("evidence", "").strip()
        and hashlib.sha256(path.read_bytes()).hexdigest() == entry.get("sha256")
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    args = ap.parse_args()

    tex = load(args.paper / "main.tex")
    t = lambda name: load(args.paper / "tables" / f"{name}.tex")

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
        data = json.loads(path.read_text(encoding="utf-8"))
        return None if data.get("partial") else data

    metrics = ["Alcubierre", "Natário", "Van den Broeck"]
    type_i = table_column(bench, 0, metrics)
    type_iv = table_column(bench, 1, metrics)
    rodal_dec_bench = table_cell(bench, "Rodal", 5)
    rodal_dec_missed = table_cell(missed, "Rodal", 4)
    rodal_dec_conv = table_cell(conv, "Rodal", 2)
    vdb_range = table_cell(vort, "Van den Broeck", 3)
    nat_iv_sweep = table_cell(vel, "Natário", 1)
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
            (
                f"{min(float(rodal_dec_missed), float(rodal_dec_bench)):.1f}",
                f"{max(float(rodal_dec_missed), float(rodal_dec_bench)):.1f}",
            ),
            r"DEC miss \$([\d.]+)\$--\$([\d.]+)\\%\$",
        ),
        (
            "Section 3.2 Van den Broeck low-end wall Type-IV vs shift_vorticity",
            (vdb_lo,),
            r"only \$(\d+)\\%\$ of the Van~den~Broeck wall is Type-IV",
        ),
    ]

    # ---- prose numbers pinned directly to the JSON artifacts -----------------
    #
    # The checks above compare prose against generated TABLES. Prose can also quote
    # a number that appears in no table, and sometimes in no artifact. These pin
    # those straight to results/*.json.
    sweep = artifact("velocity_sweep.json")
    if sweep is not None:
        rows = {(r["metric"], r["v_s"]): r for r in sweep["rows"]}

        def typeI_pct(metric: str, v: float) -> str:
            return f"{100 * rows[(metric, v)]['wall_frac_type_i']:.1f}"

        checks.append(
            (
                "Section 3.1 superluminal Type-I fractions vs velocity_sweep.json",
                (
                    typeI_pct("Alcubierre", 2.5),
                    typeI_pct("Natário", 2.5),
                    typeI_pct("Van den Broeck", 2.5),
                ),
                r"reach(?:ing|es) \$([\d.]+)\\%\$\s*\n\(Alcubierre\), \$([\d.]+)\\%\$ "
                r"\(Nat\\'ario\), and \$([\d.]+)\\%\$ \(Van~den~Broeck\)",
            )
        )
        checks.append(
            (
                "Section 3.1 Rodal margin at v_s=2.5 vs velocity_sweep.json",
                (f"{rows[('Rodal', 2.5)]['typeI_nec_min']:.3f}",),
                r"to \$(-[\d.]+)\$\s*\nat \$v_s=2\.5\$",
            )
        )
        # The grid the sweep actually ran on must be the grid the captions claim.
        n_run = sweep["config"]["N"]
        # Match the grid claim itself, not the surrounding caption wording, which
        # prose edits can move out from under the pattern.
        for caption_n in re.findall(r"benchmark grid\s*\(\$N=(\d+)\$", tex):
            if int(caption_n) != n_run:
                failures_pre.append(
                    "a caption attributes the velocity sweep to $N="
                    + caption_n
                    + f"$ but velocity_sweep.json ran at N={n_run}"
                )

    # The selected finite rays must be the rays actually step-refined.
    for filename, geodesic in (
        ("anec/retained_symplectic.json", True),
        ("anec/retained.json", False),
    ):
        data = artifact(filename)
        if data is None:
            continue  # Required-artifact gate below reports the missing file.
        for name, row in data["metrics"].items():
            for label, value_key in (
                ("near_axis", "on_axis"),
                ("minimum_found", "min_line_integral"),
            ):
                ray = row.get("selected_ray_convergence", {}).get(label)
                if ray is None:
                    failures_pre.append(f"{filename} {name}: missing {label} refinement")
                    continue
                values = [r["line_integral"] for r in ray["records"]] if geodesic else ray["values"]
                steps = (
                    [r["steps_per_reference_span"] for r in ray["records"]]
                    if geodesic
                    else ray["samples_per_reference_span"]
                )
                if (
                    len(values) < 3
                    or len(steps) != len(values)
                    or any(b != 2 * a for a, b in pairwise(steps))
                ):
                    failures_pre.append(f"{filename} {name} {label}: invalid refinement ladder")
                if len(values) < 2 or any(v is None or not math.isfinite(v) for v in values):
                    failures_pre.append(f"{filename} {name} {label}: nonfinite selected refinement")
                    continue
                if values[-1] != row[value_key] or ray["span"] != row["affine_span"]:
                    failures_pre.append(
                        f"{filename} {name} {label}: reported ray differs from refinement"
                    )
                expected_b = row["b_at_min"] if label == "minimum_found" else row["b_scan"][0]
                if ray["b"] != expected_b:
                    failures_pre.append(f"{filename} {name} {label}: wrong impact parameter")
                change = abs(values[-1] - values[-2])
                if change > 1e-8 + 1e-4 * abs(values[-1]):
                    failures_pre.append(
                        f"{filename} {name} {label}: step refinement has not stabilized"
                    )
                if abs(ray["finest_change"] - change) > 1e-15:
                    failures_pre.append(f"{filename} {name} {label}: wrong observed change")

    # The Rodal single-frame miss rates are quoted in five places, none of them a
    # table. Pin every one.
    inv = artifact("invariant_verification.json")
    if inv is not None:
        rod = next(r for r in inv["rows"] if r["metric"] == "Rodal")
        wec, dec = f"{rod['miss_wec_pct']:.0f}", f"{rod['miss_dec_pct']:.0f}"
        vdb = next(r for r in inv["rows"] if r["metric"].startswith("Van"))
        vdb_wec = f"{vdb['miss_wec_pct']:.0f}"
        vdb_dec = f"{vdb['miss_dec_pct']:.0f}"
        # These patterns run against the wrapped LaTeX source, so every literal
        # space is relaxed to \s+ below: a reflowed paragraph must not read as a
        # defect.
        for desc, pattern, expected in (
            (
                "abstract",
                r"For Rodal, the Eulerian reading misses about \$(\d+)\\%\$ of the sampled wall weak-energy",
                (wec,),
            ),
            (
                "Section 1",
                r"register \$\{\\approx\}(\d+)\\%\$ of the dominant and "
                r"\$\{\\approx\}(\d+)\\%\$ of the weak energy-condition",
                (dec, wec),
            ),
            (
                "Section 2",
                r"at \$\{\\approx\}(\d+)\\%\$ of the wall points where a boosted "
                r"observer sees a\s*\n?weak-energy violation, and at "
                r"\$\{\\approx\}(\d+)\\%\$ for the dominant energy",
                (wec, dec),
            ),
            (
                "Discussion, Van den Broeck wall miss",
                r"Van~den~Broeck is intermediate \(WEC \$(\d+)\\%\$,\s*\n?"
                r"DEC \$(\d+)\\%\$ wall miss",
                (vdb_wec, vdb_dec),
            ),
            (
                "Appendix C cross-reference",
                r"matched-parameter benchmark \(\$\{\\approx\}(\d+)\\%\$ DEC, "
                r"\$\{\\approx\}(\d+)\\%\$ WEC;",
                (dec, wec),
            ),
            # The Discussion and Conclusion point at Section 3.1 rather than
            # restating the fractions, so there is nothing to pin in either. The pair
            # is pinned at Section 1, Section 2 and the Appendix C cross-reference.
        ):
            # Literal spaces match any run of whitespace, newlines included.
            pattern = re.sub(r"(?<!\\s)(?<!\\n) ", r"\\s+", pattern)
            checks.append(
                (
                    f"Rodal single-frame miss rates ({desc}) vs invariant_verification.json",
                    expected,
                    pattern,
                )
            )

    # GZ fractions use the exact result; raw near-degeneracy labels are separate.
    cv = artifact("construction_verification.json")
    if cv is not None:
        for block in ("matched", "native"):
            for row in cv[block]["Garattini"]:
                pars = row["params"]
                required = ("H", "r_0", "R", "sigma", "evaluation_time", "units", "provenance")
                if any(key not in pars for key in required):
                    failures_pre.append(f"Garattini {block}: incomplete parameter provenance")
                    continue
                if (
                    pars["evaluation_time"] != 0
                    or abs(row["speed"] - pars["H"] * pars["r_0"]) > 1e-14
                ):
                    failures_pre.append(f"Garattini {block}: inconsistent epoch/matched motion")
                if (row["frac_type_i"], row["frac_type_ii"], row["frac_type_iv"]) != (
                    1.0,
                    0.0,
                    0.0,
                ):
                    failures_pre.append(
                        f"Garattini {block}: physical type fractions contradict the exact result"
                    )

    curv = artifact("curvature_scaling.json")
    if curv is not None:
        fits = curv["fits"]

        # The worst single-point departure from each fitted power law. Quoted in
        # Section 3.6 so the log-fit R^2 is not the only thing a reader sees.
        def sci(x: float, sig: int) -> str:
            mant, exp = f"{x:.{sig}e}".split("e")
            mant = mant.rstrip("0").rstrip(".")
            return rf"{mant}\times10^{{{int(exp)}}}"

        def worst(metric: str) -> float:
            return max(f["max_rel_dev"] for f in curv["fits"][metric].values())

        checks.append(
            (
                "Section 3.6 worst power-law deviations vs curvature_scaling.json",
                (
                    sci(curv["fits"]["Alcubierre"]["weyl_squared"]["max_rel_dev"], 1),
                    sci(worst("Natário"), 0),
                    sci(worst("Rodal"), 0),
                    f"{curv['fits']['Alcubierre']['ricci_squared']['max_rel_dev']:.2f}",
                ),
                r"is\s*\n?\$(.+?)\$ \(Alcubierre, Weyl\), \$(.+?)\$ \(Nat\\'ario, all three\) and\s*\n?"
                r"\$(.+?)\$ \(Rodal, all three\), rising to \$([\d.]+)\$ for\s*Alcubierre Ricci-squared",
            )
        )

    fibonacci = artifact("fibonacci_dec_comparison.json")
    if fibonacci is not None:
        count = f"{fibonacci['n_violations']:,}".replace(",", r"\,")
        total = f"{fibonacci['n_points']:,}".replace(",", r"\,")
        checks.append(
            (
                "Fibonacci caption: algebraic violating and sampled point counts",
                (count, total),
                r"\$(.+?)\$ violating points of the \$(.+?)\$ sampled",
            )
        )
        checks.append(
            (
                "Fibonacci caption: finest sampled detection rate",
                (f"{fibonacci['results'][-1]['detection_rate']:.1f}",),
                r"Detection rate \(circles\) reaches \$([\d.]+)\\%\$",
            )
        )
        checks.append(
            (
                "Fibonacci caption: minimum algebraic DEC slack",
                (f"{fibonacci['alg_min_dec'] / 1e-6:.4f}",),
                r"algebraic slack is \$(-[\d.]+)\\times10\^\{-6\}\$",
            )
        )

    # Appendix H quotes the type-transition comparison in prose rather than only through a
    # table, and those are the numbers that carry the answer to the exhaustiveness
    # objection. Pin them to the artifact.
    tt = artifact("type_transitions.json")
    if tt is not None:
        fam = tt["families"]["momentum_aligned"]
        t3 = tt["type_iii_chain"]
        n_iv_at_tight = sum(1 for r in t3["rows"] if r["labels"]["tol_1e-10"] == 4)
        checks.append(
            (
                "Appendix H momentum-family sample count vs type_transitions.json",
                (str(fam["n"]),),
                r"Across \$(\d+)\$ samples\s+the\s+maximum observed adjacent-grid slope",
            )
        )
        error_mantissa, error_power = f"{fam['max_lmi_vs_brute_abs_err']:.1e}".split("e")
        checks.append(
            (
                "Transition sample contraction discrepancy vs corrected finite sample",
                (error_mantissa, str(int(error_power))),
                r"Twice the margin differs from the finite-sample null minimum by at most\s*"
                r"\$([\d.]+)\\times10\^\{(-?\d+)\}\$",
            )
        )
        checks.append(
            (
                "Appendix H Type-III branch: numerical labels and negative margins",
                (
                    str(t3["n"]),
                    str(n_iv_at_tight),
                    str(sum(1 for r in t3["rows"] if r["nec_margin"] < -r["noise_floor"])),
                ),
                r"family, \$(\d+)\$ points log-spaced over[\s\S]{0,400}?"
                r"Type~IV at every one of the\s*\n?\$(\d+)\$[\s\S]{0,600}?"
                r"detects null-energy violation at all \$(\d+)\$",
            )
        )

        # The tolerance sweep is the sharpest number in the appendix and the one most
        # likely to drift, since it is the only place the label is quoted as a
        # function of a knob rather than as a verdict.
        def tol_counts(tol: str) -> tuple[str, str, str]:
            c = collections.Counter(r["labels"][tol] for r in t3["rows"])
            return (str(c[2]), str(c[3]), str(c[4]))

        checks.append(
            (
                "Appendix H Type-III tolerance sweep vs type_transitions.json",
                tol_counts("tol_2e-06") + tol_counts("tol_5e-06"),
                r"Type~II at \$(\d+)\$, Type~III at \$(\d+)\$ and Type~IV at \$(\d+)\$, "
                r"and at \$5\\times10\^\{-6\}\$ it\s*\n?returns \$(\d+)\$, \$(\d+)\$ and \$(\d+)\$",
            )
        )

    # tab:rodal_ablation is the only manuscript table whose numbers are typed into
    # an inline tabular rather than \input from a generated file. Its values exist
    # in the artifact, so pin them.
    rodal_abl = artifact("rodal_dec_diagnosis.json")
    if rodal_abl is not None:
        # The sweep is stored as two parallel lists, not as a mapping keyed by N.
        res = rodal_abl.get("sweeps", {}).get("resolution", {})
        by_n = dict(zip(res.get("values", []), res.get("rodal_dec_miss_pct", []), strict=True))
        want = [f"{by_n[n]:.2f}" for n in (25, 50, 100) if isinstance(by_n.get(n), (int, float))]
        if len(want) == 3:
            checks.append(
                (
                    "Table rodal_ablation resolution rows vs rodal_dec_diagnosis.json",
                    tuple(want),
                    r"& \$25\$\s*& ([\d.]+) & 0\.0 \\\\\s*\n\s*& \$50\$\s*& ([\d.]+) "
                    r"& 0\.0 \\\\\s*\n\s*& \$100\$\s*& ([\d.]+) & 0\.0",
                )
            )
        else:
            failures_pre.append(
                "rodal_dec_diagnosis.json no longer exposes the three resolution "
                "rows that tab:rodal_ablation types by hand; re-point this check"
            )

    failures: list[str] = list(failures_pre)
    for desc, expected, pattern in checks:
        # Every literal space is relaxed to \s+ before matching, so a reflowed
        # paragraph cannot move a line break into a pinned phrase and read as a
        # defect.
        pattern = re.sub(r"(?<!\\s)(?<!\\n) ", r"\\s+", pattern)
        m = re.search(pattern, tex)
        if m is None:
            failures.append(f"{desc}\n    prose pattern not found: {pattern}")
            continue
        found = tuple(g.strip() for g in m.groups())
        if found != tuple(e.strip() for e in expected):
            failures.append(f"{desc}\n    tables say {expected}, prose says {found}")

    # The exoticity Type-IV column is the velocity sweep expressed as a fraction.
    if abs(float(nat_exo_iv) * 100.0 - float(nat_iv_sweep)) > 0.05:
        failures.append(
            "exoticity_ranking Type-IV column vs velocity_type_structure\n"
            f"    {nat_exo_iv} (as a fraction) != {nat_iv_sweep}% for Natario"
        )

    # The wall-cell ladder is quoted with one range everywhere it appears, and the
    # range comes from the artifact. A hard-coded triple lets a stale paper and a
    # stale check agree with each other while the data says something else.
    diag = artifact("diagnostic_convergence.json")
    if diag is not None:
        cells = diag["wall_info"]["Alcubierre"]
        rungs = [f"{cells[str(n)]['cells']:.1f}" for n in diag["ladder_N"]]
        if not re.search(r"\$" + r"\$, \$".join(re.escape(c) for c in rungs) + r"\$ cells", tex):
            failures.append(
                f"wall-cell ladder {'/'.join(rungs)} (diagnostic_convergence.json) is "
                f"not the ladder main.tex states"
            )
        ranges = set(re.findall(r"wall by \$([\d.]+)\$ to\s*\n?\$?([\d.]+)\$? cells", tex))
        for lo, hi in ranges:
            if (lo, hi) != (rungs[0], rungs[-1]):
                failures.append(
                    f"wall-cell range ${lo}$ to ${hi}$ contradicts the {'/'.join(rungs)} ladder"
                )

    # Independent achieved upper bounds need not be ordered. Both must lie
    # above a valid continuum lower bound for the same wall objective.
    cen, enc2 = artifact("interval_lmi_census.json"), artifact("enclosures.json")
    if cen is not None and enc2 is not None:
        alias = {
            "Alcubierre": "Alcubierre",
            "Natário": "Natario",
            "Van den Broeck": "VanDenBroeck",
            "Rodal": "Rodal",
        }
        for drive, row in cen["results"].items():
            nec = row["conditions"]["nec"] if "conditions" in row else row
            br = enc2["results"][alias[drive]]
            if not br["lower"] <= nec["deepest_upper"]:
                failures.append(f"{drive}: census contradicts the global lower bound")
            if not br["lower"] <= br["upper"]:
                failures.append(f"{drive}: reversed global enclosure")
            if not br["lower"] <= br["point_lower"] <= br["point_upper"]:
                failures.append(f"{drive}: point enclosure contradicts the global lower bound")

    # The hand-maintained macros in paper_numbers.tex are quoted numbers no table
    # carries. emit_paper_numbers.py rewrites only the auto-sourced block, so
    # without this check nothing stands between those five and their artifacts.
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
                    f"\\{macro} is {m.group(1)}, wall_restricted_analysis.json says {want}"
                )

    # Rodal's DEC miss on the N=120 level must be the value the tables agree on.
    if rodal_dec_conv != rodal_dec_missed:
        failures.append(
            "convergence_per_metric N=120 and missed_wall_restricted disagree for "
            f"Rodal DEC: {rodal_dec_conv} vs {rodal_dec_missed}"
        )

    # No artifact feeding a table may predate the code that produced it. Watch the
    # whole library and every generator, since a quadrature weight or a polished
    # extremum moves numbers too; exclude this file, which checks rather than makes.
    src_dir = RESULTS.parent / "src" / "warpax"
    scripts_dir = pathlib.Path(__file__).resolve().parent
    if src_dir.is_dir():
        watched = list(src_dir.rglob("*.py")) + [
            q for q in scripts_dir.glob("*.py") if q.name != pathlib.Path(__file__).name
        ]
        newest_code = max((q.stat().st_mtime for q in watched), default=0.0)
        source_hashes = {
            str(q.relative_to(RESULTS.parent)): hashlib.sha256(q.read_bytes()).hexdigest()
            for q in watched
        }
        compatibility_path = args.paper / "verify" / "numerical_compatibility.json"
        compatibility = (
            json.loads(compatibility_path.read_text()) if compatibility_path.exists() else {}
        )
        for name in sorted(_TABLE_ARTIFACTS):
            path = RESULTS / name
            # A MISSING artifact is a failure, not a skip. Every JSON-pinned check
            # returns early when its input is absent, so without this a wiped results/
            # passes while checking almost nothing.
            if not path.exists():
                failures.append(
                    f"results/{name} is missing; every check that reads it was "
                    f"silently skipped. Run reproduce_all.sh before quoting a number"
                )
                continue
            if path.stat().st_mtime < newest_code and not compatible_artifact(
                path, source_hashes, compatibility
            ):
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
        # already exists, on existence alone, so a re-run after a source edit rebuilds
        # every JSON from older grids and each JSON then carries a fresh mtime.
        # reproduce_all.sh deletes the grids unless --keep-cache is passed.
        stale_npz = sorted(
            q.name
            for q in RESULTS.glob("*.npz")
            if q.stat().st_mtime < newest_code
            and not compatible_artifact(q, source_hashes, compatibility)
        )
        if stale_npz:
            failures.append(
                f"{len(stale_npz)} cached grid(s) in results/ predate the newest file "
                f"in src/warpax/ or scripts/, starting with {stale_npz[0]}; every JSON "
                f"rebuilt from them inherits pre-edit inputs. Re-run reproduce_all.sh "
                f"WITHOUT --keep-cache"
            )

    # Every table must say which script wrote it and which artifact it read, and
    # both must still exist: a header naming a renamed script reads as provenance
    # while pointing at nothing.
    tables_dir = RESULTS.parent.parent / "warpax_arxiv" / "tables"
    repo = RESULTS.parent
    if tables_dir.is_dir():
        for name in sorted(set(re.findall(r"\\input\{(tables/[^}]+)\}", tex))):
            tex_path = args.paper / name
            head = tex_path.read_text(errors="replace").lstrip().split("\n", 1)[0]
            if not head.startswith(("% Generated by", "% Hand-written")):
                failures.append(
                    f"tables/{tex_path.name} has no provenance header; emit it through "
                    f"_json_io.write_table, or mark it '% Hand-written: <why>'"
                )
                continue
            for ref in re.findall(r"(?:scripts|results)/[\w./-]+", head):
                if not (repo / ref).exists():
                    failures.append(f"tables/{tex_path.name} provenance names a missing {ref}")

    # Both ANEC tables sit side by side in the paper, so they must share one affine
    # window rule.
    try:
        for fn in ("anec/retained.json", "anec/retained_symplectic.json"):
            note = artifact(fn)["params"]["affine_span_note"]
            if "stationary" in note or "r_s = 3" not in note:
                failures.append(
                    f"{fn}: affine window is not the measured crossing-span rule "
                    f"(note: {note[:60]}...)"
                )
    except Exception as exc:  # pragma: no cover - defensive
        failures.append(f"ANEC window-rule check could not run: {exc}")

    # The manuscript quotes both integrator witnesses; neither may be asserted.
    try:
        sym = artifact("anec/retained_symplectic.json")["metrics"]
        for name, m in sym.items():
            failures.extend(f"Finite ray {name}: {error}" for error in ray_scan_failures(m))
            if m.get("scan_excluded_count", 0):
                if not re.search(
                    r"(?:failed|incomplete|excluded)[^.!?]{0,300}(?:scan|fan)|"
                    r"(?:scan|fan)[^.!?]{0,300}(?:failed|incomplete|excluded)",
                    tex,
                    re.I,
                ):
                    failures.append(
                        f"Finite ray {name}: missing manuscript disclosure of excluded fan rays"
                    )
            drift = m.get("worst_killing_energy_drift")
            if drift is None:
                failures.append(f"ANEC {name}: Killing-energy witness not computed")
            elif drift >= 1e-5:
                failures.append(f"ANEC {name}: Killing drift {drift:.2e} exceeds the quoted 1e-5")
    except Exception as exc:  # pragma: no cover - defensive
        failures.append(f"ANEC witness check could not run: {exc}")

    # Aggregate fraction differences do not check the pointwise splitting premise.
    try:
        for row in artifact("closing_speed.json")["rows"]:
            for predicted, measured, gap in zip(
                row["predicted_frac_type_iv"],
                row["measured_frac_type_iv"],
                row["fraction_gap_pp"],
                strict=True,
            ):
                if abs(100 * (measured - predicted) - gap) > 1e-10:
                    failures.append(f"closing speed {row['metric']}: inconsistent fraction gap")
    except Exception as exc:  # pragma: no cover - defensive
        failures.append(f"closing-speed comparison check could not run: {exc}")

    # The cached grids are gitignored, so the manifest is their only integrity
    # record. Keep it in the same gate as the numbers it backs.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import write_manifest

        failures.extend(f"MANIFEST: {p}" for p in write_manifest.check())
    except Exception as exc:  # pragma: no cover - defensive
        failures.append(f"MANIFEST check could not run: {exc}")

    if failures:
        print(f"{len(failures)} paper-number check(s) FAILED:\n")
        for f in failures:
            print(f"  - {f}\n")
        return 1
    # The non-`checks` gates, counted by name rather than by a constant.
    extra_gates = (
        "caption-N pin",
        "selected finite-ray refinement",
        "exoticity/velocity cross-table",
        "wall-cell ladder",
        "Rodal DEC convergence cross-table",
        "artifact existence/staleness/partial",
        "cached-grid staleness",
        "table provenance headers",
        "ANEC window rule",
        "ANEC witnesses",
        "closing-speed comparison",
        "census and point/global enclosure consistency",
        "paper_numbers.tex macros",
        "MANIFEST",
    )
    print(f"all {len(checks) + len(extra_gates)} paper-number checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
