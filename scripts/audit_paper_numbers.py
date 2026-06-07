"""Audit paper_numbers.tex against the cached analysis outputs.

Recomputes every auto-sourced macro from ``results/*.json`` and compares it to
the value committed in ``warpax_arxiv/paper_numbers.tex``; exits non-zero on any
drift. Manually-maintained macros (multi-file/derived provenance) are listed
with their source for review and their source file is checked for existence.

This is the CI guard behind the referee-proofing rule "every cited number traces
to results/*.json". Run after the K-stage reproduction.

Usage::

    python scripts/audit_paper_numbers.py
"""
from __future__ import annotations

import os
import re
import sys

from _paper_numbers_map import AUTO_SOURCED, MANUAL, RESULTS, recompute_auto

HERE = os.path.dirname(__file__)
PAPER_NUMBERS = os.path.join(
    HERE, "..", "..", "warpax_arxiv", "paper_numbers.tex"
)

_MACRO_RE = re.compile(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}")


def parse_macros(path: str) -> dict[str, str]:
    with open(path) as f:
        text = f.read()
    return {m.group(1): m.group(2) for m in _MACRO_RE.finditer(text)}


def main() -> int:
    macros = parse_macros(PAPER_NUMBERS)
    recomputed = recompute_auto()

    failures = []
    print("=" * 70)
    print("PAPER NUMBER AUDIT")
    print("=" * 70)
    print("Auto-sourced macros (recomputed vs committed):")
    for name, (_fn, _nd, src) in AUTO_SOURCED.items():
        committed = macros.get(name)
        want = recomputed[name]
        ok = committed == want
        status = "OK " if ok else "DRIFT"
        print(f"  [{status}] \\{name} = {committed!r}  (recomputed {want!r})  <- {src}")
        if not ok:
            failures.append((name, committed, want, src))

    print("\nManually-maintained macros (source existence check):")
    for name, src in MANUAL.items():
        committed = macros.get(name)
        srcfile = src.split()[0]
        exists = os.path.exists(os.path.join(RESULTS, srcfile)) or "config" in src
        mark = "ok" if (committed is not None and exists) else "MISSING"
        print(f"  [{mark}] \\{name} = {committed!r}  <- {src}")
        if committed is None or not exists:
            failures.append((name, committed, "<source missing>", src))

    # Any committed macro not covered by the map is an untracked number.
    tracked = set(AUTO_SOURCED) | set(MANUAL)
    untracked = sorted(set(macros) - tracked)
    if untracked:
        print("\nUntracked macros (add to _paper_numbers_map.py):")
        for name in untracked:
            print(f"  [WARN] \\{name} = {macros[name]!r}")

    if failures:
        print(f"\nFAIL: {len(failures)} macro(s) drifted or lack a source.")
        return 1
    print("\nPASS: all auto-sourced macros match; manual sources present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
