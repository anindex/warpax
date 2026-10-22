"""The code base carries no comments; facts live in docstrings and tests."""

import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = ("src", "tests", "scripts", "benchmarks", "examples")
PRAGMA = re.compile(r"^#\s*(noqa|type:|pragma|ruff:|fmt:|isort:)")


def _comments(path: Path) -> list[str]:
    src = path.read_text()
    found = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.COMMENT or PRAGMA.match(tok.string):
            continue
        if tok.start[0] == 1 and tok.string.startswith("#!"):
            continue
        found.append(f"{path.relative_to(ROOT)}:{tok.start[0]}: {tok.string}")
    return found


def test_no_comments_in_python_sources():
    offenders = [c for d in DIRS for p in sorted((ROOT / d).rglob("*.py")) for c in _comments(p)]
    assert not offenders, "\n".join(offenders[:20])
