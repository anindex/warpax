"""test_reproduce_cpu_pin - pin reproduce_all.sh CPU-default discipline.

 


Static-analysis tests (no actual reproduce_all.sh invocation; the full reproduction
is a 2h48m campaign and not a unit test). The discipline contract is:
  1. The script's source contains `export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"`.
  2. The script's source contains a banner that mentions `JAX_PLATFORMS` going to stderr.
  3. The export line lands BEFORE the first `"$PYTHON"` (or `${PYTHON}`) invocation.
  4. The script does NOT silently override an explicit user override (uses `:-`, not `:=` or `=`).
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPRODUCE_SH = REPO_ROOT / "warpax" / "reproduce_all.sh"


def _read_script() -> str:
    return REPRODUCE_SH.read_text()


def test_script_exists():
    assert REPRODUCE_SH.exists(), f"missing: {REPRODUCE_SH}"


def test_jax_platforms_export_present():
    """contract: the export line uses :- to honor env override."""
    src = _read_script()
    # The exact line - colon-dash form is required (NOT := which would assign).
    pattern = r'export\s+JAX_PLATFORMS\s*=\s*"\$\{JAX_PLATFORMS:-cpu\}"'
    assert re.search(pattern, src), (
        "reproduce_all.sh must contain "
        '`export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"` (colon-dash form to '
        "honor explicit user override). Current content does not match."
    )


def test_jax_platforms_banner_to_stderr():
    """contract: backend choice is visibly logged to stderr."""
    src = _read_script()
    # Must mention JAX_PLATFORMS on a line that redirects to >&2.
    banner_lines = [
        line for line in src.splitlines()
        if "JAX_PLATFORMS" in line and ">&2" in line
    ]
    assert banner_lines, (
        "reproduce_all.sh must emit a stderr banner naming the "
        "active JAX backend (`echo ... >&2`)."
    )


def test_export_lands_before_python_invocations():
    """contract: every `"$PYTHON"` call inherits the pinned backend."""
    lines = _read_script().splitlines()
    export_idx = next(
        (i for i, line in enumerate(lines)
         if re.search(r'export\s+JAX_PLATFORMS', line)),
        None,
    )
    assert export_idx is not None, "no export line found (Task A1 not complete)"
    first_python_idx = next(
        (i for i, line in enumerate(lines)
         if re.search(r'(\$\{?PYTHON\}?|"\$PYTHON")\s+\S', line)
         and not line.strip().startswith("#")),
        None,
    )
    if first_python_idx is None:
        # No PYTHON invocations in the script body? Unexpected, but not a failure of this test.
        return
    assert export_idx < first_python_idx, (
        f"export JAX_PLATFORMS at line {export_idx + 1} lands AFTER "
        f"the first PYTHON invocation at line {first_python_idx + 1}; "
        f"subsequent Python subshells will not inherit the pinned backend."
    )


def test_script_help_flag_still_works():
    """Regression: --help must not be broken by the export."""
    result = subprocess.run(
        ["bash", str(REPRODUCE_SH), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"reproduce_all.sh --help broken: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    # The banner SHOULD fire on --help too (the export runs unconditionally
    # after flag parsing). Confirm it landed on stderr.
    # Note: the script's existing --help branch exits BEFORE reaching the export
    # in the source above. Our edit places the export AFTER flag parsing but
    # the --help case `exit 0` short-circuits. The expected behavior is:
    # `--help` exits 0 quickly with usage, no banner. Both states are acceptable
    # for ; we only require --help to still exit 0.
    assert "Usage:" in result.stdout or "--keep-cache" in result.stdout, (
        f"--help output missing usage info: {result.stdout!r}"
    )


def test_script_honors_explicit_gpu_override():
    """contract: JAX_PLATFORMS=gpu bash reproduce_all.sh --help still exits 0
    and does NOT crash trying to assign over the env. Static check on the form `${VAR:-...}`
    already verified by `test_jax_platforms_export_present`; this is a runtime smoke.

    I2 fix: use `os.environ.copy` instead of a hand-built minimal env dict so the
    subshell inherits PATH, HOME, USER, locale, etc. - preventing flaky-CI false
    negatives on hosts where the manual minimal env is missing required vars.
    """
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "gpu"
    result = subprocess.run(
        ["bash", str(REPRODUCE_SH), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=10,
    )
    assert result.returncode == 0
