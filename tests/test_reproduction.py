"""The numerical pipeline runs independently of manuscript validation."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("args", [[], ["--stage", "gate"]])
def test_manuscript_gate_is_opt_in(tmp_path, args):
    repo = tmp_path / "warpax"
    repo.mkdir()
    (repo / "results").mkdir()
    elastic = repo / "results" / "elastic_shell"
    elastic.mkdir()
    coefficients = elastic / "centrifugal_response.json"
    coefficients.write_text('{"exact": "1/5"}\n')
    (repo / "figures").mkdir()
    script = repo / "reproduce_all.sh"
    shutil.copyfile(Path(__file__).resolve().parents[1] / script.name, script)
    interpreter = tmp_path / "python"
    interpreter.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n')
    interpreter.chmod(0o755)
    calls = tmp_path / "calls.txt"
    subprocess.run(
        ["bash", str(script), *args],
        env={**os.environ, "PYTHON": str(interpreter), "CALL_LOG": str(calls)},
        check=True,
        capture_output=True,
        text=True,
    )
    log = calls.read_text()
    assert ("check_paper_numbers.py" in log) == bool(args)
    assert ("run_velocity_sweep.py" in log) == (not args)
    assert coefficients.read_text() == '{"exact": "1/5"}\n'
