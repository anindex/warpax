"""The numerical pipeline runs independently of manuscript validation."""

import os
import shutil
import subprocess
import sys
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


@pytest.mark.parametrize(
    ("extra_env", "expected_secs"),
    [({}, 0.05), ({"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0.5"}, 0.5)],
)
def test_jit_cache_lowers_persistent_cache_thresholds(tmp_path, extra_env, expected_secs):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("JAX_PERSISTENT_CACHE") and k != "JAX_COMPILATION_CACHE_DIR"
    }
    env.update(
        {
            "WARPAX_JIT_CACHE": "1",
            "JAX_COMPILATION_CACHE_DIR": str(tmp_path),
            "JAX_PLATFORMS": "cpu",
            **extra_env,
        }
    )
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import warpax, jax; c = jax.config; "
            "print(c.jax_compilation_cache_dir, c.jax_persistent_cache_min_compile_time_secs, "
            "c.jax_persistent_cache_min_entry_size_bytes)",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert out == [str(tmp_path), str(expected_secs), "-1"]
