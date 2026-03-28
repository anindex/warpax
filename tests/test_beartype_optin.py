"""Tests for the optional beartype runtime type-checking hook.

Setting ``WARPAX_BEARTYPE=1`` before importing the package installs
``beartype.claw`` across every submodule; leaving it unset must not touch
beartype at all. The checks run via subprocess because the env flag has to be
honored at the real ``import warpax`` moment -- ``beartype.claw`` cannot be
retrofitted onto an already-imported package.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_subprocess(env_flag: str | None, script: str) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/usr/local/bin"}
    if env_flag is not None:
        env["WARPAX_BEARTYPE"] = env_flag
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_default_off_does_not_install_claw():
    """With WARPAX_BEARTYPE unset, the library imports cleanly without beartype.claw."""
    result = _run_subprocess(
        None,
        """
        import warpax
        # If beartype.claw were active at import time, it would have re-imported
        # every module; we just check that import succeeds.
        assert isinstance(warpax.__version__, str) and warpax.__version__
        print("OK")
        """,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "OK" in result.stdout


def test_opt_in_activates_claw():
    """With WARPAX_BEARTYPE=1, importing warpax should install beartype.claw.

    Verify by inspecting that `beartype.claw` has been imported after `import warpax`.
    """
    result = _run_subprocess(
        "1",
        """
        import sys
        import warpax
        # When beartype_this_package is called, beartype.claw.*modules are loaded.
        assert "beartype.claw" in sys.modules or any(
            m.startswith("beartype.claw") for m in sys.modules
        ), "beartype.claw was not activated by WARPAX_BEARTYPE=1"
        print("OK")
        """,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "OK" in result.stdout


def test_opt_in_off_leaves_beartype_unimported():
    """With WARPAX_BEARTYPE unset, beartype.claw must NOT be in sys.modules after import."""
    result = _run_subprocess(
        None,
        """
        import sys
        import warpax
        claw_modules = [m for m in sys.modules if m.startswith("beartype.claw")]
        assert not claw_modules, f"Unexpected beartype.claw modules loaded: {claw_modules}"
        print("OK")
        """,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "OK" in result.stdout
