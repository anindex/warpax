"""Shared test fixtures for the warpax JAX test suite.

Float64 enforcement is verified at import time. Every test that touches
JAX arrays should confirm dtype == float64 in its assertions.
"""

import jax.numpy as jnp
import pytest

from warpax.geometry.types import GridSpec
from warpax.benchmarks.minkowski import MinkowskiMetric
from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.benchmarks.alcubierre import AlcubierreMetric

# ---------------------------------------------------------------------------
# Float64 enforcement check fails LOUD if x64 is not enabled
# ---------------------------------------------------------------------------
_probe = jnp.array(1.0)
assert _probe.dtype == jnp.float64, (
    f"JAX float64 not enabled!  Got dtype={_probe.dtype}.  "
    "Ensure jax.config.update('jax_enable_x64', True) runs before any JAX import."
)


# ---------------------------------------------------------------------------
# Grid fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_grid() -> GridSpec:
    """A small 3D grid for testing (z thin-slice)."""
    return GridSpec(
        bounds=[(-5.0, 5.0), (-5.0, 5.0), (-0.5, 0.5)],
        shape=(16, 16, 4),
    )


@pytest.fixture
def fine_grid() -> GridSpec:
    """A finer grid for convergence testing."""
    return GridSpec(
        bounds=[(-5.0, 5.0), (-5.0, 5.0), (-0.5, 0.5)],
        shape=(32, 32, 8),
    )


# ---------------------------------------------------------------------------
# Coordinate fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_coords() -> jnp.ndarray:
    """Standard test coordinate 4-tuple (t, x, y, z)."""
    return jnp.array([0.0, 1.0, 2.0, 3.0])


@pytest.fixture
def origin_coords() -> jnp.ndarray:
    """Origin for edge-case testing."""
    return jnp.array([0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Metric fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def all_metrics() -> list:
    """All three benchmark metrics for parameterized tests."""
    return [MinkowskiMetric(), SchwarzschildMetric(), AlcubierreMetric()]


# ---------------------------------------------------------------------------
# --gpu-baseline pytest plugin
# ---------------------------------------------------------------------------
#
# When --gpu-baseline is passed, this plugin applies xfail markers from
# ``_gpu_xfail_registry.EXPECTED_GPU_FAILURES`` to known-bad tests on
# Blackwell sm_120 GPUs and emits a short summary at session finish so
# real regressions surface without drowning in expected sm_120 failures.


def pytest_addoption(parser) -> None:
    """Add --gpu-baseline flag."""
    parser.addoption(
        "--gpu-baseline",
        action="store_true",
        default=False,
        help=(
            "apply xfail markers to the Blackwell sm_120 expected failures "
            "(registry: tests/_gpu_xfail_registry.py) and emit a CPU/GPU "
            "delta-report summary at session finish."
        ),
    )


def pytest_collection_modifyitems(config, items) -> None:
    """xfail-tag the registry entries when --gpu-baseline is set."""
    if not config.getoption("--gpu-baseline"):
        return

    from ._gpu_xfail_registry import EXPECTED_GPU_FAILURES

    for item in items:
        if item.nodeid in EXPECTED_GPU_FAILURES:
            item.add_marker(
                pytest.mark.xfail(
                    reason="sm_120 cuBLAS LT expected failure ()",
                    strict=False,
                )
            )


def pytest_sessionfinish(session, exitstatus) -> None:
    """emit CPU/GPU delta-report summary when --gpu-baseline is set."""
    if not session.config.getoption("--gpu-baseline"):
        return

    from ._gpu_xfail_registry import EXPECTED_GPU_FAILURES

    expected_xfail_count = 0
    resolved_xfail_count = 0
    new_regression_count = 0

    # Walk session results. pytest's terminalreporter stores stats in
    # session.config.stash or on the reporter; we use a conservative
    # walk over collected items' last result (pytest-internal).
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    stats = reporter.stats

    # Expected xfails in the registry
    xfailed_nodes = {r.nodeid for r in stats.get("xfailed", [])}
    # xpassed == previously xfailed now passing => resolved xfail (investigate)
    xpassed_nodes = {r.nodeid for r in stats.get("xpassed", [])}
    # failed == new regression unless in registry (which would have been xfailed)
    failed_nodes = {r.nodeid for r in stats.get("failed", [])}

    for node in xfailed_nodes:
        if node in EXPECTED_GPU_FAILURES:
            expected_xfail_count += 1

    for node in xpassed_nodes:
        if node in EXPECTED_GPU_FAILURES:
            resolved_xfail_count += 1

    for node in failed_nodes:
        if node not in EXPECTED_GPU_FAILURES:
            new_regression_count += 1

    # Print summary to stdout (reporter writes directly for visibility)
    lines = [
        "",
        "=== GPU baseline summary ===",
        f"Expected xfails: {expected_xfail_count}/{len(EXPECTED_GPU_FAILURES)}",
        (
            f"Resolved xfails (now passing - investigate): "
            f"{resolved_xfail_count}"
        ),
        f"New regressions (not in registry, failing): {new_regression_count}",
        "=== End summary ===",
    ]
    reporter.write_line("\n".join(lines))
