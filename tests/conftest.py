"""Shared test fixtures for the warpax JAX test suite.

Float64 enforcement is verified at import time and fails LOUD if x64 is not
enabled. Every test that touches JAX arrays should confirm dtype == float64 in
its assertions.

``--gpu-baseline`` applies xfail markers from ``_gpu_xfail_registry`` to
known-bad tests on Blackwell sm_120 and summarizes at session finish, so real
regressions surface among the expected failures.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax.numpy as jnp
import pytest

from warpax.geometry.types import GridSpec

_probe = jnp.array(1.0)
assert _probe.dtype == jnp.float64, (
    f"JAX float64 not enabled!  Got dtype={_probe.dtype}.  "
    "Ensure jax.config.update('jax_enable_x64', True) runs before any JAX import."
)


@pytest.fixture
def default_grid() -> GridSpec:
    """A small 3D grid for testing (z thin-slice)."""
    return GridSpec(
        bounds=[(-5.0, 5.0), (-5.0, 5.0), (-0.5, 0.5)],
        shape=(16, 16, 4),
    )


@pytest.fixture
def sample_coords() -> jnp.ndarray:
    """Standard test coordinate 4-tuple (t, x, y, z)."""
    return jnp.array([0.0, 1.0, 2.0, 3.0])


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
    """xfail-tag the registry entries when --gpu-baseline is set.

    Staleness guard: a registry pointing at renamed tests turns every expected
    failure into a real regression, so warn when an entry whose file was
    collected matches no collected test; restricting to collected files spares
    subset runs.
    """
    if not config.getoption("--gpu-baseline"):
        return

    from ._gpu_xfail_registry import EXPECTED_GPU_FAILURES

    collected_ids = {item.nodeid for item in items}
    collected_files = {nid.split("::", 1)[0] for nid in collected_ids}
    unmatched = sorted(
        e
        for e in EXPECTED_GPU_FAILURES
        if e.split("::", 1)[0] in collected_files and e not in collected_ids
    )
    if unmatched:
        import warnings

        warnings.warn(
            "GPU xfail registry has "
            f"{len(unmatched)} entr{'y' if len(unmatched) == 1 else 'ies'} "
            "matching no collected test (stale node IDs): "
            + ", ".join(unmatched[:5])
            + (" ..." if len(unmatched) > 5 else ""),
            stacklevel=2,
        )

    for item in items:
        if item.nodeid in EXPECTED_GPU_FAILURES:
            item.add_marker(
                pytest.mark.xfail(
                    reason="sm_120 cuBLAS LT expected failure",
                    strict=False,
                )
            )


def pytest_sessionfinish(session, exitstatus) -> None:
    """emit CPU/GPU delta-report summary when --gpu-baseline is set.

    Walks the terminalreporter stats (pytest-internal): xfailed entries in the
    registry are expected, xpassed entries in the registry are resolved xfails
    to investigate, and failed nodes outside the registry are new regressions.
    """
    if not session.config.getoption("--gpu-baseline"):
        return

    from ._gpu_xfail_registry import EXPECTED_GPU_FAILURES

    expected_xfail_count = 0
    resolved_xfail_count = 0
    new_regression_count = 0

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    stats = reporter.stats

    xfailed_nodes = {r.nodeid for r in stats.get("xfailed", [])}
    xpassed_nodes = {r.nodeid for r in stats.get("xpassed", [])}
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

    lines = [
        "",
        "=== GPU baseline summary ===",
        f"Expected xfails: {expected_xfail_count}/{len(EXPECTED_GPU_FAILURES)}",
        (f"Resolved xfails (now passing - investigate): {resolved_xfail_count}"),
        f"New regressions (not in registry, failing): {new_regression_count}",
        "=== End summary ===",
    ]
    reporter.write_line("\n".join(lines))
