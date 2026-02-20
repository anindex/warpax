"""Shared test fixtures for the warpax JAX test suite.

Float64 enforcement is verified at import time.  Every test that touches
JAX arrays should confirm dtype == float64 in its assertions.
"""

import jax
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
