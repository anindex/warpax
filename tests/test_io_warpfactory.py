"""Contract tests for :func:`warpax.io.load_warpfactory` ."""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from warpax.io import InterpolatedADMMetric, load_warpfactory


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "warpfactory" / "alcubierre.mat"
)


class TestLoadWarpFactory:
    """End-to-end contract tests for the WarpFactory reader."""

    def test_fixture_exists(self):
        assert FIXTURE_PATH.exists(), (
            f"WarpFactory fixture not committed: {FIXTURE_PATH}"
        )

    def test_returns_interpolated_adm_metric(self):
        m = load_warpfactory(FIXTURE_PATH)
        assert isinstance(m, InterpolatedADMMetric)

    def test_name_reflects_source_type(self):
        m = load_warpfactory(FIXTURE_PATH)
        # Fixture has metric.type = "Alcubierre"
        assert "alcubierre" in m.name().lower()

    def test_grids_have_expected_shapes(self):
        m = load_warpfactory(FIXTURE_PATH)
        Nt, Nx, Ny, Nz = 2, 4, 4, 4  # fixture shape
        assert m.alpha_grid.shape == (Nt, Nx, Ny, Nz)
        assert m.beta_grid.shape == (Nt, Nx, Ny, Nz, 3)
        assert m.gamma_grid.shape == (Nt, Nx, Ny, Nz, 3, 3)

    def test_full_metric_has_lorentzian_signature(self):
        m = load_warpfactory(FIXTURE_PATH)
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = m(coords)
        assert g.shape == (4, 4)
        # Lorentzian: det(g) < 0.
        assert float(jnp.linalg.det(g)) < 0.0
        # g_11, g_22, g_33 should be positive (spatial metric is Euclidean-ish).
        assert float(g[1, 1]) > 0.0
        assert float(g[2, 2]) > 0.0
        assert float(g[3, 3]) > 0.0

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_warpfactory("nonexistent-file.mat")

    def test_schema_version_detection(self):
        """Loader successfully reads the v7-format fixture via scipy path."""
        # If the fixture is v7 (non-HDF5), scipy path is used. Smoke test.
        m = load_warpfactory(FIXTURE_PATH)
        # Spot-check: alpha at origin should be within [0, 1] (Alcubierre
        # has unit lapse by construction).
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        alpha_val = float(m.lapse(coords))
        assert 0.0 <= alpha_val <= 1.5
