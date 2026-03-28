"""Contract tests for :func:`warpax.io.load_cactus_slice` ."""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("h5py", reason="h5py not installed (warpax[interop])")

from warpax.io import InterpolatedADMMetric, load_cactus_slice


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "cactus" / "minkowski_slice.h5"
)


class TestLoadCactusSlice:
    """End-to-end contract tests for the Cactus / ET HDF5 reader."""

    def test_fixture_exists(self):
        assert FIXTURE_PATH.exists(), (
            f"Cactus fixture not committed: {FIXTURE_PATH}. Regenerate via "
            "`python tests/fixtures/cactus/generate_minkowski_slice.py`."
        )

    def test_returns_interpolated_adm_metric(self):
        m = load_cactus_slice(FIXTURE_PATH)
        assert isinstance(m, InterpolatedADMMetric)

    def test_name_encodes_iteration_and_timelevel(self):
        m = load_cactus_slice(FIXTURE_PATH, iteration=0, timelevel=0)
        assert "cactus" in m.name()
        assert "it0" in m.name()
        assert "tl0" in m.name()

    def test_grid_shape_matches_fixture(self):
        """Single-timelevel wrapper: Nt=1; (nx, ny, nz) all 8 per fixture."""
        m = load_cactus_slice(FIXTURE_PATH)
        assert m.alpha_grid.shape == (1, 8, 8, 8)
        assert m.beta_grid.shape == (1, 8, 8, 8, 3)
        assert m.gamma_grid.shape == (1, 8, 8, 8, 3, 3)

    def test_minkowski_round_trip(self):
        """Hand-synth fixture: g(origin) == eta_{ab} exactly."""
        m = load_cactus_slice(FIXTURE_PATH)
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = np.asarray(m(coords))
        expected = np.diag([-1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(g, expected, atol=1e-10)

    def test_lorentzian_signature_det_negative(self):
        m = load_cactus_slice(FIXTURE_PATH)
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = m(coords)
        assert float(jnp.linalg.det(g)) < 0.0

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_cactus_slice("nonexistent.h5")

    def test_unknown_iteration_raises(self):
        with pytest.raises(ValueError, match="not found"):
            load_cactus_slice(FIXTURE_PATH, iteration=99, timelevel=0)
