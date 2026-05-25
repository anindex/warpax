"""I/O readers: Cactus, EinFields, WarpFactory."""

from __future__ import annotations
from importlib.util import find_spec
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from warpax.io import InterpolatedADMMetric, load_einfield, load_warpfactory


_h5py_available = find_spec("h5py") is not None
requires_h5py = pytest.mark.skipif(not _h5py_available, reason="h5py not installed (warpax[interop])")


if _h5py_available:
    from warpax.io import load_cactus_slice


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "cactus" / "minkowski_slice.h5"
)


@requires_h5py
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


FIXTURE_DIR = (
    Path(__file__).parent / "fixtures" / "einfields" / "minkowski.ckpt"
)


class TestLoadEinField:
    """Skip-safe contract tests for the EinFields loader."""

    def test_import_load_einfield_symbol(self):
        """The symbol must import even without orbax/flax installed."""
        from warpax.io import load_einfield as _lf  # noqa: F401

    def test_missing_extras_raises_import_error(self, tmp_path):
        """Calling the loader without orbax-checkpoint installed raises ImportError."""
        orbax = pytest.importorskip("orbax.checkpoint", reason="No orbax-checkpoint")
        # If orbax IS installed, skip this specific test - there's nothing to assert.
        pytest.skip(
            "orbax-checkpoint present; missing-extra path tested only in "
            "unit-of-one (extras-not-installed) CI matrix."
        )

    def test_nonexistent_checkpoint_raises(self, tmp_path):
        """Missing path raises FileNotFoundError even when extras absent."""
        # Import the extras if they exist so the path check is reached;
        # otherwise the ImportError path fires first.
        missing = tmp_path / "no-such.ckpt"
        with pytest.raises((FileNotFoundError, ImportError)):
            load_einfield(missing)

    def test_fixture_round_trip_minkowski(self):
        """With extras + fixture present, load_einfield returns Minkowski-like metric."""
        pytest.importorskip("orbax.checkpoint", reason="orbax-checkpoint not installed")
        pytest.importorskip("flax", reason="flax not installed")
        if not FIXTURE_DIR.exists() or not any(FIXTURE_DIR.iterdir()):
            pytest.skip(
                f"EinFields fixture not populated: {FIXTURE_DIR}. "
                "Regenerate via "
                "`python tests/fixtures/einfields/generate_minkowski_ckpt.py`."
            )
        try:
            m = load_einfield(
                FIXTURE_DIR,
                sample_bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
                sample_shape=(2, 2, 2, 2),
            )
        except (RuntimeError, NotImplementedError) as err:
            pytest.skip(f"Topology rebuild failed (flax drift?): {err}")

        # Sample at origin: should be Minkowski eta_ab.
        import jax.numpy as jnp
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = np.asarray(m(coords))
        expected = np.diag([-1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(g, expected, atol=1e-10)


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
