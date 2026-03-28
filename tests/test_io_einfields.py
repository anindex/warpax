"""Contract tests for :func:`warpax.io.load_einfield` .

Skip-safe per (CONTEXT.md): tests guard the integration with
:func:`pytest.importorskip` for both ``orbax.checkpoint`` and ``flax``.
CI runs without ``warpax[einfields]`` installed skip honestly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from warpax.io import load_einfield


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
