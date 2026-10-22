"""I/O readers: Cactus, EinFields, WarpFactory.

``WARPFACTORY_FIXTURE_PATH`` is named distinctly from the cactus ``FIXTURE_PATH``:
a bare ``FIXTURE_PATH`` here would shadow the cactus one at module load, so the
cactus tests requiring h5py would silently try to open the .mat file ("file
signature not found").
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from warpax.io import load_einfield, load_warpfactory

_h5py_available = find_spec("h5py") is not None
requires_h5py = pytest.mark.skipif(
    not _h5py_available, reason="h5py not installed (warpax[interop])"
)


if _h5py_available:
    from warpax.io import load_cactus_slice


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cactus" / "minkowski_slice.h5"


@requires_h5py
class TestLoadCactusSlice:
    """End-to-end contract tests for the Cactus / ET HDF5 reader."""

    def test_name_encodes_iteration_and_timelevel(self):
        m = load_cactus_slice(FIXTURE_PATH, iteration=0, timelevel=0)
        assert "cactus" in m.name()
        assert "it0" in m.name()
        assert "tl0" in m.name()

    def test_minkowski_round_trip(self):
        """Hand-synth fixture: g(origin) == eta_{ab} exactly."""
        m = load_cactus_slice(FIXTURE_PATH)
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = np.asarray(m(coords))
        expected = np.diag([-1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(g, expected, atol=1e-10)
        assert float(np.linalg.det(g)) < 0.0

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_cactus_slice("nonexistent.h5")

    def test_unknown_iteration_raises(self):
        with pytest.raises(ValueError, match="not found"):
            load_cactus_slice(FIXTURE_PATH, iteration=99, timelevel=0)


@requires_h5py
class TestCactusShiftConvention:
    """Regression (ET shift convention): Einstein Toolkit ADMBase
    betax/betay/betaz are CONTRAVARIANT beta^i. Treating them as covariant
    beta_i and raising the index with gamma^{ij} corrupts the shift for any
    non-flat spatial metric.
    """

    @staticmethod
    def _write_synthetic_slice(path):
        """gamma = diag(4, 1, 1), alpha = 1, beta^x = 0.2 (uniform)."""
        import h5py

        n = 4
        ones = np.ones((n, n, n), dtype=np.float64)
        zeros = np.zeros((n, n, n), dtype=np.float64)
        with h5py.File(path, "w") as f:
            grp = f.create_group("ITERATION=0/TIMELEVEL=0")
            grp["alp"] = ones
            grp["betax"] = 0.2 * ones
            grp["betay"] = zeros
            grp["betaz"] = zeros
            grp["gxx"] = 4.0 * ones
            grp["gxy"] = zeros
            grp["gxz"] = zeros
            grp["gyy"] = ones
            grp["gyz"] = zeros
            grp["gzz"] = ones
            grp.attrs["time"] = 0.0
            for axis in ("x", "y", "z"):
                grp.attrs[f"{axis}0"] = -1.0
                grp.attrs[f"d{axis}"] = 2.0 / (n - 1)

    @staticmethod
    def _write_slice(path, alp, gzz):
        """Uniform 4^3 slice with lapse ``alp``, gamma = diag(1, 1, ``gzz``), zero shift."""
        import h5py

        ones = np.ones((4, 4, 4), dtype=np.float64)
        zeros = np.zeros((4, 4, 4), dtype=np.float64)
        with h5py.File(path, "w") as f:
            grp = f.create_group("ITERATION=0/TIMELEVEL=0")
            grp["alp"] = alp * ones
            for key in ("betax", "betay", "betaz", "gxy", "gxz", "gyz"):
                grp[key] = zeros
            grp["gxx"] = ones
            grp["gyy"] = ones
            grp["gzz"] = gzz * ones

    def test_negative_lapse_raises(self, tmp_path):
        """alp = -1 everywhere (64 points, first index (0, 0, 0, 0)) fails the lapse check
        before the spatial metric is examined, even though gzz = -1 is also invalid."""
        path = tmp_path / "bad_lapse.h5"
        self._write_slice(path, alp=-1.0, gzz=-1.0)
        with pytest.raises(
            ValueError, match=r"lapse alpha .* 64 grid point\(s\); first at index \(0, 0, 0, 0\)"
        ):
            load_cactus_slice(path)

    def test_non_spd_spatial_metric_raises(self, tmp_path):
        """alp = 1 with gamma = diag(1, 1, -1): min eigenvalue -1 at all 64 points fails
        the positive-definiteness check on the spatial metric."""
        path = tmp_path / "bad_gamma.h5"
        self._write_slice(path, alp=1.0, gzz=-1.0)
        with pytest.raises(
            ValueError, match=r"spatial metric .* not positive definite at 64 grid point\(s\)"
        ):
            load_cactus_slice(path)

    def test_contravariant_shift_and_four_metric(self, tmp_path):
        """beta^x must be the raw ET value 0.2 (NOT gamma^{xx} * 0.2 = 0.05);
        the 4-metric then has g_{0x} = gamma_xx beta^x = 4 * 0.2 = 0.8 and
        g_00 = -(alpha^2 - gamma_ij beta^i beta^j) = -(1 - 4*0.04) = -0.84."""
        slice_path = tmp_path / "tilted_slice.h5"
        self._write_synthetic_slice(slice_path)

        m = load_cactus_slice(slice_path)
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])

        beta = np.asarray(m.shift(coords))
        np.testing.assert_allclose(beta, [0.2, 0.0, 0.0], atol=1e-12)

        g = np.asarray(m(coords))
        np.testing.assert_allclose(g[0, 1], 0.8, atol=1e-12)
        np.testing.assert_allclose(g[1, 0], 0.8, atol=1e-12)
        np.testing.assert_allclose(g[0, 0], -0.84, atol=1e-12)
        np.testing.assert_allclose(g[1, 1], 4.0, atol=1e-12)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "einfields" / "minkowski.ckpt"


class TestLoadEinField:
    """Skip-safe contract tests for the EinFields loader."""

    def test_nonexistent_checkpoint_raises(self, tmp_path):
        """Missing path raises FileNotFoundError even when extras absent
        (if the extras are missing the ImportError path fires first)."""
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

        import jax.numpy as jnp

        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = np.asarray(m(coords))
        expected = np.diag([-1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(g, expected, atol=1e-10)


WARPFACTORY_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "warpfactory" / "alcubierre.mat"


class TestLoadWarpFactory:
    """End-to-end contract tests for the WarpFactory reader."""

    def test_name_reflects_source_type(self):
        m = load_warpfactory(WARPFACTORY_FIXTURE_PATH)
        assert "alcubierre" in m.name().lower()

    def test_full_metric_has_lorentzian_signature(self):
        m = load_warpfactory(WARPFACTORY_FIXTURE_PATH)
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = m(coords)
        assert g.shape == (4, 4)
        assert float(jnp.linalg.det(g)) < 0.0
        assert float(g[1, 1]) > 0.0
        assert float(g[2, 2]) > 0.0
        assert float(g[3, 3]) > 0.0

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_warpfactory("nonexistent-file.mat")

    def test_v7_scipy_path_loads_unit_lapse(self):
        """The v7 (non-HDF5) fixture goes through the scipy path; Alcubierre
        has unit lapse by construction, so alpha(origin) must be exactly 1."""
        m = load_warpfactory(WARPFACTORY_FIXTURE_PATH)
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        alpha_val = float(m.lapse(coords))
        np.testing.assert_allclose(alpha_val, 1.0, atol=1e-12)


def _lapse_only_metric(nx: int = 33, nt: int = 4, interp_method: str = "cubic"):
    """alpha = 1 + x^2 on nx points over [-8, 8]; t, y, z carry nt/4/4 constant samples."""
    import jax.numpy as jnp

    from warpax.geometry import GridSpec
    from warpax.io import InterpolatedADMMetric

    x = np.linspace(-8.0, 8.0, nx)
    alpha = np.broadcast_to((1.0 + x**2)[None, :, None, None], (nt, nx, 4, 4))
    beta = np.zeros((nt, nx, 4, 4, 3))
    gamma = np.broadcast_to(np.eye(3), (nt, nx, 4, 4, 3, 3))
    spec = GridSpec(
        bounds=[(0.0, 1.0), (-8.0, 8.0), (-1.0, 1.0), (-1.0, 1.0)], shape=(nt, nx, 4, 4)
    )
    return InterpolatedADMMetric(
        alpha_grid=jnp.asarray(alpha),
        beta_grid=jnp.asarray(beta),
        gamma_grid=jnp.asarray(gamma),
        grid_spec=spec,
        interp_method=interp_method,
    )


class TestCubicInterpolation:
    """Uniform cubic B-spline path of InterpolatedADMMetric (package A15)."""

    def test_cubic_recovers_scalar_curvature(self):
        """ds^2 = -alpha^2 dt^2 + dx^2 + dy^2 + dz^2 has R = -2 alpha''/alpha, so for
        alpha = 1 + x^2 the value at x = 0.5 is -4/1.25 = -3.2."""
        import jax.numpy as jnp

        from warpax.geometry import compute_curvature_chain

        m = _lapse_only_metric(interp_method="cubic")
        res = compute_curvature_chain(m, jnp.array([0.0, 0.5, 0.0, 0.0]))
        np.testing.assert_allclose(float(res.ricci_scalar), -3.2, atol=1e-2)

    def test_linear_gives_zero_curvature(self):
        """Linear interpolation is affine inside every cell, so alpha'' = 0 and R = 0."""
        import jax.numpy as jnp

        from warpax.geometry import compute_curvature_chain

        m = _lapse_only_metric(interp_method="linear")
        res = compute_curvature_chain(m, jnp.array([0.0, 0.5, 0.0, 0.0]))
        np.testing.assert_allclose(float(res.ricci_scalar), 0.0, atol=1e-8)

    def test_cubic_reproduces_knots(self):
        """The prefiltered spline interpolates: lapse at every x knot equals 1 + x^2 to 1e-12."""
        import jax
        import jax.numpy as jnp

        m = _lapse_only_metric(interp_method="cubic")
        x = jnp.linspace(-8.0, 8.0, 33)
        vals = jax.vmap(lambda xi: m.lapse(jnp.array([0.0, xi, 0.0, 0.0])))(x)
        np.testing.assert_allclose(np.asarray(vals), 1.0 + np.asarray(x) ** 2, atol=1e-12)

    def test_cubic_second_derivative_continuous_at_cell_face(self):
        """A cubic B-spline is C2, so d^2 alpha/dx^2 just left and right of the knot x = 1
        differ by less than 1e-8."""
        import jax
        import jax.numpy as jnp

        m = _lapse_only_metric(interp_method="cubic")

        def d2(xv):
            f = lambda s: m.lapse(jnp.array([0.0, s, 0.0, 0.0]))
            return jax.grad(jax.grad(f))(xv)

        jump = float(d2(jnp.float64(1.0 - 1e-7)) - d2(jnp.float64(1.0 + 1e-7)))
        assert abs(jump) < 1e-8

    @pytest.mark.parametrize("interp_method", ["linear", "cubic"])
    def test_single_time_slice_is_finite(self, interp_method):
        """Nt = 1: the singleton axis has zero span, index 0 is used, and values stay finite."""
        import jax.numpy as jnp

        m = _lapse_only_metric(nt=1, interp_method=interp_method)
        coords = jnp.array([0.3, 0.5, 0.1, -0.2])
        assert np.isfinite(float(m.lapse(coords)))
        assert np.all(np.isfinite(np.asarray(m(coords))))

    def test_linear_nan_channel_does_not_poison_others(self):
        """A NaN planted in gamma[..., 0, 1] leaves gamma[..., 0, 0] = 1 under linear
        interpolation because channels are interpolated independently. The grid is fed
        to the interpolator directly, since InterpolatedADMMetric rejects non-finite grids."""
        import jax.numpy as jnp

        from warpax.geometry import GridSpec
        from warpax.io._interpolation import _interpolate_tensor

        gamma = np.broadcast_to(np.eye(3), (4, 4, 4, 4, 3, 3)).copy()
        gamma[..., 0, 1] = np.nan
        spec = GridSpec(bounds=[(0.0, 1.0)] * 4, shape=(4, 4, 4, 4))
        g = _interpolate_tensor(jnp.asarray(gamma), jnp.array([0.5, 0.5, 0.5, 0.5]), spec, "linear")
        assert float(g[0, 0]) == 1.0
        assert np.isnan(float(g[0, 1]))

    def test_non_finite_grid_raises(self):
        """A NaN in gamma[..., 0, 1] at every grid point of a 4^4 grid is rejected with
        count 256 and first index (0, 0, 0, 0)."""
        import jax.numpy as jnp

        from warpax.geometry import GridSpec
        from warpax.io import InterpolatedADMMetric

        gamma = np.broadcast_to(np.eye(3), (4, 4, 4, 4, 3, 3)).copy()
        gamma[..., 0, 1] = np.nan
        spec = GridSpec(bounds=[(0.0, 1.0)] * 4, shape=(4, 4, 4, 4))
        with pytest.raises(
            ValueError, match=r"non-finite at 256 grid point\(s\); first at index \(0, 0, 0, 0\)"
        ):
            InterpolatedADMMetric(
                alpha_grid=jnp.ones((4, 4, 4, 4)),
                beta_grid=jnp.zeros((4, 4, 4, 4, 3)),
                gamma_grid=jnp.asarray(gamma),
                grid_spec=spec,
            )

    def test_integer_coords_keep_fractional_bounds(self):
        """x in [-8, 8] on 33 points has spacing 0.5, so x = 1 (integer coords) sits at
        fractional index 18 and the lapse must equal 1 + 1^2 = 2, not a truncated-bounds value."""
        import jax.numpy as jnp

        m = _lapse_only_metric(interp_method="cubic")
        val = float(m.lapse(jnp.array([0, 1, 0, 0])))
        np.testing.assert_allclose(val, 2.0, atol=1e-12)

    @pytest.mark.parametrize("nt", [2, 3])
    def test_short_axis_reproduces_knots(self, nt):
        """Samples (1, 2, ..., nt) along a time axis with nt = 2 or 3 points: the mirror-row
        prefilter must reproduce the knots exactly (unfiltered coefficients would smooth
        (1, 2, 3) into (4/3, 2, 8/3))."""
        import jax.numpy as jnp

        from warpax.geometry import GridSpec
        from warpax.io import InterpolatedADMMetric

        alpha = np.broadcast_to(
            np.arange(1, nt + 1, dtype=float)[:, None, None, None], (nt, 4, 4, 4)
        )
        spec = GridSpec(bounds=[(0.0, 1.0)] * 4, shape=(nt, 4, 4, 4))
        m = InterpolatedADMMetric(
            alpha_grid=jnp.asarray(alpha),
            beta_grid=jnp.zeros((nt, 4, 4, 4, 3)),
            gamma_grid=jnp.asarray(np.broadcast_to(np.eye(3), (nt, 4, 4, 4, 3, 3))),
            grid_spec=spec,
            interp_method="cubic",
        )
        ts = np.linspace(0.0, 1.0, nt)
        vals = [float(m.lapse(jnp.array([t, 0.5, 0.5, 0.5]))) for t in ts]
        np.testing.assert_allclose(vals, np.arange(1, nt + 1, dtype=float), atol=1e-12)


class TestAdmFromCovariantGrid:
    """Shared ADM decomposition with validation (package B5)."""

    @staticmethod
    def _write_mat(path, g_full):
        from scipy.io import savemat

        n = g_full.shape[2:]
        coords = {k: np.linspace(-1.0, 1.0, s) for k, s in zip("txyz", n, strict=True)}
        savemat(str(path), {"metric": {"tensor": g_full, "coords": coords, "type": "Synthetic"}})

    def test_minkowski_decomposes_exactly(self):
        from warpax.io._adm_grid import adm_from_covariant_grid

        g = np.broadcast_to(np.diag([-1.0, 1.0, 1.0, 1.0]), (2, 3, 3, 3, 4, 4))
        alpha, beta, gamma = adm_from_covariant_grid(g)
        assert alpha.shape == (2, 3, 3, 3)
        assert np.array_equal(alpha, np.ones((2, 3, 3, 3)))
        assert np.array_equal(beta, np.zeros((2, 3, 3, 3, 3)))
        assert np.array_equal(gamma, np.broadcast_to(np.eye(3), (2, 3, 3, 3, 3, 3)))

    def test_matches_previous_inline_decomposition(self):
        """Shifted, non-flat grid: results equal an inline inv + einsum + sqrt decomposition bitwise."""
        from warpax.io._adm_grid import adm_from_covariant_grid

        rng = np.random.default_rng(0)
        n = (2, 3, 3, 3)
        a = rng.normal(size=(*n, 3, 3))
        gamma = np.eye(3) + 0.1 * (a + np.swapaxes(a, -1, -2))
        beta_lower = 0.1 * rng.normal(size=(*n, 3))
        g = np.zeros((*n, 4, 4))
        g[..., 1:, 1:] = gamma
        g[..., 0, 1:] = beta_lower
        g[..., 1:, 0] = beta_lower
        gamma_uu = np.linalg.inv(gamma)
        beta_upper = np.einsum("...ij,...j->...i", gamma_uu, beta_lower)
        g[..., 0, 0] = np.einsum("...i,...i->...", beta_lower, beta_upper) - 1.5**2
        alpha, beta, gamma_out = adm_from_covariant_grid(g)
        alpha_ref = np.sqrt(np.einsum("...i,...i->...", beta_lower, beta_upper) - g[..., 0, 0])
        assert np.array_equal(alpha, alpha_ref)
        assert np.array_equal(beta, beta_upper)
        assert np.array_equal(gamma_out, gamma)

    def test_no_valid_lapse_raises(self):
        from warpax.io._adm_grid import adm_from_covariant_grid

        g = np.broadcast_to(np.diag([-1.0, 1.0, 1.0, 1.0]), (2, 2, 2, 2, 4, 4)).copy()
        g[1, 0, 1, 0, 0, 0] = 1.0
        with pytest.raises(ValueError, match=r"1 grid point\(s\); first at index \(1, 0, 1, 0\)"):
            adm_from_covariant_grid(g)

    def test_non_finite_lapse_raises(self):
        from warpax.io._adm_grid import adm_from_covariant_grid

        g = np.broadcast_to(np.diag([-1.0, 1.0, 1.0, 1.0]), (2, 2, 2, 2, 4, 4)).copy()
        g[0, 0, 0, 0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            adm_from_covariant_grid(g)

    def test_non_spd_gamma_raises(self):
        from warpax.io._adm_grid import adm_from_covariant_grid

        g = np.broadcast_to(np.diag([-1.0, 1.0, 1.0, 1.0]), (2, 2, 2, 2, 4, 4)).copy()
        g[0, 1, 1, 1, 2, 2] = -1.0
        g[1, 1, 1, 1, 3, 3] = 0.0
        with pytest.raises(ValueError, match=r"not positive definite at 2 grid point\(s\)"):
            adm_from_covariant_grid(g)

    def test_asymmetric_gamma_raises(self):
        from warpax.io._adm_grid import adm_from_covariant_grid

        g = np.broadcast_to(np.diag([-1.0, 1.0, 1.0, 1.0]), (2, 2, 2, 2, 4, 4)).copy()
        g[0, 0, 0, 0, 1, 2] = 0.5
        with pytest.raises(ValueError, match="not symmetric"):
            adm_from_covariant_grid(g)

    def test_warpfactory_loader_rejects_positive_g00(self, tmp_path):
        pytest.importorskip("scipy.io")
        g_full = np.broadcast_to(
            np.diag([1.0, 1.0, 1.0, 1.0])[:, :, None, None, None, None], (4, 4, 2, 3, 3, 3)
        )
        path = tmp_path / "bad_lapse.mat"
        self._write_mat(path, np.ascontiguousarray(g_full))
        with pytest.raises(ValueError, match="no valid lapse"):
            load_warpfactory(path)

    def test_warpfactory_loader_accepts_minkowski(self, tmp_path):
        pytest.importorskip("scipy.io")
        g_full = np.broadcast_to(
            np.diag([-1.0, 1.0, 1.0, 1.0])[:, :, None, None, None, None], (4, 4, 2, 3, 3, 3)
        )
        path = tmp_path / "minkowski.mat"
        self._write_mat(path, np.ascontiguousarray(g_full))
        m = load_warpfactory(path)
        np.testing.assert_allclose(float(m.lapse(jnp.zeros(4))), 1.0, atol=1e-12)


class TestEinfieldsBatchedSampling:
    """Vectorized grid sampling equals the previous per-point loop exactly."""

    @staticmethod
    def _loop_reference(network_call, sample_bounds, sample_shape):
        axes = [
            np.linspace(lo, hi, n) for (lo, hi), n in zip(sample_bounds, sample_shape, strict=True)
        ]
        coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 4)
        out = np.zeros((coords.shape[0], 4, 4), dtype=np.float64)
        for i, c in enumerate(coords):
            out[i] = np.asarray(network_call(jnp.asarray(c)), dtype=np.float64)
        return out.reshape(*sample_shape, 4, 4)

    def test_minkowski_fixture_constant_call(self):
        ocp = pytest.importorskip("orbax.checkpoint", reason="orbax-checkpoint not installed")
        if not FIXTURE_DIR.exists() or not any(FIXTURE_DIR.iterdir()):
            pytest.skip(f"EinFields fixture not populated: {FIXTURE_DIR}.")
        from warpax.io.einfields import _sample_on_grid

        state = ocp.StandardCheckpointer().restore(FIXTURE_DIR)
        eta_arr = np.asarray(state["eta_metric"], dtype=np.float64)

        def network_call(_coords):
            return jnp.asarray(eta_arr)

        bounds = ((-1.0, 1.0), (-2.0, 2.0), (-1.0, 1.0), (-3.0, 3.0))
        shape = (2, 3, 2, 3)
        got = _sample_on_grid(network_call, bounds, shape)
        ref = self._loop_reference(network_call, bounds, shape)
        assert got.shape == (*shape, 4, 4)
        assert np.array_equal(got, ref)

    def test_coordinate_dependent_call(self):
        from warpax.io.einfields import _sample_on_grid

        def network_call(c):
            return jnp.diag(jnp.array([-1.0 - c[1] ** 2, 1.0 + c[0], 1.0 + c[2] * c[3], 2.0]))

        bounds = ((-1.0, 1.0), (-2.0, 2.0), (-1.0, 1.0), (-3.0, 3.0))
        shape = (2, 3, 2, 3)
        got = _sample_on_grid(network_call, bounds, shape)
        ref = self._loop_reference(network_call, bounds, shape)
        assert np.array_equal(got, ref)
