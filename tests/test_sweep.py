"""Tests for parameter space sweep and transport visualization."""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


# -- Data structure tests -----------------------------------------------------

class TestSweepPoint:

    def test_fields_and_dict(self):
        from warpax.optimization.sweep import SweepPoint

        pt = SweepPoint(
            compactness=0.1,
            thickness_ratio=0.5,
            rho_max=1e-4,
            transport=0.007,
            transport_invariant=0.0,
            ec_feasible=True,
            worst_ec_margin=0.01,
            constraint_residual=0.001,
            mass=3.0,
            tidal=1e-6,
        )
        assert pt.compactness == 0.1
        assert pt.ec_feasible is True
        assert pt.transport == 0.007

        d = pt._asdict()
        assert "compactness" in d
        assert d["ec_feasible"] is True


class TestSweepResult:

    def _make_result(self, nc=3, nt=2):
        from warpax.optimization.sweep import SweepPoint, SweepResult

        points = []
        for i in range(nc):
            for j in range(nt):
                points.append(SweepPoint(
                    compactness=0.05 * (i + 1),
                    thickness_ratio=0.3 + 0.1 * j,
                    rho_max=1e-4,
                    transport=0.001 * (i + 1) * (j + 1),
                    transport_invariant=0.0,
                    ec_feasible=(i + j) % 2 == 0,
                    worst_ec_margin=0.01 if (i + j) % 2 == 0 else -0.05,
                    constraint_residual=0.001,
                    mass=2.0 + 0.5 * i,
                    tidal=1e-6 * (j + 1),
                ))
        return SweepResult(
            points=points,
            compactness_values=jnp.linspace(0.05, 0.15, nc),
            thickness_values=jnp.linspace(0.3, 0.4, nt),
        )

    def test_to_grids_shape(self):
        result = self._make_result(nc=3, nt=2)
        grids = result.to_grids()
        assert grids["transport"].shape == (3, 2)
        assert grids["ec_feasible"].shape == (3, 2)
        assert grids["mass"].shape == (3, 2)

    def test_save_load_round_trip(self, tmp_path):
        result = self._make_result(nc=3, nt=2)
        save_file = str(tmp_path / "sweep.npz")
        result.save(save_file)

        from warpax.optimization.sweep import SweepResult
        loaded = SweepResult.load(save_file)
        assert len(loaded.points) == len(result.points)
        assert loaded.points[0].compactness == result.points[0].compactness
        assert np.allclose(
            np.asarray(loaded.compactness_values),
            np.asarray(result.compactness_values),
        )

    def test_out_of_order_checkpoint_preserves_cell_placement(self, tmp_path):
        """Regression (checkpoint corruption in parallel mode): checkpoints
        previously serialized a None-compacted points list, so out-of-order
        (parallel) completion scrambled to_grids() cell placement after a
        save/load round trip. The list must stay full-length with explicit
        nulls; list position k always maps to cell (k // Nt, k % Nt).
        """
        from warpax.optimization.sweep import SweepPoint, SweepResult

        nc, nt = 2, 3
        total = nc * nt
        filled = (4, 1, 5, 0)  # parallel completion order with holes (2, 3)

        def _mk(k):
            return SweepPoint(
                compactness=0.05 * (k // nt + 1),
                thickness_ratio=0.3 + 0.1 * (k % nt),
                rho_max=1e-4,
                transport=float(k + 1),  # unique marker per grid cell
                transport_invariant=0.0,
                ec_feasible=True,
                worst_ec_margin=0.01,
                constraint_residual=0.001,
                mass=2.0,
                tidal=1e-6,
            )

        points: list = [None] * total
        for k in filled:
            points[k] = _mk(k)

        partial = SweepResult(
            points=points,
            compactness_values=jnp.linspace(0.05, 0.10, nc),
            thickness_values=jnp.linspace(0.3, 0.5, nt),
        )
        save_file = str(tmp_path / "checkpoint.npz")
        partial.save(save_file)

        loaded = SweepResult.load(save_file)
        assert len(loaded.points) == total
        for k in range(total):
            if k in filled:
                assert loaded.points[k] is not None
                assert loaded.points[k].transport == float(k + 1)
            else:
                assert loaded.points[k] is None

        grids = loaded.to_grids()
        for k in filled:
            i, j = divmod(k, nt)
            assert grids["transport"][i, j] == float(k + 1), (
                f"point {k} placed in wrong cell after checkpoint round trip"
            )
        for k in range(total):
            if k not in filled:
                i, j = divmod(k, nt)
                assert np.isnan(grids["transport"][i, j])


# -- Density scaling ----------------------------------------------------------

class TestDensityScaling:

    def test_positive_density(self):
        from warpax.optimization.sweep import _rho_from_compactness

        rho = _rho_from_compactness(0.1, R_1=10.0, R_2=20.0)
        assert rho > 0

    def test_linear_in_compactness(self):
        """rho_0 scales linearly with compactness at fixed geometry."""
        from warpax.optimization.sweep import _rho_from_compactness

        rho_low = _rho_from_compactness(0.05, R_1=10.0, R_2=20.0)
        rho_high = _rho_from_compactness(0.10, R_1=10.0, R_2=20.0)
        assert abs(rho_high / rho_low - 2.0) < 0.01

    def test_thinner_shell_higher_density(self):
        from warpax.optimization.sweep import _rho_from_compactness

        rho_thick = _rho_from_compactness(0.1, R_1=10.0, R_2=20.0)
        rho_thin = _rho_from_compactness(0.1, R_1=15.0, R_2=20.0)
        assert rho_thin > rho_thick


# -- Single-point evaluation --------------------------------------------------

class TestEvaluatePoint:

    def test_higher_density_higher_transport(self):
        """Transport increases with density, and a single evaluation returns
        sane fields (nonzero transport/mass, finite residual/tidal)."""
        from warpax.optimization.sweep import _evaluate_point

        r_low = _evaluate_point(
            ansatz="tshell", R_1=10.0, R_2=20.0, rho_0=5e-5,
            n_density=4, n_velocity=4, n_grid=256, n_probes=3, n_ec_starts=2,
        )
        # Sanity of a single point (folded from test_returns_nonzero_transport)
        assert r_low["transport"] > 0
        assert r_low["mass"] > 0
        assert np.isfinite(r_low["constraint_residual"])
        assert np.isfinite(r_low["tidal"])
        assert isinstance(r_low["ec_feasible"], bool)

        r_high = _evaluate_point(
            ansatz="tshell", R_1=10.0, R_2=20.0, rho_0=2e-4,
            n_density=4, n_velocity=4, n_grid=256, n_probes=3, n_ec_starts=2,
        )
        assert r_high["transport"] > r_low["transport"]


# -- Sweep driver -------------------------------------------------------------

class TestSweepDriver:

    @pytest.mark.slow
    def test_minimal_sweep(self):
        """2x2 sweep completes and returns valid results."""
        from warpax.optimization.sweep import sweep_transport

        result = sweep_transport(
            ansatz="tshell",
            compactness_range=(0.05, 0.10),
            thickness_range=(0.4, 0.6),
            n_compactness=2,
            n_thickness=2,
            n_density=3,
            n_velocity=3,
            n_grid=256,
            n_probes=3,
            n_ec_starts=2,
            progress=False,
        )

        assert len(result.points) == 4
        assert result.compactness_values.shape == (2,)
        assert result.thickness_values.shape == (2,)

        for pt in result.points:
            assert isinstance(pt.ec_feasible, bool)
            assert np.isfinite(pt.compactness)
            assert np.isfinite(pt.thickness_ratio)
            assert pt.transport > 0


# -- Visualization ------------------------------------------------------------

class TestPhaseDiagramPlot:

    def _make_sweep(self):
        from warpax.optimization.sweep import SweepPoint, SweepResult

        nc, nt = 5, 4
        points = []
        for i in range(nc):
            for j in range(nt):
                c = 0.03 * (i + 1)
                t_ratio = 0.3 + 0.1 * j
                transport = 0.01 * np.sin(i * 0.5) * np.cos(j * 0.5)
                feas = (i + j) < 5
                points.append(SweepPoint(
                    compactness=c,
                    thickness_ratio=t_ratio,
                    rho_max=1e-4 * (i + 1),
                    transport=max(transport, 0.0),
                    transport_invariant=0.0,
                    ec_feasible=feas,
                    worst_ec_margin=0.01 if feas else -0.05,
                    constraint_residual=0.001 * (i + 1),
                    mass=2.0 + 0.3 * i,
                    tidal=1e-6 * (j + 1),
                ))
        return SweepResult(
            points=points,
            compactness_values=jnp.linspace(0.03, 0.15, nc),
            thickness_values=jnp.linspace(0.3, 0.6, nt),
        )

    def test_single_panel(self):
        import matplotlib
        matplotlib.use("Agg")
        from warpax.visualization.phase_diagram import plot_phase_diagram

        sweep = self._make_sweep()
        fig = plot_phase_diagram(sweep)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_single_panel_pdf(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from warpax.visualization.phase_diagram import plot_phase_diagram

        sweep = self._make_sweep()
        pdf_path = str(tmp_path / "diag.pdf")
        result = plot_phase_diagram(sweep, save_path=pdf_path)
        # When save_path is given, figure is saved and closed; returns None.
        assert result is None

        from pathlib import Path
        assert Path(pdf_path).exists()
        assert Path(pdf_path).stat().st_size > 0

    def test_summary_panel_count(self):
        import matplotlib
        matplotlib.use("Agg")
        from warpax.visualization.phase_diagram import plot_phase_summary

        sweep = self._make_sweep()
        fig = plot_phase_summary(sweep)
        assert fig is not None
        assert len(fig.axes) >= 4
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_summary_pdf(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from warpax.visualization.phase_diagram import plot_phase_summary

        sweep = self._make_sweep()
        pdf_path = str(tmp_path / "summary.pdf")
        result = plot_phase_summary(sweep, save_path=pdf_path)
        assert result is None

        from pathlib import Path
        assert Path(pdf_path).exists()
