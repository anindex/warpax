"""Tests for parameter space sweep and transport visualization."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest


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
                points.append(
                    SweepPoint(
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
                    )
                )
        return SweepResult(
            points=points,
            compactness_values=jnp.linspace(0.05, 0.15, nc),
            thickness_values=jnp.linspace(0.3, 0.4, nt),
        )

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
        """Regression (checkpoint corruption in parallel mode): a
        None-compacted points list lets out-of-order (parallel) completion
        scramble to_grids() cell placement after a save/load round trip.
        The list must stay full-length with explicit
        nulls; list position k always maps to cell (k // Nt, k % Nt).
        """
        from warpax.optimization.sweep import SweepPoint, SweepResult

        nc, nt = 2, 3
        total = nc * nt
        filled = (4, 1, 5, 0)

        def _mk(k):
            return SweepPoint(
                compactness=0.05 * (k // nt + 1),
                thickness_ratio=0.3 + 0.1 * (k % nt),
                rho_max=1e-4,
                transport=float(k + 1),
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


class TestEvaluatePoint:
    def test_higher_density_higher_transport(self):
        """Transport increases with density, and a single evaluation returns
        sane fields (nonzero transport/mass, finite residual/tidal)."""
        from warpax.optimization.sweep import _evaluate_point

        r_low = _evaluate_point(
            ansatz="tshell",
            R_1=10.0,
            R_2=20.0,
            rho_0=5e-5,
            n_density=3,
            n_velocity=3,
            n_grid=256,
            n_probes=3,
        )
        assert r_low["transport"] > 0
        assert r_low["mass"] > 0
        assert np.isfinite(r_low["constraint_residual"])
        assert np.isfinite(r_low["tidal"])
        assert isinstance(r_low["ec_feasible"], bool)

        r_high = _evaluate_point(
            ansatz="tshell",
            R_1=10.0,
            R_2=20.0,
            rho_0=2e-4,
            n_density=3,
            n_velocity=3,
            n_grid=256,
            n_probes=3,
        )
        assert r_high["transport"] > r_low["transport"]


class TestSweepDriver:
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
                points.append(
                    SweepPoint(
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
                    )
                )
        return SweepResult(
            points=points,
            compactness_values=jnp.linspace(0.03, 0.15, nc),
            thickness_values=jnp.linspace(0.3, 0.6, nt),
        )

    def test_single_panel_pdf(self, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        from warpax.visualization.phase_diagram import plot_phase_diagram

        sweep = self._make_sweep()
        pdf_path = str(tmp_path / "diag.pdf")
        result = plot_phase_diagram(sweep, save_path=pdf_path)
        import matplotlib.pyplot as plt

        assert not plt.fignum_exists(result.number)

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
        import matplotlib.pyplot as plt

        assert not plt.fignum_exists(result.number)

        from pathlib import Path

        assert Path(pdf_path).exists()


class TestSweepRobustness:
    def _fake_result(self):
        return {
            "rho_max": 1e-4,
            "transport": 0.01,
            "transport_invariant": 0.001,
            "ec_feasible": True,
            "worst_ec_margin": 0.02,
            "constraint_residual": 1e-6,
            "mass": 1.0,
            "tidal": 1e-7,
        }

    def _sweep(self, monkeypatch, tmp_path=None, fail_on=None):
        import warpax.optimization.sweep as sweep_mod

        monkeypatch.delenv("WARPAX_SWEEP_PARALLEL", raising=False)
        calls = []

        def fake_evaluate_point(**kwargs):
            calls.append(kwargs)
            if fail_on is not None and len(calls) == fail_on:
                raise ValueError("boom")
            return self._fake_result()

        monkeypatch.setattr(sweep_mod, "_evaluate_point", fake_evaluate_point)
        return sweep_mod.sweep_transport(
            n_compactness=2,
            n_thickness=3,
            progress=False,
            save_path=None if tmp_path is None else str(tmp_path / "sweep.npz"),
        )

    def test_failed_point_records_nan_not_sentinels(self, monkeypatch):
        with pytest.warns(UserWarning, match="failed"):
            result = self._sweep(monkeypatch, fail_on=2)
        bad = result.points[1]
        assert np.isnan(bad.transport)
        assert np.isnan(bad.transport_invariant)
        assert np.isnan(bad.worst_ec_margin)
        assert bad.ec_feasible is False
        for k, pt in enumerate(result.points):
            if k == 1:
                continue
            assert np.isfinite(pt.transport) and np.isfinite(pt.worst_ec_margin)
            assert pt.ec_feasible is True

    def test_unexpected_exception_propagates(self, monkeypatch):
        import warpax.optimization.sweep as sweep_mod

        def fake_evaluate_point(**kwargs):
            raise TypeError("not a numerical failure")

        monkeypatch.setattr(sweep_mod, "_evaluate_point", fake_evaluate_point)
        with pytest.raises(TypeError):
            sweep_mod.sweep_transport(n_compactness=1, n_thickness=1, progress=False)

    def test_save_is_atomic_and_load_round_trips(self, monkeypatch, tmp_path):
        from warpax.optimization.sweep import SweepResult

        result = self._sweep(monkeypatch, tmp_path=tmp_path)
        names = sorted(p.name for p in tmp_path.iterdir())
        assert names == ["sweep.json", "sweep.npz"]
        loaded = SweepResult.load(str(tmp_path / "sweep.npz"))
        assert loaded.points == result.points

    def test_load_rejects_truncated_json(self, monkeypatch, tmp_path):
        import json

        from warpax.optimization.sweep import SweepResult

        self._sweep(monkeypatch, tmp_path=tmp_path)
        json_path = tmp_path / "sweep.json"
        records = json.loads(json_path.read_text())
        json_path.write_text(json.dumps(records[:-1]))
        with pytest.raises(ValueError, match="expected 6"):
            SweepResult.load(str(tmp_path / "sweep.npz"))
