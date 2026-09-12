"""Resumption and fixed-ray convergence regressions for the retained diagnostics."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_anec_symplectic as rays


def result(value):
    return SimpleNamespace(
        symplectic=SimpleNamespace(
            line_integral=value,
            max_abs_g_kk=1e-12,
            null_preserved=True,
            geodesic_complete=True,
        ),
        killing_drift=1e-12,
        method_used="symplectic",
        projection=None,
    )


def test_resume_completed_ray_after_interruption(tmp_path, monkeypatch):
    path = tmp_path / "rays.json"
    provenance = {"source": "version1", "steps": 32768}
    checkpoint = rays.Checkpoint(path, provenance)
    calls = []

    def evaluate(metric, b, span, steps):
        calls.append((b, span, steps))
        if b == 0.2:
            raise RuntimeError("interruption")
        return result(-0.1)

    monkeypatch.setattr(rays, "_rigorous_at", evaluate)
    first = rays._ray_record(None, 0.1, 32.0, checkpoint=checkpoint, name="metric")
    with pytest.raises(RuntimeError, match="interruption"):
        rays._ray_record(None, 0.2, 32.0, checkpoint=checkpoint, name="metric")
    resumed = rays.Checkpoint(path, provenance)
    assert rays._ray_record(None, 0.1, 32.0, checkpoint=resumed, name="metric") == first
    assert len(calls) == 2
    assert len(resumed.data["records"]) == 1
    assert not path.with_suffix(".json.tmp").exists()
    for stale in ({"source": "version2", "steps": 32768}, {"source": "version1", "steps": 65536}):
        with pytest.raises(ValueError, match="Stale"):
            rays.Checkpoint(path, stale)


def test_unstable_selected_ray_extends_without_relaxing_tolerance(tmp_path, monkeypatch):
    values = [-1.1, -1.02, -1.001, -1.00001, -1.0]
    calls = []

    def evaluate(metric, b, span, steps):
        calls.append((b, span, steps))
        return result(values[len(calls) - 1])

    monkeypatch.setattr(rays, "_rigorous_at", evaluate)
    checkpoint = rays.Checkpoint(tmp_path / "rays.json", {"test": True})
    ray = rays.selected_ray_convergence(None, 0.7, 32.0, checkpoint, "metric")
    assert ray["step_stable"]
    assert len(ray["records"]) == 5
    assert calls == [(0.7, 32.0, 16384 * 2**i) for i in range(5)]
    assert ray["relative_tolerance"] == 1e-4
    assert ray["absolute_tolerance"] == 1e-8
    assert ray["finest_change"] <= 1e-8 + 1e-4 * abs(values[-1])
    assert ray["observed_spread"] == max(values) - min(values)
    saved = json.loads(checkpoint.path.read_text())["records"]
    assert next(v for k, v in saved.items() if ":selected:" in k) == ray


def test_unstable_selected_ray_retains_every_failed_level(tmp_path, monkeypatch):
    monkeypatch.setattr(rays, "_rigorous_at", lambda metric, b, span, steps: result(-float(steps)))
    checkpoint = rays.Checkpoint(tmp_path / "rays.json", {"test": True})
    ray = rays.selected_ray_convergence(None, 0.7, 32.0, checkpoint, "metric")
    assert not ray["step_stable"]
    assert len(ray["records"]) == rays.SELECTED_MAX_LEVELS
    saved = json.loads(checkpoint.path.read_text())["records"]
    assert next(v for k, v in saved.items() if ":selected:" in k) == ray
    assert len([k for k in saved if ":ray:" in k]) == rays.SELECTED_MAX_LEVELS


def test_reader_keeps_extended_ladder_and_checks_actual_finest_pair():
    import run_exoticity_anec_convergence as convergence

    records = [
        {
            "steps_per_reference_span": 16384 * 2**i,
            "num_steps": 32768 * 2**i,
            "line_integral": value,
        }
        for i, value in enumerate([-1.1, -1.02, -1.001, -1.00001, -1.0])
    ]
    ray = {"records": records, "step_stable": True}
    summary = convergence.selected_series(ray, "metric")
    assert summary["values"] == [r["line_integral"] for r in records]
    assert len(summary["steps"]) == 5
    records[-1]["line_integral"] = -0.9
    with pytest.raises(AssertionError, match="not step-stable"):
        convergence.selected_series(ray, "metric")


def test_nan_scan_does_not_select_failed_ray_or_hide_unbracketed_value():
    values = [-1.0, -2.0, -1.0, None, -4.0, -0.5]
    eligible = [True, True, True, False, True, True]
    assert rays._coarse_basin(values, eligible) == 1
    assert rays._finite_argmin(values, eligible) == 4
    with pytest.raises(RuntimeError, match="No completed"):
        rays._finite_argmin([None, float("nan")], [False, True])


def test_paper_gate_accepts_disclosed_failed_scan_but_rejects_bad_selected_ray():
    import check_paper_numbers as gate

    row = {
        "b_scan": [0.1, 0.2],
        "line_integral_scan": [-0.1, None],
        "scan_completed": [True, False],
        "scan_null_preserved": [True, False],
        "scan_excluded_indices": [1],
        "scan_excluded_count": 1,
        "scan_excluded_b": [0.2],
        "all_null_preserved": False,
        "all_selected_null_preserved": True,
        "selected_ray_convergence": {
            "minimum_found": {
                "step_stable": True,
                "records": [{"geodesic_complete": True, "null_preserved": True}],
            }
        },
    }
    assert gate.ray_scan_failures(row) == []
    row["scan_excluded_count"] = 0
    assert "wrong excluded scan indices or count" in gate.ray_scan_failures(row)
    row["scan_excluded_count"] = 1
    row["selected_ray_convergence"]["minimum_found"]["records"][0]["geodesic_complete"] = False
    assert any("incomplete" in e for e in gate.ray_scan_failures(row))


def test_spatial_axis_checkpoint_rejects_changed_source_or_partial(tmp_path, monkeypatch):
    import run_exoticity_anec_convergence as extra

    monkeypatch.setattr(extra, "grid_axes_provenance", lambda: {"source": "current"})
    monkeypatch.setattr(extra, "GRID_N", [80])
    monkeypatch.setattr(extra, "ORDER", ["metric"])
    path = tmp_path / "axes.json"
    data = {
        "provenance": {"source": "current"},
        "complete": True,
        "axes": {"80:metric": {"nec_severity": 0.2, "type_iv_frac": 0.1}},
    }
    path.write_text(json.dumps(data))
    assert extra.load_grid_axes(path) == data["axes"]
    data["complete"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incomplete"):
        extra.load_grid_axes(path)
    data["complete"] = True
    data["provenance"] = {"source": "old"}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Stale"):
        extra.load_grid_axes(path)


def test_cached_nonfinite_selected_levels_remain_reportable(tmp_path, monkeypatch):
    monkeypatch.setattr(rays, "_rigorous_at", lambda metric, b, span, steps: result(float("nan")))
    path = tmp_path / "rays.json"
    first = rays.Checkpoint(path, {"test": True})
    ray = rays.selected_ray_convergence(None, 0.7, 32.0, first, "metric")
    assert not ray["step_stable"]
    resumed = rays.Checkpoint(path, {"test": True})
    monkeypatch.setattr(
        rays, "_rigorous_at", lambda *args: pytest.fail("recomputed cached failed ray")
    )
    again = rays.selected_ray_convergence(None, 0.7, 32.0, resumed, "metric")
    assert not again["step_stable"]
    assert len(again["records"]) == rays.SELECTED_MAX_LEVELS
    assert all(r["line_integral"] is None for r in again["records"])
