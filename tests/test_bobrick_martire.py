"""Regression tests for warpax.classify.bobrick_martire .

Pins the Bobrick-Martire class for each canonical metric:

- Class I : Killing-field structure + no matter (Minkowski, Schwarzschild)
- Class II : Alcubierre-family shape-function-supported bubble (Alcubierre, Rodal)
- Class III : Matter-shell / junction-structured (WarpShell)

Also exercises determinism (``bobrick_martire`` called twice returns the same
integer class for each canonical metric).
"""
from __future__ import annotations

import pytest

from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric, SchwarzschildMetric
from warpax.classify import ClassifiedMetric, bobrick_martire
from warpax.metrics import RodalMetric, WarpShellMetric


class TestBobrickMartire:
    """Pin Bobrick-Martire class for each canonical metric .

    Class I : Killing-field structure + no matter (Minkowski, Schwarzschild)
    Class II : Alcubierre-family shape-function-supported (Alcubierre, Rodal)
    Class III : Matter-shell / junction-structured (WarpShell)
    """

    def test_minkowski_is_class_i(self):
        result = bobrick_martire(MinkowskiMetric())
        assert isinstance(result, ClassifiedMetric)
        assert result.bobrick_class == 1
        assert result.stationary is True
        assert result.shape_function_supported is False

    def test_schwarzschild_is_class_i(self):
        result = bobrick_martire(SchwarzschildMetric(M=1.0))
        assert result.bobrick_class == 1
        assert result.stationary is True

    def test_alcubierre_is_class_ii(self):
        result = bobrick_martire(AlcubierreMetric())
        assert result.bobrick_class == 2
        assert result.shape_function_supported is True

    def test_rodal_is_class_ii(self):
        result = bobrick_martire(RodalMetric())
        assert result.bobrick_class == 2
        assert result.shape_function_supported is True

    def test_warpshell_is_class_iii(self):
        result = bobrick_martire(WarpShellMetric())
        assert result.bobrick_class == 3
        assert result.shape_function_supported is True

    @pytest.mark.parametrize(
        "metric_cls,expected_class",
        [
            (MinkowskiMetric, 1),
            (SchwarzschildMetric, 1),
            (AlcubierreMetric, 2),
            (RodalMetric, 2),
            (WarpShellMetric, 3),
        ],
    )
    def test_determinism(self, metric_cls, expected_class):
        """Classifier returns identical class across repeated calls on each metric."""
        metric = (
            metric_cls(M=1.0)
            if metric_cls is SchwarzschildMetric
            else metric_cls()
        )
        r1 = bobrick_martire(metric)
        r2 = bobrick_martire(metric)
        assert r1.bobrick_class == r2.bobrick_class == expected_class
        assert r1.stationary is r2.stationary
        assert r1.comoving_fluid is r2.comoving_fluid
        assert r1.shape_function_supported is r2.shape_function_supported

    def test_classified_metric_is_namedtuple(self):
        """``ClassifiedMetric`` exposes named attributes (for API consumers)."""
        result = bobrick_martire(MinkowskiMetric())
        # NamedTuple: field access + tuple-unpack both work.
        c, st, cf, sfs = result
        assert c == result.bobrick_class
        assert st is result.stationary
        assert cf is result.comoving_fluid
        assert sfs is result.shape_function_supported
