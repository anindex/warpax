"""Tests for scripts/_json_io.dump_json (strict RFC 8259 JSON policy).

Regression: scripts wrote results/*.json via bare ``json.dump``, which
defaults to ``allow_nan=True`` and emits invalid ``NaN`` / ``Infinity``
literals (RFC 8259 forbids them); strict parsers reject those files.
``dump_json`` converts non-finite floats to ``null`` and dumps with
``allow_nan=False``.
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from _json_io import dump_json, sanitize_nonfinite


def test_sanitize_nonfinite_recursive():
    """NaN/inf are nulled at any nesting depth; finite values untouched."""
    payload = {
        "ok": 1.5,
        "bad": float("nan"),
        "nested": {"inf": float("inf"), "list": [1.0, float("-inf"), "s"]},
        "tuple": (2.0, float("nan")),
        "int": 7,
        "flag": True,
    }
    out = sanitize_nonfinite(payload)
    assert out == {
        "ok": 1.5,
        "bad": None,
        "nested": {"inf": None, "list": [1.0, None, "s"]},
        "tuple": [2.0, None],
        "int": 7,
        "flag": True,
    }


def test_dump_json_emits_strict_json(tmp_path):
    """Output must parse under a strict (no-NaN-constant) JSON parser."""
    path = tmp_path / "out.json"
    dump_json({"a": float("nan"), "b": [float("inf"), 3.0]}, path)
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text

    def _reject(_):
        raise AssertionError("bare non-finite constant leaked into output")

    parsed = json.loads(text, parse_constant=_reject)
    assert parsed == {"a": None, "b": [None, 3.0]}


def test_dump_json_preserves_finite_floats(tmp_path):
    """Finite payloads round-trip exactly."""
    path = tmp_path / "out.json"
    payload = {"x": 0.123456789, "rows": [{"v": -1e-30}, {"v": 2.0}]}
    dump_json(payload, path)
    assert json.loads(path.read_text()) == payload


def test_dump_json_default_str_fallback(tmp_path):
    """``default=str`` (used by several scripts) is forwarded to json.dump."""
    path = tmp_path / "out.json"

    class Odd:
        def __str__(self):
            return "odd"

    dump_json({"obj": Odd(), "nan": float("nan")}, path, default=str)
    assert json.loads(path.read_text()) == {"obj": "odd", "nan": None}


def test_dump_json_numpy_float64_nan(tmp_path):
    """np.float64 NaN (a float subclass) is nulled like a builtin float."""
    import numpy as np

    path = tmp_path / "out.json"
    dump_json({"v": np.float64("nan"), "w": np.float64(1.5)}, path)
    parsed = json.loads(path.read_text())
    assert parsed == {"v": None, "w": 1.5}
    assert math.isfinite(parsed["w"])
