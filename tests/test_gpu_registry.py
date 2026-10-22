"""Every GPU xfail registry node id points at a test that still exists."""

from __future__ import annotations

import os

import pytest

from ._gpu_xfail_registry import EXPECTED_GPU_FAILURES

_ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.mark.parametrize("node_id", sorted(EXPECTED_GPU_FAILURES))
def test_registry_entry_exists(node_id):
    path, *parts = node_id.split("::")
    full = os.path.join(_ROOT, path)
    assert os.path.exists(full), node_id
    text = open(full).read()
    *classes, name = parts
    for cls in classes:
        assert f"class {cls}" in text, node_id
    assert f"def {name.split('[')[0]}(" in text, node_id
