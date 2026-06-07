"""Parity regression net: certified outputs must not drift across refactors.

Re-runs the shared golden harness and compares against the snapshot captured
by ``scripts/capture_goldens.py`` (``tests/fixtures/parity_goldens.npz``).

Engineering/performance refactors (Phases 1-2 of the R3 plan) must keep every
key here bit-exact (integer/type fields) or within a tight tolerance (floats).
The only intentional movers are the targeted bug fixes, which get their own
dedicated tests and a deliberate golden re-capture with reviewed diff.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ._golden_harness import compute_goldens

_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "parity_goldens.npz"

# Fields compared with exact equality (integer-valued / categorical).
_EXACT_SUFFIXES = ("/he_type", "/is_vacuum")

_RTOL = 1e-10
_ATOL = 1e-12


@pytest.fixture(scope="module")
def stored():
    if not _FIXTURE.exists():
        pytest.skip(
            f"golden fixture missing: {_FIXTURE} "
            "(run scripts/capture_goldens.py)"
        )
    return dict(np.load(_FIXTURE, allow_pickle=False))


@pytest.fixture(scope="module")
def live():
    return compute_goldens()


def _is_exact(key: str) -> bool:
    return any(key.endswith(s) for s in _EXACT_SUFFIXES) or key == "bugA/he_type"


def test_no_golden_keys_dropped(stored, live):
    """Every previously-captured key must still be produced (catches a block
    silently failing during a refactor)."""
    missing = sorted(set(stored) - set(live))
    assert not missing, f"golden keys no longer produced: {missing}"


@pytest.mark.parametrize("key", sorted(dict(np.load(_FIXTURE)).keys()) if _FIXTURE.exists() else [])
def test_golden_parity(key, stored, live):
    """Each golden value matches the snapshot (exact for ints, tight for floats)."""
    assert key in live, f"missing live value for {key}"
    want = stored[key]
    got = np.asarray(live[key])
    assert got.shape == want.shape, f"{key}: shape {got.shape} != {want.shape}"
    if _is_exact(key):
        assert np.array_equal(got, want), f"{key}: exact mismatch"
    else:
        np.testing.assert_allclose(
            got, want, rtol=_RTOL, atol=_ATOL, equal_nan=True,
            err_msg=f"{key}: float drift beyond rtol={_RTOL}",
        )
