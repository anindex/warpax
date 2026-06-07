#!/usr/bin/env python
"""Capture certified-output goldens for parity testing.

Run on a known-good tree BEFORE any refactor; the snapshot is stored at
``tests/fixtures/parity_goldens.npz`` and asserted by
``tests/test_parity_golden.py``. Re-run intentionally (and review the diff)
only when a change is *meant* to move a number.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python scripts/capture_goldens.py
"""
from __future__ import annotations

import pathlib
import sys

# Allow running from the repo root without installation.
_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_TESTS = pathlib.Path(__file__).resolve().parent.parent / "tests"
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import numpy as np

from _golden_harness import compute_goldens


def main() -> int:
    goldens = compute_goldens()
    out_dir = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parity_goldens.npz"
    np.savez(out_path, **goldens)
    print(f"[golden] wrote {len(goldens)} keys -> {out_path}")
    for k in sorted(goldens):
        print(f"  {k}: shape={goldens[k].shape} dtype={goldens[k].dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
