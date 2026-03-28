"""Smoke tests for the Blackwell sm_120 xfail registry.

Pins registry invariants (count, format, loadability) so the
``--gpu-baseline`` harness stays self-consistent across regens.

See also
--------
- ``tests/_gpu_xfail_registry.py`` - the registry module
- ``tests/conftest.py`` - the pytest plugin hooks
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


# Local import of the registry (sys.path-prepended) - the tests/ dir is a
# pytest test root, not a package, so absolute `from warpax.tests...` imports
# won't resolve. The conftest.py hooks use the same relative import pattern
# (`from ._gpu_xfail_registry import ...`) inside pytest plugin lifecycles.
_TESTS_DIR = Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _gpu_xfail_registry import (  # noqa: E402 (sys.path insert above)
    EXPECTED_GPU_FAILURES,
    EXPECTED_GPU_FAILURES_COUNT_TARGET,
)


class TestGPUBaselineRegistry:
    """Registry invariants (loadability, format, count)."""

    def test_registry_loads(self):
        """EXPECTED_GPU_FAILURES is a non-empty frozenset."""
        assert isinstance(EXPECTED_GPU_FAILURES, frozenset)
        assert len(EXPECTED_GPU_FAILURES) > 0

    def test_registry_format_valid(self):
        """Every entry matches file.py::(Class::)?method - accepts both
        ``file::Class::method`` and ``file::func`` forms (pytest allows
        module-level tests without an enclosing class).
        """
        # Module-level: file.py::test_name
        # Method: file.py::Class::test_name
        pattern = re.compile(
            r"^[^:]+\.py::[^:]+(?:\[[^\]]+\])?"  # file::name (with optional [param])
            r"(?:::[^:]+(?:\[[^\]]+\])?)?$"  # optional ::method (with [param])
        )
        bad_entries = [
            entry for entry in EXPECTED_GPU_FAILURES if not pattern.match(entry)
        ]
        assert not bad_entries, (
            f"{len(bad_entries)} registry entries fail format: {bad_entries[:3]}"
        )

    def test_registry_count_matches_spec(self):
        """len(EXPECTED_GPU_FAILURES) == EXPECTED_GPU_FAILURES_COUNT_TARGET.

        Locked at 79 from the baseline log.
        Any change requires a CHANGELOG entry.
        """
        assert len(EXPECTED_GPU_FAILURES) == EXPECTED_GPU_FAILURES_COUNT_TARGET


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
