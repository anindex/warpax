"""Shared numerical floors and the ``WARPAX_STRICT`` runtime guard.

* :data:`R_EPS` -- additive floor for radicands inside ``sqrt`` to keep
  reverse-mode AD finite at the axis; changes values by ``sqrt(R_EPS) ~ 1e-30``.
* :data:`LAPSE_EPS` -- minimum lapse used by WarpShell-family metrics so
  the ADM normal stays finite.
* :data:`DENOM_EPS` -- generic denominator floor for ratios that must
  remain finite.
* :func:`strict_mode_enabled` -- ``True`` when ``WARPAX_STRICT=1``; used to
  gate expensive consistency probes (uniform-grid asserts, signature flips).
"""

from __future__ import annotations

import os

R_EPS: float = 1e-60
LAPSE_EPS: float = 1e-30
DENOM_EPS: float = 1e-30


def strict_mode_enabled() -> bool:
    """Return ``True`` when the ``WARPAX_STRICT`` environment variable is set to ``"1"``."""
    return os.environ.get("WARPAX_STRICT", "") == "1"


__all__ = ["R_EPS", "LAPSE_EPS", "DENOM_EPS", "strict_mode_enabled"]
