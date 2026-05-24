"""Shared numerical utilities (constants, floors, autodiff-safe helpers)."""

from ._constants import DENOM_EPS, LAPSE_EPS, R_EPS, strict_mode_enabled
from ._grid import assert_uniform_grid

__all__ = [
    "DENOM_EPS",
    "LAPSE_EPS",
    "R_EPS",
    "assert_uniform_grid",
    "strict_mode_enabled",
]
