"""Shared numerical utilities (constants, floors, autodiff-safe helpers)."""

from ._constants import DENOM_EPS, LAPSE_EPS, R_EPS, strict_mode_enabled
from ._grid import assert_uniform_grid
from ._hermite import cubic_interp, hermite_cubic, local_cubic_slopes, not_a_knot_slopes

__all__ = [
    "DENOM_EPS",
    "LAPSE_EPS",
    "R_EPS",
    "assert_uniform_grid",
    "cubic_interp",
    "hermite_cubic",
    "local_cubic_slopes",
    "not_a_knot_slopes",
    "strict_mode_enabled",
]
