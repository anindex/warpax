"""Color scale utilities for 3D rendering.

Resolves colormaps and color limits from FrameData rendering hints with
physics-aware fallbacks:
- EC violation margins: ``RdBu_r`` diverging, symmetric around zero.
- Magnitude fields: ``inferno`` sequential.
- HE type classification: ``tab10`` categorical.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._conversion import _magnitude_clim, _symmetric_clim

if TYPE_CHECKING:
    from ._frame_data import FrameData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARGIN_FIELDS: frozenset[str] = frozenset(
    {"nec_margin", "wec_margin", "sec_margin", "dec_margin"}
)
"""Scalar fields representing EC violation margins (diverging colormap)."""

_DIVERGING_FIELDS: frozenset[str] = MARGIN_FIELDS | frozenset(
    {"rho", "energy_density", "ricci_scalar"}
)
"""All fields that use symmetric (diverging) color limits."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_diverging(field: str) -> bool:
    """Return True if *field* uses a diverging (zero-centered) colormap.

    Diverging fields include EC margins, energy density, Ricci scalar,
    and energy density rho.
    """
    return field in _DIVERGING_FIELDS


def resolve_cmap(frame: "FrameData", field: str) -> str:
    """Resolve the colormap name for *field* from *frame* hints.

    Priority:
    1. ``frame.colormaps[field]`` if present.
    2. Physics-aware fallback: ``RdBu_r`` for margins, ``tab10`` for HE type,
       ``inferno`` for magnitude fields.

    Parameters
    ----------
    frame : FrameData
        Frozen snapshot with rendering hints.
    field : str
        Scalar field name.

    Returns
    -------
    str
        Matplotlib-compatible colormap name.
    """
    # Explicit hint from freeze-time
    if field in frame.colormaps:
        return frame.colormaps[field]

    # Physics-aware fallbacks
    if field in MARGIN_FIELDS or field.endswith("_margin"):
        return "RdBu_r"
    if field == "he_type":
        return "tab10"
    if field in _DIVERGING_FIELDS:
        return "RdBu_r"

    return "inferno"


def resolve_clim(frame: "FrameData", field: str) -> tuple[float, float]:
    """Resolve color limits for *field* from *frame* hints.

    Priority:
    1. ``frame.clim[field]`` if present.
    2. Compute from ``frame.scalar_fields[field]``: symmetric for diverging,
       magnitude for sequential.

    Parameters
    ----------
    frame : FrameData
        Frozen snapshot with rendering hints and scalar data.
    field : str
        Scalar field name.

    Returns
    -------
    tuple[float, float]
        ``(vmin, vmax)`` color limits.

    Raises
    ------
    ValueError
        If *field* is not in ``frame.scalar_fields``.
    """
    # Explicit hint from freeze-time
    if field in frame.clim:
        return frame.clim[field]

    # Compute from data
    if field not in frame.scalar_fields:
        raise ValueError(
            f"Field {field!r} not in FrameData scalar_fields. "
            f"Available: {frame.field_names}"
        )

    arr = frame.scalar_fields[field]
    if is_diverging(field):
        return _symmetric_clim(arr)
    return _magnitude_clim(arr)


def resolve_clim_from_array(arr: np.ndarray, field: str) -> tuple[float, float]:
    """Resolve color limits from a raw array (no FrameData required).

    Parameters
    ----------
    arr : np.ndarray
        Scalar field data.
    field : str
        Field name (used to determine diverging vs sequential).

    Returns
    -------
    tuple[float, float]
        ``(vmin, vmax)`` color limits.
    """
    if is_diverging(field):
        return _symmetric_clim(arr)
    return _magnitude_clim(arr)
