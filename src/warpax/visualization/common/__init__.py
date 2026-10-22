"""Backend-agnostic visualization data layer.

Provides FrameData, freeze functions, scene builders and physics profiles
that Manim and matplotlib backends can import without backend-specific
dependencies.

All modules are pure Python/NumPy/Equinox; no Manim required.
"""

from __future__ import annotations

from ._conversion import (
    DIVERGING_FIELDS,
    eulerian_energy_density_grid,
    eulerian_wec_fields,
    freeze_curvature,
)
from ._frame_data import FrameData
from ._physics import (
    build_ec_frame_sequence,
    build_frame_sequence,
    make_velocity_sweep,
    rampdown_profile,
)
from ._scenes import scene_observer_sweep, scene_velocity_ramp

__all__ = [
    "DIVERGING_FIELDS",
    "FrameData",
    "build_ec_frame_sequence",
    "build_frame_sequence",
    "eulerian_energy_density_grid",
    "eulerian_wec_fields",
    "freeze_curvature",
    "make_velocity_sweep",
    "rampdown_profile",
    "scene_observer_sweep",
    "scene_velocity_ramp",
]
