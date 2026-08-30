"""Backend-agnostic visualization data layer.

Provides FrameData, freeze functions, scene builders, physics profiles,
color/theme utilities, and overlay helpers that Manim and matplotlib
backends can import without backend-specific dependencies.

All modules are pure Python/NumPy/Equinox; no Manim required.
"""

from __future__ import annotations

# Color scale utilities
from ._color import is_diverging, resolve_clim, resolve_clim_from_array, resolve_cmap

# JAX-to-NumPy conversion
from ._conversion import (
    eulerian_energy_density_grid,
    eulerian_wec_fields,
    freeze_curvature,
    freeze_ec,
)

# Data bridge
from ._frame_data import FrameData

# Rendering themes
# Velocity profiles and frame sequence builders
from ._physics import (
    build_ec_frame_sequence,
    build_frame_sequence,
    constant_velocity,
    linear_ramp,
    make_velocity_sweep,
    rampdown_profile,
    sigmoid_ramp,
)

# Scene builders and overlay helpers
from ._scenes import (
    add_text_overlay,
    add_watermark,
    scene_observer_sweep,
    scene_velocity_ramp,
    scene_velocity_rampdown,
)

__all__ = [
    # Data bridge
    "FrameData",
    # Scenes and overlays
    "add_text_overlay",
    "add_watermark",
    "build_ec_frame_sequence",
    "build_frame_sequence",
    "constant_velocity",
    # Conversion
    "eulerian_energy_density_grid",
    "eulerian_wec_fields",
    "freeze_curvature",
    "freeze_ec",
    "is_diverging",
    # Themes
    # Physics profiles
    "linear_ramp",
    "make_velocity_sweep",
    "rampdown_profile",
    "resolve_clim",
    "resolve_clim_from_array",
    # Color
    "resolve_cmap",
    "scene_observer_sweep",
    "scene_velocity_ramp",
    "scene_velocity_rampdown",
    "sigmoid_ramp",
]
