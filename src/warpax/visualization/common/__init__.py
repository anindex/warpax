"""Backend-agnostic visualization data layer.

Provides FrameData, freeze functions, scene builders, physics profiles,
color/theme utilities, and overlay helpers that Manim and matplotlib
backends can import without backend-specific dependencies.

All modules are pure Python/NumPy/Equinox no Manim required.
"""
from __future__ import annotations

# Data bridge
from ._frame_data import FrameData

# JAX-to-NumPy conversion
from ._conversion import freeze_curvature, freeze_ec

# Color scale utilities
from ._color import is_diverging, resolve_clim, resolve_clim_from_array, resolve_cmap

# Rendering themes
from ._themes import PAPER_THEME, PRESENTATION_THEME, RenderTheme, get_theme

# Velocity profiles and frame sequence builders
from ._physics import (
    build_ec_frame_sequence,
    build_frame_sequence,
    collapse_profile,
    constant_velocity,
    linear_ramp,
    make_velocity_sweep,
    sigmoid_ramp,
)

# Scene builders and overlay helpers
from ._scenes import (
    add_text_overlay,
    add_watermark,
    scene_bubble_collapse,
    scene_observer_sweep,
    scene_velocity_ramp,
)

__all__ = [
    # Data bridge
    "FrameData",
    # Conversion
    "freeze_curvature",
    "freeze_ec",
    # Color
    "resolve_cmap",
    "resolve_clim",
    "resolve_clim_from_array",
    "is_diverging",
    # Themes
    "RenderTheme",
    "get_theme",
    "PAPER_THEME",
    "PRESENTATION_THEME",
    # Physics profiles
    "linear_ramp",
    "sigmoid_ramp",
    "collapse_profile",
    "constant_velocity",
    "make_velocity_sweep",
    "build_frame_sequence",
    "build_ec_frame_sequence",
    # Scenes and overlays
    "add_text_overlay",
    "add_watermark",
    "scene_bubble_collapse",
    "scene_velocity_ramp",
    "scene_observer_sweep",
]
