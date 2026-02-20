"""Manim visualization backend for warp drive geometries.

Requires: pip install warpax[manim]

Exports bridge functions (framedata_to_surface, framedata_to_heatmap),
shared scene utilities, image/GIF conversion helpers, and seven
showcase animation scenes (3 x 3D, 4 x 2D heatmap).
"""
from __future__ import annotations

try:
    import manim  # noqa: F401

    _HAS_MANIM = True
except ImportError:
    _HAS_MANIM = False

if _HAS_MANIM:
    from ._heatmap import framedata_to_heatmap
    from ._scene_utils import (
        COLORS_3B1B,
        compute_global_clim,
        make_parameter_display,
        play_title_card,
    )
    from ._surface import framedata_to_surface

    # Scene classes (3D)
    from ._bubble_collapse import BubbleCollapse
    from ._observer_sweep import ObserverSweep
    from ._velocity_ramp import VelocityRamp

    # Scene classes (2D heatmap)
    from ._boost_arrows import BoostArrows
    from ._expansion_shear import ExpansionShear
    from ._heatmap_contour import ECHeatmapContour
    from ._split_screen import SplitScreen

    # Image and GIF utilities
    from ._image_utils import compute_symlog_clim, extract_zero_contour, frame_to_rgba
    from ._gif_utils import mp4_to_gif

__all__ = [
    # Bridge functions
    "framedata_to_surface",
    "framedata_to_heatmap",
    # Scene utilities
    "play_title_card",
    "make_parameter_display",
    "compute_global_clim",
    "COLORS_3B1B",
    # Scene classes (3D)
    "BubbleCollapse",
    "VelocityRamp",
    "ObserverSweep",
    # Scene classes (2D heatmap)
    "ECHeatmapContour",
    "ExpansionShear",
    "SplitScreen",
    "BoostArrows",
    # Image and GIF utilities
    "frame_to_rgba",
    "extract_zero_contour",
    "compute_symlog_clim",
    "mp4_to_gif",
]
