"""Manim visualization backend for warp drive geometries (pip install warpax[manim])."""

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
    from ._wall_velocity_sweep import WallAndVelocitySweep
    from ._boost_rapidity_sweep import BoostRapiditySweep
    from ._velocity_sweep import VelocitySweep

    # Scene classes (2D heatmap)
    from ._boost_arrows import WorstCaseBoostDirections, WorstCaseNullDirections
    from ._eulerian_kinematics import EulerianKinematics2D
    from ._nec_margin import NECMargin2D
    from ._split_screen import EulerianVsWorstCaseNEC
    from ._kretschmann import KretschmannInvariant2D

    # Image and GIF utilities
    from ._image_utils import (
        compute_symlog_clim,
        extract_contours,
        extract_zero_contour,
        frame_to_rgba,
    )
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
    "WallAndVelocitySweep",
    "VelocitySweep",
    "BoostRapiditySweep",
    # Scene classes (2D heatmap)
    "NECMargin2D",
    "EulerianKinematics2D",
    "EulerianVsWorstCaseNEC",
    "WorstCaseNullDirections",
    "WorstCaseBoostDirections",
    "KretschmannInvariant2D",
    # Image and GIF utilities
    "frame_to_rgba",
    "extract_contours",
    "extract_zero_contour",
    "compute_symlog_clim",
    "mp4_to_gif",
]
