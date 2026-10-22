"""Manim visualization backend for warp drive geometries (pip install warpax[manim])."""

from __future__ import annotations

try:
    import manim

    _HAS_MANIM = True
except ImportError:
    _HAS_MANIM = False

if _HAS_MANIM:
    from ._boost_arrows import WorstCaseBoostDirections, WorstCaseNullDirections
    from ._boost_rapidity_sweep import BoostRapiditySweep
    from ._eulerian_kinematics import EulerianKinematics2D
    from ._gif_utils import mp4_to_gif
    from ._heatmap import framedata_to_heatmap
    from ._image_utils import (
        compute_symlog_clim,
        extract_contours,
        extract_zero_contour,
        frame_to_rgba,
    )
    from ._kretschmann import KretschmannInvariant2D
    from ._nec_margin import NECMargin2D
    from ._scene_utils import (
        COLORS_3B1B,
        compute_global_clim,
        make_parameter_display,
        play_title_card,
    )
    from ._split_screen import EulerianVsWorstCaseNEC
    from ._surface import framedata_to_surface
    from ._velocity_sweep import VelocitySweep
    from ._wall_velocity_sweep import WallAndVelocitySweep

__all__ = [
    "COLORS_3B1B",
    "BoostRapiditySweep",
    "EulerianKinematics2D",
    "EulerianVsWorstCaseNEC",
    "KretschmannInvariant2D",
    "NECMargin2D",
    "VelocitySweep",
    "WallAndVelocitySweep",
    "WorstCaseBoostDirections",
    "WorstCaseNullDirections",
    "compute_global_clim",
    "compute_symlog_clim",
    "extract_contours",
    "extract_zero_contour",
    "frame_to_rgba",
    "framedata_to_heatmap",
    "framedata_to_surface",
    "make_parameter_display",
    "mp4_to_gif",
    "play_title_card",
]
