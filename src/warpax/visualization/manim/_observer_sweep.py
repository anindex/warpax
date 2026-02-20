"""ObserverSweep: Manim ThreeDScene showing EC violations across observer boosts.

Demonstrates how energy condition violations in the Alcubierre warp bubble
change as the observer rapidity sweeps from 0 (Eulerian) to 5 (highly
boosted).  A yellow arrow indicates the dominant boost direction.  The
geometry is fixed at v_s=0.5; only the observer changes.

Dual-layer rendering:

- **Upper layer:** translucent wireframe embedding surface showing
  ``energy_density`` (RdBu_r colorscale) - static curvature reference.
- **Lower layer:** flat heatmap showing ``wec_margin_sweep`` (RdYlGn
  colorscale: green = WEC satisfied, red = violated) - evolves with
  observer rapidity.

Both layers are always visible, enabling simultaneous comparison of
curvature magnitude and violation structure.

Usage::

    manim render -ql --format mp4 \\
        src/warpax/visualization/manim/_observer_sweep.py ObserverSweep
"""
from __future__ import annotations

import numpy as np
from manim import (
    DEGREES,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    UL,
    UR,
    WHITE,
    YELLOW,
    Arrow3D,
    DecimalNumber,
    FadeOut,
    MathTex,
    Rectangle,
    Text,
    ThreeDScene,
    ValueTracker,
    VGroup,
    always_redraw,
    linear,
)

from warpax.visualization.manim._heatmap import framedata_to_heatmap
from warpax.visualization.manim._scene_utils import (
    COLORS_3B1B,
    compute_auto_exaggeration,
    compute_global_clim,
    format_metric_equation,
    make_axes_for_frames,
    make_violation_indicator,
)
from warpax.visualization.manim._surface import framedata_to_surface


class ObserverSweep(ThreeDScene):
    """Alcubierre observer sweep: rapidity 0 to 5 at fixed v_s=0.5.

    Dual-layer layout (matching BubbleCollapse):

    - Upper: translucent wireframe embedding (energy_density, RdBu_r)
    - Lower: flat heatmap (wec_margin_sweep, RdYlGn)

    Live rapidity counter, static v_s label, boost direction arrow,
    defining equation overlay, violation status indicator, dual color
    legends.  Static camera for clear arrow visibility.
    """

    def construct(self) -> None:
        # --- Dark background ---
        self.camera.background_color = COLORS_3B1B["background"]

        # ==================================================================
        # Step 1: Pre-compute all FrameData (60 frames)
        # ==================================================================
        from warpax.geometry import GridSpec
        from warpax.visualization.common._scenes import scene_observer_sweep

        grid_spec = GridSpec(
            bounds=[(-3, 3), (-3, 3), (-3, 3)],
            shape=(30, 30, 30),
        )

        all_frames = scene_observer_sweep(
            grid_spec,
            n_frames=60,
            v_s=0.5,
            n_directions=3,
        )
        n_total = len(all_frames)

        # ==================================================================
        # Step 2: Setup 3D scene (static camera)
        # ==================================================================
        self.set_camera_orientation(
            phi=60 * DEGREES,
            theta=-70 * DEGREES,
        )

        # Build axes from energy_density range (static curvature reference)
        axes = make_axes_for_frames(all_frames, "energy_density")

        # Global color limits (prevents flickering).
        # WEC margin grows as cosh²(ζ) with rapidity, so raw min/max
        # would compress moderate-rapidity color contrast.  Use the
        # 2nd–98th percentile for a robust range.
        ed_clim = compute_global_clim(all_frames, "energy_density")
        wec_clim = compute_global_clim(
            all_frames, "wec_margin_sweep", percentile=2.0,
        )

        # Auto-exaggeration for embedding
        exag = compute_auto_exaggeration(all_frames, "energy_density")

        # z_extent for heatmap positioning
        max_abs = max(abs(ed_clim[0]), abs(ed_clim[1]))
        z_extent = max_abs * exag * 1.3

        # ==================================================================
        # Step 3: Title + equation overlay - top center
        # ==================================================================
        title_text = Text(
            "Alcubierre Observer Sweep",
            font_size=22, color=WHITE, weight="LIGHT",
        )
        equation = format_metric_equation("Alcubierre")
        equation.scale(0.65)
        header = VGroup(title_text, equation).arrange(DOWN, buff=0.15)
        header.to_edge(UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(header)
        self.add(header)

        # ==================================================================
        # Step 4: ValueTracker + always_redraw surfaces (dual-layer)
        # ==================================================================
        frame_idx = ValueTracker(0)

        def _make_surface():
            """Wireframe embedding surface: always energy_density."""
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_total - 1))
            frame = all_frames[idx].with_clim("energy_density", ed_clim)
            return framedata_to_surface(
                frame, "energy_density", axes,
                exaggeration=exag,
                resolution=(32, 32),
            )

        def _make_heatmap():
            """Flat heatmap: wec_margin_sweep - varies with rapidity."""
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_total - 1))
            frame = all_frames[idx].with_clim("wec_margin_sweep", wec_clim)
            return framedata_to_heatmap(
                frame, "wec_margin_sweep", axes,
                z_offset=-z_extent * 0.85,
                resolution=(48, 48),
                colormap="RdYlGn",
            )

        embedding = always_redraw(_make_surface)
        heatmap = always_redraw(_make_heatmap)

        # ==================================================================
        # Step 5: Boost direction arrow
        # ==================================================================
        origin = axes.c2p(0, 0, 0)
        tip = axes.c2p(2.0, 0, 0)
        arrow = Arrow3D(
            start=origin,
            end=tip,
            color=YELLOW,
            stroke_width=6,
        )

        # ==================================================================
        # Step 6: Parameter displays - upper-left (DecimalNumber + become)
        # ==================================================================
        # NOTE: In ThreeDScene, DecimalNumber.set_value() recreates internal
        # submobjects that lose their fixed-in-frame registration.  We use
        # ``become()`` in the updaters instead, which replaces geometry
        # in-place and preserves the registration.

        # Rapidity counter
        z_label = MathTex(r"\zeta", font_size=32, color=WHITE)
        z_eq = MathTex(r"=", font_size=32, color=WHITE)
        z_num = DecimalNumber(
            0.0, num_decimal_places=2, font_size=32, color=YELLOW,
        )
        z_row = VGroup(z_label, z_eq, z_num).arrange(RIGHT, buff=0.1)

        # Static v_s row
        v_label = MathTex(r"v_s", font_size=32, color=WHITE)
        v_eq = MathTex(r"=", font_size=32, color=WHITE)
        v_num = DecimalNumber(
            0.5, num_decimal_places=2, font_size=32, color=YELLOW,
        )
        v_row = VGroup(v_label, v_eq, v_num).arrange(RIGHT, buff=0.1)

        # Boost direction label
        boost_label = MathTex(
            r"\hat{n}_{\text{boost}} = \hat{x}",
            font_size=28,
            color=YELLOW,
        )

        # Stack with = signs aligned
        param_block = VGroup(z_row, v_row, boost_label).arrange(
            DOWN, buff=0.15, aligned_edge=LEFT,
        )
        param_block.to_corner(UL, buff=0.5)

        # Register every leaf mobject individually
        self.add_fixed_in_frame_mobjects(
            z_label, z_eq, z_num, v_label, v_eq, v_num, boost_label,
        )

        def _update_rapidity(mob):
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_total - 1))
            zeta_val = all_frames[idx].t  # rapidity stored in t field
            new_mob = DecimalNumber(
                zeta_val, num_decimal_places=2, font_size=32, color=YELLOW,
            )
            new_mob.move_to(mob)
            mob.become(new_mob)
            # Re-register so new leaf glyphs stay fixed-in-frame
            self.add_fixed_in_frame_mobjects(mob)

        z_num.add_updater(_update_rapidity)

        # ==================================================================
        # Step 7: Color legends - upper-right (dual legends)
        # ==================================================================
        # Energy density legend (RdBu_r) --
        ed_colors = ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"]
        ed_strips = VGroup(*[
            Rectangle(
                width=0.28, height=0.10,
                fill_color=c, fill_opacity=0.95,
                stroke_width=0.3, stroke_color=WHITE,
            ) for c in ed_colors
        ]).arrange(RIGHT, buff=0)
        ed_title = Text(
            "Energy density (wireframe)",
            font_size=16, color=WHITE, weight="LIGHT",
        )
        ed_lo = MathTex(r"-", font_size=18, color="#2166AC")
        ed_hi = MathTex(r"+", font_size=18, color="#B2182B")
        ed_bar_row = VGroup(ed_lo, ed_strips, ed_hi).arrange(
            RIGHT, buff=0.06,
        )
        ed_field = MathTex(r"T_{00}", font_size=22, color=WHITE)
        ed_top_row = VGroup(ed_field, ed_title).arrange(
            RIGHT, buff=0.12,
        )
        ed_legend = VGroup(ed_top_row, ed_bar_row).arrange(
            DOWN, buff=0.08, aligned_edge=LEFT,
        )

        # WEC margin legend (RdYlGn) --
        wec_colors = ["#A50026", "#F46D43", "#FFFFBF", "#66BD63", "#006837"]
        wec_strips = VGroup(*[
            Rectangle(
                width=0.28, height=0.10,
                fill_color=c, fill_opacity=0.95,
                stroke_width=0.3, stroke_color=WHITE,
            ) for c in wec_colors
        ]).arrange(RIGHT, buff=0)
        wec_title = Text(
            "WEC margin (heatmap)",
            font_size=16, color=WHITE, weight="LIGHT",
        )
        wec_lo = MathTex(
            r"\text{violated}", font_size=14, color="#A50026",
        )
        wec_hi = MathTex(
            r"\text{satisfied}", font_size=14, color="#006837",
        )
        wec_bar_row = VGroup(wec_lo, wec_strips, wec_hi).arrange(
            RIGHT, buff=0.06,
        )
        # Violation dot integrated into WEC legend header
        from manim import Dot
        violation_ind = make_violation_indicator("wec_margin_sweep")
        dot_mob: Dot = violation_ind[0]  # type: ignore[assignment]
        dot_mob.scale(0.7)
        wec_field = VGroup(dot_mob, wec_title).arrange(
            RIGHT, buff=0.1,
        )
        wec_legend = VGroup(wec_field, wec_bar_row).arrange(
            DOWN, buff=0.08, aligned_edge=LEFT,
        )

        # Stack both legends, right-aligned in UR corner
        legend_group = VGroup(ed_legend, wec_legend).arrange(
            DOWN, buff=0.25, aligned_edge=LEFT,
        )
        legend_group.to_corner(UR, buff=0.35)

        def _update_violation(mob):
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_total - 1))
            frame = all_frames[idx]
            if "wec_margin_sweep" in frame.scalar_fields:
                wec_min = float(
                    np.min(frame.scalar_fields["wec_margin_sweep"])
                )
                if wec_min < 0:
                    dot_mob.set_color(COLORS_3B1B["violation_red"])
                else:
                    dot_mob.set_color(COLORS_3B1B["safe_green"])
            else:
                dot_mob.set_color(WHITE)

        dot_mob.add_updater(_update_violation)
        self.add_fixed_in_frame_mobjects(legend_group)

        # ==================================================================
        # Step 8: Animate (static camera - no orbit)
        # ==================================================================
        self.add(axes, embedding, heatmap, arrow)
        self.play(
            frame_idx.animate.set_value(n_total - 1),
            run_time=19,
            rate_func=linear,
        )

        # Fade out header before outro
        self.play(FadeOut(header), run_time=0.5)
        self.wait(2)  # Outro freeze frame
