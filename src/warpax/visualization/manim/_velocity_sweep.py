"""VelocitySweep: Alcubierre v_s sweep with NEC margin slab (dual-layer 3D).

Quasi-static sweep of the bubble velocity (v_s: 0.1 -> 0.99); each frame is a
static metric snapshot, not a time evolution.

Usage: manim render -ql --format mp4 \\
    src/warpax/visualization/manim/_velocity_sweep.py VelocitySweep
"""

from __future__ import annotations

# Fork-safety for the 3D render: manim spawns ffmpeg while JAX's thread pool is
# live, which deadlocks on fork under Python 3.14. When this module is the manim
# render entrypoint (JAX not yet imported), enable gRPC fork handlers and drop
# the CUDA-probe threads (the render runs JAX on CPU) before the warpax imports
# below pull JAX in. Skipped when JAX is already loaded (imported as a library)
# so it never mutates a live process's env. ``setdefault`` lets a caller override.
import os as _os
import sys as _sys

if "jax" not in _sys.modules:
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")
    _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _os.environ.setdefault("MKL_NUM_THREADS", "1")
    _os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "1")
    if "xla_cpu_multi_thread_eigen" not in _os.environ.get("XLA_FLAGS", ""):
        _os.environ["XLA_FLAGS"] = (
            _os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
        ).strip()

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
    cap_recursion_for_3d_render,
    compute_auto_exaggeration,
    compute_global_clim,
    format_metric_equation,
    make_axes_for_frames,
    make_violation_indicator,
)
from warpax.visualization.manim._surface import framedata_to_surface


class VelocitySweep(ThreeDScene):
    """Alcubierre velocity sweep: v_s sweeps 0.1 to 0.99 (quasi-static).

    Dual-layer layout (matching WallAndVelocitySweep):

    - Upper: translucent wireframe embedding (Eulerian energy density rho_Eul,
      <= 0 everywhere -> one-sided blue depth scale)
    - Lower: flat colored slab (nec_margin_sweep, one-sided violation-depth
      ``nec_depth`` colormap)

    Live v_s counter, defining equation overlay, violation status
    indicator, dual color legends.
    """

    def construct(self) -> None:
        cap_recursion_for_3d_render()  # py3.14 + 3D Cairo: avoid C-stack segfault

        # --- Dark background ---
        self.camera.background_color = COLORS_3B1B["background"]

        from warpax.geometry import GridSpec
        from warpax.visualization.common._scenes import scene_velocity_ramp

        grid_spec = GridSpec(
            bounds=[(-3, 3), (-3, 3), (-3, 3)],
            shape=(30, 30, 30),
        )

        all_frames = scene_velocity_ramp(
            grid_spec,
            n_frames=90,
            v_start=0.1,
            v_end=0.99,
        )
        n_total = len(all_frames)

        self.set_camera_orientation(
            phi=60 * DEGREES,
            theta=-45 * DEGREES,
        )

        # Build axes from energy_density range across all frames
        axes = make_axes_for_frames(all_frames, "energy_density")

        # Global color limits (prevents flickering). rho_Eul and the NEC margin
        # are both <= 0 for Alcubierre -> one-sided depth scales (deepest -> 0).
        ed_clim = (compute_global_clim(all_frames, "energy_density")[0], 0.0)
        nec_clim = (compute_global_clim(all_frames, "nec_margin_sweep")[0], 0.0)

        # Auto-exaggeration for embedding
        exag = compute_auto_exaggeration(all_frames, "energy_density")

        # z_extent for heatmap positioning
        max_abs = max(abs(ed_clim[0]), abs(ed_clim[1]))
        z_extent = max_abs * exag * 1.3

        title_text = Text(
            "Alcubierre Velocity Sweep",
            font_size=22,
            color=WHITE,
            weight="LIGHT",
        )
        equation = format_metric_equation("Alcubierre")
        equation.scale(0.65)
        header = VGroup(title_text, equation).arrange(DOWN, buff=0.15)
        header.to_edge(UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(header)
        self.add(header)

        frame_idx = ValueTracker(0)

        def _make_surface():
            """Wireframe embedding surface: always energy_density."""
            idx = int(round(frame_idx.get_value()))
            idx = max(0, min(idx, n_total - 1))
            frame = all_frames[idx].with_clim("energy_density", ed_clim)
            return framedata_to_surface(
                frame,
                "energy_density",
                axes,
                exaggeration=exag,
                resolution=(32, 32),
            )

        def _make_heatmap():
            """Flat heatmap: nec_margin_sweep, one-sided violation-depth ramp."""
            idx = int(round(frame_idx.get_value()))
            idx = max(0, min(idx, n_total - 1))
            frame = all_frames[idx].with_clim("nec_margin_sweep", nec_clim)
            return framedata_to_heatmap(
                frame,
                "nec_margin_sweep",
                axes,
                z_offset=-z_extent * 0.85,
                resolution=(48, 48),
                colormap="nec_depth",
            )

        embedding = always_redraw(_make_surface)
        heatmap = always_redraw(_make_heatmap)

        v_label = MathTex(r"v_s", font_size=32, color=WHITE)
        v_eq = MathTex(r"=", font_size=32, color=WHITE)
        v_num = DecimalNumber(
            0.1,
            num_decimal_places=2,
            font_size=32,
            color=YELLOW,
        )
        v_row = VGroup(v_label, v_eq, v_num).arrange(RIGHT, buff=0.1)
        v_row.to_corner(UL, buff=0.5)

        # Register every leaf mobject individually
        self.add_fixed_in_frame_mobjects(v_label, v_eq, v_num)

        def _update_v(mob):
            idx = int(round(frame_idx.get_value()))
            idx = max(0, min(idx, n_total - 1))
            v_val = all_frames[idx].v_s
            new_mob = DecimalNumber(
                v_val,
                num_decimal_places=2,
                font_size=32,
                color=YELLOW,
            )
            new_mob.move_to(mob)
            mob.become(new_mob)
            # Re-register so new leaf glyphs stay fixed-in-frame
            self.add_fixed_in_frame_mobjects(mob)

        v_num.add_updater(_update_v)

        # Eulerian energy density legend: one-sided (rho_Eul <= 0 everywhere).
        ed_colors = ["#2166AC", "#67A9CF", "#A9C9E0", "#D1E5F0", "#F7F7F7"]
        ed_strips = VGroup(
            *[
                Rectangle(
                    width=0.28,
                    height=0.10,
                    fill_color=c,
                    fill_opacity=0.95,
                    stroke_width=0.3,
                    stroke_color=WHITE,
                )
                for c in ed_colors
            ]
        ).arrange(RIGHT, buff=0)
        ed_title = Text(
            "Eulerian energy density (wireframe; ≤ 0)",
            font_size=16,
            color=WHITE,
            weight="LIGHT",
        )
        ed_lo = MathTex(r"\text{most }-", font_size=14, color="#2166AC")
        ed_hi = MathTex(r"0", font_size=14, color=WHITE)
        ed_bar_row = VGroup(ed_lo, ed_strips, ed_hi).arrange(
            RIGHT,
            buff=0.06,
        )
        ed_field = MathTex(r"\rho_{\rm Eul}", font_size=22, color=WHITE)
        ed_top_row = VGroup(ed_field, ed_title).arrange(
            RIGHT,
            buff=0.12,
        )
        ed_legend = VGroup(ed_top_row, ed_bar_row).arrange(
            DOWN,
            buff=0.08,
            aligned_edge=LEFT,
        )

        # NEC margin legend: one-sided violation-depth ramp (matches the slab).
        nec_colors = ["#CFE0FF", "#7FA0FF", "#3B4CC0", "#27306E", "#15151F"]
        nec_strips = VGroup(
            *[
                Rectangle(
                    width=0.28,
                    height=0.10,
                    fill_color=c,
                    fill_opacity=0.95,
                    stroke_width=0.3,
                    stroke_color=WHITE,
                )
                for c in nec_colors
            ]
        ).arrange(RIGHT, buff=0)
        nec_title = Text(
            "NEC margin (≤ 0; negative = violation)",
            font_size=16,
            color=WHITE,
            weight="LIGHT",
        )
        nec_lo = MathTex(
            r"\text{deepest}",
            font_size=14,
            color="#7FA0FF",
        )
        nec_hi = MathTex(r"0", font_size=14, color=WHITE)
        nec_bar_row = VGroup(nec_lo, nec_strips, nec_hi).arrange(
            RIGHT,
            buff=0.06,
        )
        # Violation dot integrated into NEC legend header
        from manim import Dot

        violation_ind = make_violation_indicator("nec_margin_sweep")
        dot_mob: Dot = violation_ind[0]  # type: ignore[assignment]
        dot_mob.scale(0.7)
        nec_field = VGroup(dot_mob, nec_title).arrange(
            RIGHT,
            buff=0.1,
        )
        nec_legend = VGroup(nec_field, nec_bar_row).arrange(
            DOWN,
            buff=0.08,
            aligned_edge=LEFT,
        )

        # Stack both legends, right-aligned in UR corner
        legend_group = VGroup(ed_legend, nec_legend).arrange(
            DOWN,
            buff=0.25,
            aligned_edge=LEFT,
        )
        legend_group.to_corner(UR, buff=0.35).shift(DOWN * 1.0)

        def _update_violation(mob):
            idx = int(round(frame_idx.get_value()))
            idx = max(0, min(idx, n_total - 1))
            frame = all_frames[idx]
            if "nec_margin_sweep" in frame.scalar_fields:
                mid_z = frame.grid_shape[2] // 2
                nec_min = float(np.min(frame.scalar_fields["nec_margin_sweep"][:, :, mid_z]))
                if nec_min < 0:
                    dot_mob.set_color(COLORS_3B1B["violation_red"])
                else:
                    dot_mob.set_color(COLORS_3B1B["safe_green"])
            else:
                dot_mob.set_color(WHITE)

        dot_mob.add_updater(_update_violation)
        self.add_fixed_in_frame_mobjects(legend_group)

        # GR-conventions footer (static metric per frame, units, signature).
        from warpax.visualization.manim._scene_utils import make_conventions_caption

        caption = make_conventions_caption()
        if caption.width > 8.0:
            caption.scale_to_fit_width(8.0)
        caption.to_edge(DOWN, buff=0.06)
        self.add_fixed_in_frame_mobjects(caption)

        self.add(axes, embedding, heatmap)
        self.begin_ambient_camera_rotation(rate=0.015, about="theta")
        self.play(
            frame_idx.animate.set_value(n_total - 1),
            run_time=22,
            rate_func=linear,
        )
        self.stop_ambient_camera_rotation()

        # Fade out header before outro
        self.play(FadeOut(header), run_time=0.5)
        self.wait(2)  # Outro freeze frame
