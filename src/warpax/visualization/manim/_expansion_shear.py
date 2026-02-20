"""ExpansionShear: 2D Manim Scene showing expansion θ heatmap with shear overlay.

Renders a full-screen equatorial-plane heatmap of the expansion scalar θ
(computed from the Eulerian congruence kinematics) with two overlays:

1. **Shear contour** (σ² at 50 % of max) as yellow dashed paths -
   shows where shear is concentrated (primarily at the bubble wall).
2. **Bubble wall contour** (f = 0.5) as cyan outline - shows the physical
   extent of the warp bubble.

The bipolar expansion structure is visible in the colormap: θ > 0 (warm)
ahead of the bubble where space expands, θ < 0 (cool) behind where
space contracts.

Layout
------
- **Top centre**: title + metric equation (static)
- **Upper-left**: live v_s parameter display
- **Upper-right**: dual color legend (θ heatmap + σ² contour)
- **Lower-left**: contour annotation legend
- **Lower-right**: ω² = 0 (Frobenius) note

Usage::

    manim render -ql --format mp4 \\
        src/warpax/visualization/manim/_expansion_shear.py ExpansionShear
"""
from __future__ import annotations

import numpy as np
import matplotlib.cm as _mcm
from matplotlib.colors import LinearSegmentedColormap

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    UL,
    UR,
    WHITE,
    YELLOW,
    config,
    BackgroundRectangle,
    DashedVMobject,
    FadeIn,
    ImageMobject,
    MathTex,
    Rectangle,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    linear,
)

from warpax.visualization.manim._image_utils import (
    extract_zero_contour,
    frame_to_rgba,
)
from warpax.visualization.manim._scene_utils import COLORS_3B1B

# ---------------------------------------------------------------------------
# Dark-midpoint diverging colormap for θ: blue (contraction) -> dark -> red (expansion)
# ---------------------------------------------------------------------------
_DARK_DIVERGE_THETA = LinearSegmentedColormap.from_list(
    "dark_diverge_theta", ["#3B4CC0", "#1A1A2E", "#B40426"], N=256,
)
try:
    _mcm.register_cmap(name="dark_diverge_theta", cmap=_DARK_DIVERGE_THETA)
except (ValueError, AttributeError):
    try:
        import matplotlib as _mpl
        _mpl.colormaps.register(_DARK_DIVERGE_THETA, name="dark_diverge_theta")
    except Exception:
        pass


class ExpansionShear(Scene):
    """2D heatmap of expansion θ with shear σ² contour overlay.

    Dark-midpoint colormap, dual color legends, live
    parameter display, contour annotation legend.
    """

    def construct(self) -> None:
        # ==================================================================
        # Configuration
        # ==================================================================
        n_frames = 20
        v_start = 0.1
        v_end = 0.99
        cmap_name = "dark_diverge_theta"
        animation_duration = 12.0
        hold_duration = 1.0

        grid_n = 30
        grid_bounds = [(-3, 3), (-3, 3), (-3, 3)]

        img_height = config.frame_height - 2.2
        img_width = img_height

        self.camera.background_color = COLORS_3B1B["background"]

        # ==================================================================
        # Step 1: Compute kinematic scalars at each velocity
        # ==================================================================
        import equinox as eqx

        from warpax.analysis.kinematic_scalars import compute_kinematic_scalars_grid
        from warpax.benchmarks import AlcubierreMetric
        from warpax.geometry import GridSpec
        from warpax.visualization.common._frame_data import FrameData
        from warpax.visualization.common._physics import make_velocity_sweep

        grid_spec = GridSpec(bounds=grid_bounds, shape=(grid_n, grid_n, grid_n))
        v_values = make_velocity_sweep(v_start, v_end, n_frames)
        metric = AlcubierreMetric(v_s=v_start)

        X, Y, Z = grid_spec.meshgrid
        x_np, y_np, z_np = np.asarray(X), np.asarray(Y), np.asarray(Z)

        print("Computing kinematic scalars at each velocity...")
        frames: list[FrameData] = []

        try:
            from tqdm.auto import tqdm
            iterator = tqdm(list(enumerate(v_values)), desc="Kinematic sweep", unit="frame")
        except ImportError:
            iterator = list(enumerate(v_values))

        for i, v_s in iterator:
            metric_v = eqx.tree_at(lambda m: m.v_s, metric, v_s)
            theta_grid, sigma_sq_grid, omega_sq_grid = compute_kinematic_scalars_grid(
                metric_v, grid_spec, t=0.0,
            )
            theta_np = np.asarray(theta_grid)
            sigma_sq_np = np.asarray(sigma_sq_grid)
            max_abs = max(abs(float(np.nanmin(theta_np))),
                         abs(float(np.nanmax(theta_np))))
            if max_abs < 1e-15:
                max_abs = 1.0

            frame = FrameData(
                x=x_np, y=y_np, z=z_np,
                scalar_fields={"theta": theta_np, "sigma_sq": sigma_sq_np},
                metric_name=metric_v.name(),
                v_s=v_s,
                grid_shape=grid_spec.shape,
                t=0.0,
                colormaps={"theta": cmap_name, "sigma_sq": "inferno"},
                clim={"theta": (-max_abs, max_abs),
                      "sigma_sq": (0.0, float(np.nanmax(sigma_sq_np)))},
            )
            frames.append(frame)

        # ==================================================================
        # Step 2: Global fixed clim for theta
        # ==================================================================
        global_max_theta = 0.0
        for frame in frames:
            mid_z = frame.grid_shape[2] // 2
            theta_2d = frame.scalar_fields["theta"][:, :, mid_z]
            frame_max_theta = float(np.nanmax(np.abs(theta_2d)))
            if frame_max_theta > global_max_theta:
                global_max_theta = frame_max_theta
        if global_max_theta < 1e-15:
            global_max_theta = 1.0
        linthresh = max(global_max_theta * 0.01, 1e-10)
        global_clim_theta = (-global_max_theta, global_max_theta, linthresh)

        # ==================================================================
        # Step 3: Pre-render all RGBA arrays
        # ==================================================================
        print("Pre-rendering RGBA frames...")
        rgba_frames = [
            frame_to_rgba(f, "theta", global_clim_theta, cmap_name=cmap_name)
            for f in frames
        ]

        # ==================================================================
        # Step 4: Pre-extract shear + bubble contours
        # ==================================================================
        print("Extracting contours...")
        shear_contour_paths: list[list[np.ndarray]] = []
        bubble_contour_paths: list[list[np.ndarray]] = []

        for frame in frames:
            mid_z = frame.grid_shape[2] // 2
            sigma_2d = frame.scalar_fields["sigma_sq"][:, :, mid_z]
            x_1d = frame.x[:, 0, 0]
            y_1d = frame.y[0, :, 0]
            x_range = (float(x_1d[0]), float(x_1d[-1]))
            y_range = (float(y_1d[0]), float(y_1d[-1]))

            frame_max_sigma = float(np.nanmax(sigma_2d))
            shear_level = frame_max_sigma * 0.5
            if shear_level > 1e-15:
                sc = extract_zero_contour(
                    sigma_2d, x_range, y_range,
                    level=shear_level, scene_width=img_width,
                )
            else:
                sc = []
            shear_contour_paths.append(sc)

            from warpax.visualization.manim._image_utils import extract_bubble_contour
            bc = extract_bubble_contour(frame, level=0.5, scene_width=img_width)
            bubble_contour_paths.append(bc)

        # ==================================================================
        # Step 5: Static header - title + equation
        # ==================================================================
        title_text = Text(
            "Expansion θ and Shear σ²",
            font_size=28, color=WHITE, weight="LIGHT",
        )
        from warpax.visualization.manim._scene_utils import format_metric_equation
        equation = format_metric_equation("Alcubierre")
        equation.scale(0.6)
        header = VGroup(title_text, equation).arrange(DOWN, buff=0.1)
        header.to_edge(UP, buff=0.15)
        self.add(header)

        # ==================================================================
        # Step 6: Build scene layout
        # ==================================================================
        frame_idx = ValueTracker(0)
        heatmap_center = DOWN * 0.3

        def _make_heatmap():
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_frames - 1))
            img = ImageMobject(rgba_frames[idx])
            img.height = img_height
            img.move_to(heatmap_center)
            return img

        heatmap = always_redraw(_make_heatmap)

        def _make_shear_contour():
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_frames - 1))
            group = VGroup()
            for verts in shear_contour_paths[idx]:
                if len(verts) < 2:
                    continue
                vmob = VMobject(stroke_color=YELLOW, stroke_width=2)
                points_3d = [np.array([v[0], v[1], 0.0]) + heatmap_center for v in verts]
                vmob.set_points_as_corners(points_3d)
                vmob = DashedVMobject(vmob, num_dashes=25, dashed_ratio=0.5)
                group.add(vmob)
            return group

        shear_contour = always_redraw(_make_shear_contour)

        def _make_bubble_contour():
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_frames - 1))
            group = VGroup()
            for verts in bubble_contour_paths[idx]:
                if len(verts) < 2:
                    continue
                vmob = VMobject(stroke_color="#58C4DD", stroke_width=2.5)
                points_3d = [np.array([v[0], v[1], 0.0]) + heatmap_center for v in verts]
                vmob.set_points_as_corners(points_3d)
                group.add(vmob)
            return group

        bubble_contour = always_redraw(_make_bubble_contour)

        # ==================================================================
        # Step 7: Parameter display - upper-left
        # ==================================================================
        vs_mathtex = [
            MathTex(f"{f.v_s:.2f}", font_size=30, color=YELLOW)
            for f in frames
        ]

        def _make_param():
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_frames - 1))
            row = VGroup(
                MathTex(r"v_s", font_size=30, color=WHITE),
                MathTex(r"=", font_size=30, color=WHITE),
                vs_mathtex[idx].copy(),
            ).arrange(RIGHT, buff=0.08)
            row.to_corner(UL, buff=0.35)
            return row

        param_display = always_redraw(_make_param)

        # ==================================================================
        # Step 8: Dual color legends - upper-right
        # ==================================================================
        # θ heatmap legend --
        theta_colors = ["#3B4CC0", "#2A3377", "#1A1A2E", "#672015", "#B40426"]
        theta_strips = VGroup(*[
            Rectangle(
                width=0.24, height=0.08,
                fill_color=c, fill_opacity=0.95,
                stroke_width=0.3, stroke_color=WHITE,
            ) for c in theta_colors
        ]).arrange(RIGHT, buff=0)
        theta_title = Text(
            "θ (expansion)", font_size=14, color=WHITE, weight="LIGHT",
        )
        theta_lo = MathTex(r"-", font_size=16, color="#3B4CC0")
        theta_hi = MathTex(r"+", font_size=16, color="#B40426")
        theta_bar = VGroup(theta_lo, theta_strips, theta_hi).arrange(RIGHT, buff=0.05)
        theta_legend = VGroup(theta_title, theta_bar).arrange(
            DOWN, buff=0.06, aligned_edge=LEFT,
        )

        # σ² contour legend --
        shear_sample = DashedVMobject(
            VMobject(color=YELLOW, stroke_width=2.0).set_points_as_corners(
                [np.array([-0.3, 0, 0]), np.array([0.3, 0, 0])]
            ),
            num_dashes=6,
        )
        shear_label = Text(
            "σ² contour (50 % max)",
            font_size=14, color=YELLOW, weight="LIGHT",
        )
        shear_legend = VGroup(shear_sample, shear_label).arrange(RIGHT, buff=0.12)

        legend_group = VGroup(theta_legend, shear_legend).arrange(
            DOWN, buff=0.18, aligned_edge=LEFT,
        )
        legend_group.to_corner(UR, buff=0.3)
        legend_bg = BackgroundRectangle(
            legend_group, fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8, buff=0.1,
        )
        legend_with_bg = VGroup(legend_bg, legend_group)

        # ==================================================================
        # Step 9: Contour annotation - lower-left
        # ==================================================================
        solid_sample = VMobject(color="#58C4DD", stroke_width=2.5)
        solid_sample.set_points_as_corners(
            [np.array([-0.3, 0, 0]), np.array([0.3, 0, 0])]
        )
        solid_label = Text(
            "Bubble wall (f = 0.5)",
            font_size=14, color="#58C4DD", weight="LIGHT",
        )
        solid_row = VGroup(solid_sample, solid_label).arrange(RIGHT, buff=0.12)
        solid_row.to_corner(DOWN + LEFT, buff=0.25)
        solid_row_bg = BackgroundRectangle(
            solid_row, fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8, buff=0.1,
        )
        solid_row_group = VGroup(solid_row_bg, solid_row)

        # ω² note - lower-right --
        vorticity_note = Text(
            "ω² = 0 (Frobenius)",
            font_size=14, color=WHITE, weight="LIGHT",
        )
        vorticity_note.set_opacity(0.7)
        vorticity_note.to_corner(DOWN + RIGHT, buff=0.25)
        vorticity_bg = BackgroundRectangle(
            vorticity_note, fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8, buff=0.1,
        )
        vorticity_group = VGroup(vorticity_bg, vorticity_note)

        # ==================================================================
        # Step 10: Assemble and animate
        # ==================================================================
        self.add(heatmap, shear_contour, bubble_contour)
        self.add(param_display, legend_with_bg, solid_row_group, vorticity_group)

        self.play(FadeIn(heatmap), run_time=0.5)
        self.play(
            frame_idx.animate.set_value(n_frames - 1),
            run_time=animation_duration,
            rate_func=linear,
        )
        self.wait(hold_duration)
