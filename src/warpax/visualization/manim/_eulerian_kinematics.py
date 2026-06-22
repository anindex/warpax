"""EulerianKinematics2D: expansion θ heatmap with shear σ² and bubble-wall overlays.

Kinematic decomposition of the Eulerian congruence n^a (the unit normal to the
t = const slices): expansion θ = −K (trace of extrinsic curvature), shear σ², and
vorticity ω² ≡ 0 (hypersurface-orthogonal, by Frobenius).

Usage: manim render -ql --format mp4 \\
    src/warpax/visualization/manim/_eulerian_kinematics.py EulerianKinematics2D
"""

from __future__ import annotations

import numpy as np
import matplotlib as _mpl
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
    Group,
    ImageMobject,
    MathTex,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    linear,
)

from warpax.visualization.manim._image_utils import (
    extract_contours,
    frame_to_rgba,
)
from warpax.visualization.manim._scene_utils import COLORS_3B1B


_DARK_DIVERGE_THETA = LinearSegmentedColormap.from_list(
    "dark_diverge_theta",
    ["#3B4CC0", "#1A1A2E", "#B40426"],
    N=256,
)
try:
    _mpl.colormaps.register(_DARK_DIVERGE_THETA, name="dark_diverge_theta")
except ValueError:
    pass  # already registered


class EulerianKinematics2D(Scene):
    """2D heatmap of expansion θ = −K with shear σ² contour overlay.

    Kinematic scalars of the Eulerian congruence n^a. Dark-midpoint diverging
    colormap for the signed θ (expansion behind the ship, contraction in front),
    inferno σ² iso-contours, the f = 0.5 bubble wall, and a ω² ≡ 0 note.
    """

    def construct(self) -> None:
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

        import equinox as eqx

        from warpax.analysis.kinematic_scalars import compute_kinematic_scalars_grid
        from warpax.benchmarks import AlcubierreMetric
        from warpax.geometry import GridSpec
        from warpax.visualization.common._frame_data import FrameData
        from warpax.visualization.common._physics import (
            _shape_function_grid,
            make_velocity_sweep,
        )

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
                metric_v,
                grid_spec,
                t=0.0,
            )
            theta_np = np.asarray(theta_grid)
            sigma_sq_np = np.asarray(sigma_sq_grid)
            omega_sq_np = np.asarray(omega_sq_grid)
            max_abs = max(abs(float(np.nanmin(theta_np))), abs(float(np.nanmax(theta_np))))
            if not np.isfinite(max_abs) or max_abs < 1e-15:
                max_abs = 1.0
            sigma_max = float(np.nanmax(sigma_sq_np))
            if not np.isfinite(sigma_max) or sigma_max <= 0.0:
                sigma_max = 1.0

            # Real analytic shape function f(r_s), so the f = 0.5 wall overlay
            # is the genuine bubble wall rather than a heuristic circle.
            f_grid = _shape_function_grid(metric_v, grid_spec, 0.0)

            scalar_fields = {
                "theta": theta_np,
                "sigma_sq": sigma_sq_np,
                "omega_sq": omega_sq_np,
            }
            colormaps = {
                "theta": cmap_name,
                "sigma_sq": "inferno",
                "omega_sq": "inferno",
            }
            clim = {
                "theta": (-max_abs, max_abs),
                "sigma_sq": (0.0, sigma_max),
                "omega_sq": (0.0, 1.0),
            }
            if f_grid is not None:
                scalar_fields["shape_function"] = f_grid
                colormaps["shape_function"] = "viridis"
                clim["shape_function"] = (0.0, 1.0)

            frame = FrameData(
                x=x_np,
                y=y_np,
                z=z_np,
                scalar_fields=scalar_fields,
                metric_name=metric_v.name(),
                v_s=v_s,
                grid_shape=grid_spec.shape,
                t=0.0,
                colormaps=colormaps,
                clim=clim,
            )
            frames.append(frame)

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

        print("Pre-rendering RGBA frames...")
        rgba_frames = [
            frame_to_rgba(f, "theta", global_clim_theta, cmap_name=cmap_name) for f in frames
        ]

        print("Extracting contours...")
        # Multi-level shear σ² iso-contours at fixed *global* fractions of the
        # peak shear, so the family is stable across the velocity sweep and
        # reads as nested shells. Stronger-shear lines are drawn thicker/brighter.
        global_max_sigma = 0.0
        for frame in frames:
            mid_z = frame.grid_shape[2] // 2
            fm = float(np.nanmax(frame.scalar_fields["sigma_sq"][:, :, mid_z]))
            if np.isfinite(fm) and fm > global_max_sigma:
                global_max_sigma = fm
        if global_max_sigma < 1e-15:
            global_max_sigma = 1.0
        shear_fracs = [0.2, 0.4, 0.6, 0.8]
        shear_levels = sorted(global_max_sigma * fr for fr in shear_fracs)
        n_shear = len(shear_levels)
        # ``shear_levels`` ascends (weakest first); style index 0 is the faintest.
        shear_styles = [
            {
                "stroke_width": 1.0 + 1.0 * (i / max(n_shear - 1, 1)),
                "opacity": 0.5 + 0.5 * (i / max(n_shear - 1, 1)),
            }
            for i in range(n_shear)
        ]

        from warpax.visualization.manim._image_utils import extract_bubble_contour

        shear_contour_sets: list[list[tuple[float, list[np.ndarray]]]] = []
        bubble_contour_paths: list[list[np.ndarray]] = []

        for frame in frames:
            mid_z = frame.grid_shape[2] // 2
            sigma_2d = frame.scalar_fields["sigma_sq"][:, :, mid_z]
            x_1d = frame.x[:, 0, 0]
            y_1d = frame.y[0, :, 0]
            x_range = (float(x_1d[0]), float(x_1d[-1]))
            y_range = (float(y_1d[0]), float(y_1d[-1]))

            shear_contour_sets.append(
                extract_contours(
                    sigma_2d,
                    x_range,
                    y_range,
                    shear_levels,
                    scene_width=img_width,
                )
            )
            bc = extract_bubble_contour(frame, level=0.5, scene_width=img_width)
            bubble_contour_paths.append(bc)

        title_text = Text(
            "Expansion θ and Shear σ²",
            font_size=28,
            color=WHITE,
            weight="LIGHT",
        )
        from warpax.visualization.manim._scene_utils import format_metric_equation

        equation = format_metric_equation("Alcubierre")
        equation.scale(0.6)
        header = VGroup(title_text, equation).arrange(DOWN, buff=0.1)
        header.to_edge(UP, buff=0.15)
        self.add(header)

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
            for (_lvl, paths), style in zip(shear_contour_sets[idx], shear_styles):
                for verts in paths:
                    if len(verts) < 2:
                        continue
                    vmob = VMobject(
                        stroke_color=YELLOW,
                        stroke_width=style["stroke_width"],
                        stroke_opacity=style["opacity"],
                    )
                    points_3d = [np.array([v[0], v[1], 0.0]) + heatmap_center for v in verts]
                    vmob.set_points_as_corners(points_3d)
                    group.add(DashedVMobject(vmob, num_dashes=25, dashed_ratio=0.5))
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

        vs_mathtex = [MathTex(f"{f.v_s:.2f}", font_size=30, color=YELLOW) for f in frames]

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

        # θ heatmap legend (diverging; quantitative SymLog ticks) --
        from warpax.visualization.manim._scene_utils import make_colorbar_legend

        theta_legend = make_colorbar_legend(
            cmap_name,
            global_clim_theta[0],
            global_clim_theta[1],
            global_clim_theta[2],
            "θ = −K  (expansion of n^a)",
            bar_width=1.5,
        )

        # σ² contour legend --
        shear_sample = DashedVMobject(
            VMobject(color=YELLOW, stroke_width=2.0).set_points_as_corners(
                [np.array([-0.3, 0, 0]), np.array([0.3, 0, 0])]
            ),
            num_dashes=6,
        )
        shear_label = MathTex(
            r"\sigma^2 \equiv \sigma_{ab}\sigma^{ab}\ "
            r"\text{(iso-levels } 0.2\text{-}0.8 \times \text{max)}",
            font_size=24,
            color=YELLOW,
        )
        shear_legend = VGroup(shear_sample, shear_label).arrange(RIGHT, buff=0.12)

        legend_group = Group(theta_legend, shear_legend).arrange(
            DOWN,
            buff=0.18,
            aligned_edge=LEFT,
        )
        legend_group.to_corner(UR, buff=0.3).shift(DOWN * 1.15)
        legend_bg = BackgroundRectangle(
            legend_group,
            fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8,
            buff=0.1,
        )
        legend_with_bg = Group(legend_bg, legend_group)

        solid_sample = VMobject(color="#58C4DD", stroke_width=2.5)
        solid_sample.set_points_as_corners([np.array([-0.3, 0, 0]), np.array([0.3, 0, 0])])
        solid_label = Text(
            "Bubble wall (f = 0.5)",
            font_size=14,
            color="#58C4DD",
            weight="LIGHT",
        )
        solid_row = VGroup(solid_sample, solid_label).arrange(RIGHT, buff=0.12)
        solid_row.to_corner(DOWN + LEFT, buff=0.25)
        solid_row_bg = BackgroundRectangle(
            solid_row,
            fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8,
            buff=0.1,
        )
        solid_row_group = VGroup(solid_row_bg, solid_row)

        # ω² note - lower-right -- backed by the measured (stored) field.
        max_omega = max(
            (float(np.nanmax(np.abs(f.scalar_fields["omega_sq"]))) for f in frames),
            default=0.0,
        )
        omega_text = (
            "ω² = 0 (Frobenius)" if max_omega < 1e-12 else f"max ω² = {max_omega:.1e} (Frobenius)"
        )
        vorticity_note = Text(
            omega_text,
            font_size=14,
            color=WHITE,
            weight="LIGHT",
        )
        vorticity_note.set_opacity(0.7)
        vorticity_note.to_corner(DOWN + RIGHT, buff=0.25)
        vorticity_bg = BackgroundRectangle(
            vorticity_note,
            fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8,
            buff=0.1,
        )
        vorticity_group = VGroup(vorticity_bg, vorticity_note)

        from warpax.visualization.manim._scene_utils import (
            make_conventions_caption,
        )

        caption = make_conventions_caption()
        if caption.width > 7.0:
            caption.scale_to_fit_width(7.0)
        caption.to_edge(DOWN, buff=0.04)
        caption_group = VGroup(
            BackgroundRectangle(
                caption,
                fill_color=COLORS_3B1B["background"],
                fill_opacity=0.8,
                buff=0.05,
            ),
            caption,
        )

        self.add(heatmap, shear_contour, bubble_contour)
        self.add(param_display, legend_with_bg, solid_row_group, vorticity_group)
        self.add(caption_group)

        self.play(FadeIn(heatmap), run_time=0.5)
        self.play(
            frame_idx.animate.set_value(n_frames - 1),
            run_time=animation_duration,
            rate_func=linear,
        )
        self.wait(hold_duration)
