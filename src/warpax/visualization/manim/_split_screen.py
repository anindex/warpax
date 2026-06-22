"""EulerianVsWorstCaseNEC: Eulerian 6-null vs worst-case NEC margin panels.

Usage: manim render -ql --format mp4 \\
    src/warpax/visualization/manim/_split_screen.py EulerianVsWorstCaseNEC
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
    WHITE,
    YELLOW,
    BackgroundRectangle,
    DashedVMobject,
    Group,
    ImageMobject,
    Line,
    MathTex,
    Scene,
    Text,
    ValueTracker,
    VMobject,
    VGroup,
    always_redraw,
    linear,
)

from warpax.visualization.manim._image_utils import (
    compute_symlog_clim,
)
from warpax.visualization.manim._scene_utils import COLORS_3B1B


_DARK_DIVERGE_SS = LinearSegmentedColormap.from_list(
    "dark_diverge_ss",
    ["#3B4CC0", "#1A1A2E", "#B40426"],
    N=256,
)
try:
    _mpl.colormaps.register(_DARK_DIVERGE_SS, name="dark_diverge_ss")
except ValueError:
    pass  # already registered


def _contour_to_vmobject(
    paths: list[np.ndarray],
    color: str = WHITE,
    stroke_width: float = 2.0,
    dashed: bool = False,
) -> VGroup:
    """Convert contour paths to Manim VMobjects."""
    group = VGroup()
    for verts in paths:
        if len(verts) < 2:
            continue
        vmob = VMobject(color=color, stroke_width=stroke_width)
        points_3d = [np.array([v[0], v[1], 0.0]) for v in verts]
        vmob.set_points_as_corners(points_3d)
        if dashed:
            vmob = DashedVMobject(vmob, num_dashes=30)
        group.add(vmob)
    return group


def _build_eulerian_frames(
    metric_name: str = "Alcubierre",
    v_s_values: list[float] | None = None,
    grid_shape: tuple[int, int, int] = (30, 30, 30),
    bounds: list[tuple[float, float]] | None = None,
):
    """Build FrameData list with *genuine* Eulerian-observer NEC margins.

    Uses ``compute_eulerian_ec`` (the single normal-observer baseline:
    ``min`` over the six axis-aligned null rays of the Eulerian tetrad) rather
    than ``verify_grid`` (whose ``nec_margin`` is already the observer-robust
    worst-case). This makes the left panel a true Eulerian baseline, distinct
    from the dense observer-robust worst-case in the right panel -- the gap
    between them is exactly what observer-robust verification buys.
    """
    import equinox as eqx
    import jax

    from warpax.benchmarks import AlcubierreMetric
    from warpax.energy_conditions.verifier import compute_eulerian_ec
    from warpax.geometry import GridSpec
    from warpax.geometry.grid import evaluate_curvature_grid
    from warpax.visualization.common._conversion import (
        _symmetric_clim,
        eulerian_energy_density_grid,
    )
    from warpax.visualization.common._frame_data import FrameData
    from warpax.visualization.common._physics import _shape_function_grid

    if bounds is None:
        bounds = [(-3, 3), (-3, 3), (-3, 3)]
    if v_s_values is None:
        v_s_values = [0.5]

    grid_spec = GridSpec(bounds=bounds, shape=grid_shape)
    X, Y, Z = grid_spec.meshgrid
    x_np, y_np, z_np = np.asarray(X), np.asarray(Y), np.asarray(Z)
    metric = AlcubierreMetric(v_s=v_s_values[0])
    n_points = int(np.prod(grid_shape))

    frames = []
    for v_s in v_s_values:
        metric_t = eqx.tree_at(lambda m: m.v_s, metric, v_s)
        curv = evaluate_curvature_grid(
            metric_t,
            grid_spec,
            compute_invariants=True,
        )
        T_flat = curv.stress_energy.reshape(n_points, 4, 4)
        g_flat = curv.metric.reshape(n_points, 4, 4)
        ginv_flat = curv.metric_inv.reshape(n_points, 4, 4)
        ec = jax.vmap(compute_eulerian_ec)(T_flat, g_flat, ginv_flat)
        nec_eul = np.asarray(ec["nec"]).reshape(grid_shape)
        energy_density = eulerian_energy_density_grid(curv.stress_energy, curv.metric_inv)

        scalar_fields = {
            "nec_margin": nec_eul,
            "energy_density": energy_density,
        }
        colormaps = {"nec_margin": "RdBu_r", "energy_density": "RdBu_r"}
        clim = {
            "nec_margin": _symmetric_clim(nec_eul),
            "energy_density": _symmetric_clim(energy_density),
        }

        f_grid = _shape_function_grid(metric_t, grid_spec, 0.0)
        if f_grid is not None:
            scalar_fields["shape_function"] = f_grid
            colormaps["shape_function"] = "viridis"
            fmin, fmax = float(f_grid.min()), float(f_grid.max())
            clim["shape_function"] = (fmin, fmax if fmax > fmin else fmin + 1.0)

        frames.append(
            FrameData(
                x=x_np,
                y=y_np,
                z=z_np,
                scalar_fields=scalar_fields,
                metric_name=metric_name,
                v_s=v_s,
                grid_shape=grid_shape,
                t=0.0,
                colormaps=colormaps,
                clim=clim,
            )
        )
    return frames


def _build_robust_frames(
    metric_name: str = "Alcubierre",
    v_s_values: list[float] | None = None,
    grid_shape: tuple[int, int, int] = (30, 30, 30),
    bounds: list[tuple[float, float]] | None = None,
):
    """Build FrameData list with observer-robust NEC margins."""
    from warpax.benchmarks import AlcubierreMetric
    from warpax.geometry import GridSpec
    from warpax.visualization.common import build_ec_frame_sequence

    if bounds is None:
        bounds = [(-3, 3), (-3, 3), (-3, 3)]
    if v_s_values is None:
        v_s_values = [0.5]

    grid_spec = GridSpec(bounds=bounds, shape=grid_shape)
    metric = AlcubierreMetric(v_s=v_s_values[0])

    return build_ec_frame_sequence(
        metric,
        grid_spec,
        v_s_values=v_s_values,
        progress=True,
    )


class EulerianVsWorstCaseNEC(Scene):
    """Eulerian-frame vs worst-case NEC margin split-screen comparison.

    Left: NEC margin over the 6 axis-aligned Eulerian-frame nulls. Right:
    worst-case over a dense null sphere (k·n_Eul = −1). One-sided depth colormap,
    shared color legend, contour annotations, clean parameter display.
    """

    metric_name: str = "Alcubierre"
    n_frames: int = 12
    v_start: float = 0.1
    v_end: float = 0.99
    grid_shape: tuple[int, int, int] = (45, 45, 45)
    bounds: list[tuple[float, float]] = [(-3, 3), (-3, 3), (-3, 3)]
    panel_height: float = 3.8
    panel_width: float = 4.0
    run_time: float = 12.0

    def construct(self) -> None:
        self.camera.background_color = COLORS_3B1B["background"]
        cmap_name = "nec_depth"

        v_values = list(np.linspace(self.v_start, self.v_end, self.n_frames))

        eulerian_frames = _build_eulerian_frames(
            metric_name=self.metric_name,
            v_s_values=v_values,
            grid_shape=self.grid_shape,
            bounds=self.bounds,
        )
        robust_frames = _build_robust_frames(
            metric_name=self.metric_name,
            v_s_values=v_values,
            grid_shape=self.grid_shape,
            bounds=self.bounds,
        )

        n_total = min(len(eulerian_frames), len(robust_frames))

        # Both panels are NEC margins (<= 0): one-sided shared scale.
        eul_clim = compute_symlog_clim(eulerian_frames, "nec_margin", one_sided=True)
        rob_clim = compute_symlog_clim(
            robust_frames,
            "nec_margin_sweep",
            one_sided=True,
        )

        vmin = min(eul_clim[0], rob_clim[0])
        vmax = max(eul_clim[1], rob_clim[1])
        linthresh = min(eul_clim[2], rob_clim[2])
        shared_clim = (vmin, vmax, linthresh)

        def _render_panel_rgba(
            frame: object,
            field_name: str,
            clim: tuple,
        ) -> np.ndarray:
            """Render heatmap + contours into a single RGBA array."""
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            from scipy.ndimage import zoom as _ndzoom

            mid_z = frame.grid_shape[2] // 2
            data_2d = np.asarray(frame.scalar_fields[field_name][:, :, mid_z], dtype=float)
            # Replace NaN *before* zoom - bicubic interpolation propagates
            # even a single NaN to the entire upsampled array.
            data_2d = np.nan_to_num(data_2d, nan=0.0)
            vmin, vmax, lt = clim

            # Upsample 8x for smooth contour lines (matches _image_utils).
            # Transpose (Nx,Ny)->(Ny,Nx) so imshow(origin="lower") draws
            # physical x horizontal and y vertical (see B4 in _image_utils).
            # Bilinear (order=1), not bicubic: bicubic overshoots/rings on the
            # sharp wall ring, which the robust clim then renders as a visible
            # checkerboard. Bilinear upsampling does not ring.
            up = _ndzoom(data_2d.T, 8, order=1)
            norm = mcolors.SymLogNorm(linthresh=lt, vmin=vmin, vmax=vmax)
            cmap = _mpl.colormaps[cmap_name]
            rgba = cmap(norm(up))  # float [0,1]

            # Render contours into the array using matplotlib figure
            x_1d = frame.x[:, 0, 0]
            y_1d = frame.y[0, :, 0]
            x_range = (float(x_1d[0]), float(x_1d[-1]))
            y_range = (float(y_1d[0]), float(y_1d[-1]))

            dpi = 200
            fig_w = up.shape[1] / dpi
            fig_h = up.shape[0] / dpi
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            fig.patch.set_alpha(0.0)
            ax.set_position([0, 0, 1, 1])
            ax.set_facecolor("none")
            ax.imshow(
                rgba,
                extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                origin="lower",
                aspect="auto",
            )

            # Graded NEC depth contours (white dashed) at fixed global fractions
            # of the shared deepest margin, so both panels share the same levels.
            x_c = np.linspace(x_range[0], x_range[1], up.shape[1])
            y_c = np.linspace(y_range[0], y_range[1], up.shape[0])
            nec_levels = sorted(vmin * fr for fr in (0.1, 0.3, 0.5, 0.7, 0.9))
            n_lvl = len(nec_levels)
            lws = [1.1 - 0.6 * (i / max(n_lvl - 1, 1)) for i in range(n_lvl)]
            ax.contour(
                x_c,
                y_c,
                up,
                levels=nec_levels,
                colors="white",
                linewidths=lws,
                linestyles="dashed",
                alpha=0.85,
            )

            # Bubble wall contour (cyan solid). Bilinear (order=1) + clip to
            # [0, 1] keeps the f = 0.5 wall crisp and ring-free.
            if "shape_function" in frame.scalar_fields:
                sf_2d = np.asarray(
                    frame.scalar_fields["shape_function"][:, :, mid_z],
                    dtype=float,
                )
                sf_2d = np.clip(np.nan_to_num(sf_2d, nan=0.0), 0.0, 1.0)
                sf_up = _ndzoom(sf_2d.T, 8, order=1)  # match panel orientation
                ax.contour(
                    x_c,
                    y_c,
                    sf_up,
                    levels=[0.5],
                    colors="#58C4DD",
                    linewidths=1.0,
                    linestyles="solid",
                )
            else:
                # Circular fallback
                theta = np.linspace(0, 2 * np.pi, 128)
                r = (x_range[1] - x_range[0]) / 4.0
                ax.plot(
                    r * np.cos(theta),
                    r * np.sin(theta),
                    color="#58C4DD",
                    linewidth=1.2,
                )

            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            import io

            bio = io.BytesIO()
            fig.savefig(
                bio,
                format="png",
                dpi=dpi,
                transparent=True,
                pad_inches=0,
            )
            plt.close(fig)
            bio.seek(0)
            from PIL import Image as _PILImage

            pil_img = _PILImage.open(bio).convert("RGBA")
            return np.array(pil_img)

        eul_rgba_list = [_render_panel_rgba(f, "nec_margin", shared_clim) for f in eulerian_frames]
        rob_rgba_list = [
            _render_panel_rgba(f, "nec_margin_sweep", shared_clim) for f in robust_frames
        ]

        title_text = Text(
            "Eulerian vs Observer-Robust NEC Margin",
            font_size=24,
            color=WHITE,
            weight="LIGHT",
        )
        from warpax.visualization.manim._scene_utils import format_metric_equation

        equation = format_metric_equation(self.metric_name)
        equation.scale(0.55)
        header = VGroup(title_text, equation).arrange(DOWN, buff=0.08)
        header.to_edge(UP, buff=0.12)
        self.add(header)

        gap = 0.3
        panel_shift_down = DOWN * 0.35  # shift panels down to avoid header overlap
        left_center = LEFT * (self.panel_width / 2 + gap / 2) + panel_shift_down
        right_center = RIGHT * (self.panel_width / 2 + gap / 2) + panel_shift_down

        # Panel labels
        left_label = Text(
            "Eulerian",
            font_size=22,
            color=WHITE,
            weight="LIGHT",
        )
        left_label.move_to(left_center + UP * (self.panel_height / 2 + 0.3))

        right_label = Text(
            "Observer-Robust",
            font_size=22,
            color=WHITE,
            weight="LIGHT",
        )
        right_label.move_to(right_center + UP * (self.panel_height / 2 + 0.3))

        self.add(left_label, right_label)

        # Sub-labels
        left_sub = MathTex(
            r"\text{Eulerian-frame nulls (6 axes)}",
            font_size=20,
            color=YELLOW,
        )
        left_sub.move_to(left_center + DOWN * (self.panel_height / 2 + 0.25))

        right_sub = MathTex(
            r"\min_{\hat{k}\in S^2}\, T_{ab}k^a k^b,\ \ k\cdot n_{\rm Eul}=-1",
            font_size=20,
            color=YELLOW,
        )
        right_sub.move_to(right_center + DOWN * (self.panel_height / 2 + 0.25))

        self.add(left_sub, right_sub)

        # Vertical divider
        divider = Line(
            UP * (self.panel_height / 2 + 0.5),
            DOWN * (self.panel_height / 2 + 0.1),
            color=WHITE,
            stroke_width=1.5,
        )
        self.add(divider)

        frame_idx = ValueTracker(0)

        def _make_left():
            idx = max(0, min(int(frame_idx.get_value()), n_total - 1))
            img = ImageMobject(eul_rgba_list[idx])
            img.height = self.panel_height
            img.width = self.panel_width
            img.move_to(left_center)
            return img

        def _make_right():
            idx = max(0, min(int(frame_idx.get_value()), n_total - 1))
            img = ImageMobject(rob_rgba_list[idx])
            img.height = self.panel_height
            img.width = self.panel_width
            img.move_to(right_center)
            return img

        left_panel = always_redraw(_make_left)
        right_panel = always_redraw(_make_right)

        vs_mathtex = [MathTex(f"{v:.2f}", font_size=28, color=YELLOW) for v in v_values[:n_total]]

        def _make_vs():
            idx = max(0, min(int(frame_idx.get_value()), n_total - 1))
            row = VGroup(
                MathTex(r"v_s", font_size=28, color=WHITE),
                MathTex(r"=", font_size=28, color=WHITE),
                vs_mathtex[idx].copy(),
            ).arrange(RIGHT, buff=0.06)
            row.to_corner(DOWN + RIGHT, buff=0.15)
            return row

        vs_display = always_redraw(_make_vs)

        from warpax.visualization.manim._scene_utils import make_colorbar_legend

        color_legend = make_colorbar_legend(
            cmap_name,
            shared_clim[0],
            shared_clim[1],
            shared_clim[2],
            "NEC margin (shared; negative = violation)",
            bar_width=1.4,
        )
        color_legend.to_edge(DOWN, buff=0.5)
        cl_bg = BackgroundRectangle(
            color_legend,
            fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8,
            buff=0.1,
        )
        color_legend = Group(cl_bg, color_legend)

        dash_sample = DashedVMobject(
            VMobject(color=WHITE, stroke_width=1.5).set_points_as_corners(
                [np.array([-0.25, 0, 0]), np.array([0.25, 0, 0])]
            ),
            num_dashes=5,
        )
        dash_label = Text(
            "NEC depth",
            font_size=12,
            color=WHITE,
            weight="LIGHT",
        )
        dash_row = VGroup(dash_sample, dash_label).arrange(RIGHT, buff=0.08)

        solid_sample = VMobject(color="#58C4DD", stroke_width=2.0)
        solid_sample.set_points_as_corners([np.array([-0.25, 0, 0]), np.array([0.25, 0, 0])])
        solid_label = Text(
            "Bubble wall",
            font_size=12,
            color="#58C4DD",
            weight="LIGHT",
        )
        solid_row = VGroup(solid_sample, solid_label).arrange(RIGHT, buff=0.08)

        annot = VGroup(dash_row, solid_row).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        annot.to_corner(DOWN + LEFT, buff=0.15)
        annot_bg = BackgroundRectangle(
            annot,
            fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8,
            buff=0.1,
        )
        annot = VGroup(annot_bg, annot)

        from warpax.visualization.manim._scene_utils import (
            make_conventions_caption,
        )

        caption = make_conventions_caption()
        if caption.width > 7.0:
            caption.scale_to_fit_width(7.0)
        caption.to_edge(DOWN, buff=0.04)
        caption_grp = VGroup(
            BackgroundRectangle(
                caption,
                fill_color=COLORS_3B1B["background"],
                fill_opacity=0.8,
                buff=0.05,
            ),
            caption,
        )

        self.add(left_panel, right_panel)
        self.add(vs_display, color_legend, annot)
        self.add(caption_grp)

        self.play(
            frame_idx.animate.set_value(n_total - 1),
            run_time=self.run_time,
            rate_func=linear,
        )
        self.wait(2)
