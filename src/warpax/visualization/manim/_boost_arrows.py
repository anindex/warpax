"""BoostArrows: worst-case boost direction arrow field overlay.

A 2D ``Scene`` that overlays
directional arrows on an observer-robust NEC margin heatmap, showing
at each grid point which boost direction and magnitude produces the
worst-case energy condition violation.

The arrow direction is the spatial direction of the worst-case observer
boost projected onto the equatorial (x-y) plane.  Arrow **length**
encodes boost magnitude ``|sinh(ζ*)|`` and arrow **colour** encodes
the local NEC margin value (red = violated, green = satisfied),
matching the RdYlGn colorscale used in the BubbleCollapse heatmap
layer.

Arrows are subsampled (every 3-4 grid points) to avoid visual clutter.

Data pipeline: uses the observer sweep from
``build_ec_frame_sequence`` with explicit per-observer margin tracking
to identify which observer gives the worst margin at each point.

Usage::

    manim render -qm --format mp4 \\
        src/warpax/visualization/manim/_boost_arrows.py BoostArrows
"""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    UL,
    UR,
    WHITE,
    YELLOW,
    Arrow,
    BackgroundRectangle,
    DashedVMobject,
    Dot,
    FadeIn,
    ImageMobject,
    MathTex,
    Rectangle,
    Scene,
    Text,
    VGroup,
    VMobject,
)

import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from warpax.visualization.manim._image_utils import (
    compute_symlog_clim,
    extract_bubble_contour,
    extract_zero_contour,
    frame_to_rgba,
)
from warpax.visualization.manim._scene_utils import COLORS_3B1B

# Dark-midpoint diverging colormap: blue -> dark gray -> red.
_DARK_DIVERGE = LinearSegmentedColormap.from_list(
    "dark_diverge",
    ["#3B4CC0", "#1A1A2E", "#B40426"],
    N=256,
)
import matplotlib.cm as _mcm
try:
    _mcm.register_cmap(name="dark_diverge", cmap=_DARK_DIVERGE)
except (ValueError, AttributeError):
    # matplotlib >= 3.7 uses colormaps registry
    try:
        import matplotlib as _mpl
        _mpl.colormaps.register(_DARK_DIVERGE, name="dark_diverge")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _contour_to_vmobject(
    paths: list[np.ndarray],
    color: str = WHITE,
    stroke_width: float = 2.0,
    dashed: bool = False,
) -> VMobject | VGroup:
    """Convert contour path vertices to Manim VMobjects."""
    if not paths:
        return VGroup()

    group = VGroup()
    for verts in paths:
        if len(verts) < 2:
            continue
        path = VMobject(color=color, stroke_width=stroke_width)
        points_3d = [np.array([v[0], v[1], 0.0]) for v in verts]
        path.set_points_as_corners(points_3d)
        if dashed:
            path = DashedVMobject(path, num_dashes=30)
        group.add(path)
    return group


def _compute_boost_direction_field(
    metric_name: str = "Alcubierre",
    v_s: float = 0.5,
    grid_shape: tuple[int, int, int] = (30, 30, 30),
    bounds: list[tuple[float, float]] | None = None,
    n_rapidity: int = 12,
    n_directions: int = 6,
    zeta_max: float = 5.0,
):
    """Compute worst-case boost direction and magnitude at each grid point.

    Uses the observer sweep approach: evaluates NEC margins for a grid of
    (rapidity, direction) observer parameters and finds the argmin at each
    spatial point.

    Parameters
    ----------
    metric_name : str
        Metric to analyze (currently only ``"Alcubierre"`` supported).
    v_s : float
        Warp bubble velocity.
    grid_shape : tuple[int, int, int]
        Grid resolution.
    bounds : list[tuple[float, float]]
        Coordinate bounds.
    n_rapidity : int
        Number of rapidity samples.
    n_directions : int
        Number of angular direction samples.
    zeta_max : float
        Maximum rapidity.

    Returns
    -------
    tuple
        ``(frame, direction_2d, magnitude_2d)`` where:

        - ``frame`` is a FrameData with ``nec_margin_sweep``
        - ``direction_2d`` is ``(Nx, Ny, 2)`` unit direction vectors
        - ``magnitude_2d`` is ``(Nx, Ny)`` boost magnitudes ``|sinh(zeta*)|``
    """
    import jax.numpy as jnp

    from warpax.benchmarks import AlcubierreMetric
    from warpax.energy_conditions.sweep import sweep_nec_margins
    from warpax.geometry import GridSpec
    from warpax.geometry.grid import evaluate_curvature_grid
    from warpax.visualization.common._conversion import _symmetric_clim
    from warpax.visualization.common._frame_data import FrameData

    if bounds is None:
        bounds = [(-3, 3), (-3, 3), (-3, 3)]

    grid_spec = GridSpec(bounds=bounds, shape=grid_shape)
    metric = AlcubierreMetric(v_s=v_s)

    # Evaluate geometry
    result = evaluate_curvature_grid(
        metric, grid_spec, compute_invariants=True,
    )

    # Flatten for sweep
    N = int(np.prod(grid_spec.shape))
    T_flat = result.stress_energy.reshape(N, 4, 4)
    g_flat = result.metric.reshape(N, 4, 4)

    # Build a denser direction set for better angle coverage
    # (theta, phi) pairs on a hemisphere
    directions = []
    n_theta = max(int(np.sqrt(n_directions)), 2)
    n_phi = max(n_directions // n_theta, 2)
    for i_t in range(n_theta):
        theta = float(np.pi / 2 * (i_t / max(n_theta - 1, 1)))
        for i_p in range(n_phi):
            phi = float(2 * np.pi * i_p / n_phi)
            directions.append((theta, phi))
    # Ensure +x direction is included
    directions.append((float(np.pi / 2), 0.0))

    # Build full observer parameter set: (zeta, theta, phi)
    zetas = np.linspace(0.1, zeta_max, n_rapidity)  # skip zero rapidity
    obs_params_list = []
    for zeta in zetas:
        for theta, phi in directions:
            obs_params_list.append([float(zeta), theta, phi])

    obs_params = jnp.array(obs_params_list)
    nec_params = obs_params[:, 1:]  # (K, 2) for NEC

    # Sweep NEC margins: (N, K)
    nec_margins = sweep_nec_margins(T_flat, g_flat, nec_params)
    nec_np = np.asarray(nec_margins)  # (N, K)

    # Find worst observer per point (argmin across K observers)
    worst_k = np.argmin(nec_np, axis=-1)  # (N,)
    worst_margin = np.min(nec_np, axis=-1)  # (N,)

    # Extract direction and magnitude for worst observer at each point
    obs_params_np = np.asarray(obs_params)  # (K, 3)

    # Get (zeta, theta, phi) of worst observer per point
    worst_zeta = obs_params_np[worst_k, 0]  # (N,)
    worst_theta = obs_params_np[worst_k, 1]  # (N,)
    worst_phi = obs_params_np[worst_k, 2]  # (N,)

    # Convert to spatial direction vector (x, y components for equatorial plane)
    dir_x = np.sin(worst_theta) * np.cos(worst_phi)  # (N,)
    dir_y = np.sin(worst_theta) * np.sin(worst_phi)  # (N,)

    # Normalize direction in 2D
    dir_norm = np.sqrt(dir_x**2 + dir_y**2 + 1e-30)
    dir_x_hat = dir_x / dir_norm
    dir_y_hat = dir_y / dir_norm

    # Magnitude: |sinh(zeta_*)|
    magnitude = np.abs(np.sinh(worst_zeta))

    # Reshape to grid
    direction_2d = np.stack(
        [
            dir_x_hat.reshape(grid_shape),
            dir_y_hat.reshape(grid_shape),
        ],
        axis=-1,
    )  # (Nx, Ny, Nz, 2)

    magnitude_2d_full = magnitude.reshape(grid_shape)  # (Nx, Ny, Nz)

    # Extract equatorial slice
    mid_z = grid_shape[2] // 2
    dir_eq = direction_2d[:, :, mid_z, :]  # (Nx, Ny, 2)
    mag_eq = magnitude_2d_full[:, :, mid_z]  # (Nx, Ny)

    # Also compute worst_margin reshaped for the FrameData
    worst_margin_grid = worst_margin.reshape(grid_shape)
    energy_density = np.asarray(result.stress_energy[..., 0, 0])

    # Extract coordinates
    X, Y, Z = grid_spec.meshgrid
    x_np = np.asarray(X)
    y_np = np.asarray(Y)
    z_np = np.asarray(Z)

    scalar_fields = {
        "nec_margin_sweep": worst_margin_grid,
        "energy_density": energy_density,
    }
    colormaps = {
        "nec_margin_sweep": "RdBu_r",
        "energy_density": "RdBu_r",
    }
    clim = {
        "nec_margin_sweep": _symmetric_clim(worst_margin_grid),
        "energy_density": _symmetric_clim(energy_density),
    }

    frame = FrameData(
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

    return frame, dir_eq, mag_eq


def _make_arrow_field(
    direction_2d: np.ndarray,
    magnitude_2d: np.ndarray,
    nec_margin_2d: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    scene_width: float,
    scene_height: float,
    subsample: int = 4,
    arrow_scale: float = 0.3,
    min_magnitude: float = 0.05,
    max_arrow_length: float = 0.6,
) -> VGroup:
    """Create Manim Arrow mobjects coloured by local NEC margin.

    Parameters
    ----------
    direction_2d : np.ndarray
        ``(Nx, Ny, 2)`` unit direction vectors.
    magnitude_2d : np.ndarray
        ``(Nx, Ny)`` boost magnitudes.
    nec_margin_2d : np.ndarray
        ``(Nx, Ny)`` NEC margin values for colouring (negative = violated).
    x_range, y_range : tuple[float, float]
        Data coordinate ranges.
    scene_width, scene_height : float
        Size of the image in scene coordinates.
    subsample : int
        Sample every N-th grid point (default 4).
    arrow_scale : float
        Overall scale factor for arrow length.
    min_magnitude : float
        Skip arrows with magnitude below this fraction of max.
    max_arrow_length : float
        Clamp maximum arrow length in scene units.

    Returns
    -------
    VGroup
        Group of coloured Manim Arrow mobjects.
    """
    Nx, Ny = direction_2d.shape[:2]
    max_mag = float(np.max(magnitude_2d))
    if max_mag < 1e-15:
        return VGroup()

    # Normalize magnitudes for display
    norm_mag = magnitude_2d / max_mag  # [0, 1]

    # NEC margin -> colour via RdYlGn 5-stop scale
    nec_abs_max = max(abs(float(np.nanmin(nec_margin_2d))),
                      abs(float(np.nanmax(nec_margin_2d))),
                      1e-15)

    # 5-stop RdYlGn colour stops (same as BubbleCollapse heatmap)
    _RdYlGn_stops = [
        (-1.0, np.array([0.647, 0.000, 0.149])),   # #A50026
        (-0.4, np.array([0.957, 0.427, 0.263])),   # #F46D43
        ( 0.0, np.array([1.000, 1.000, 0.749])),   # #FFFFBF
        ( 0.4, np.array([0.400, 0.741, 0.388])),   # #66BD63
        ( 1.0, np.array([0.000, 0.408, 0.216])),   # #006837
    ]

    def _nec_to_hex(val: float) -> str:
        """Map a NEC margin value to an RdYlGn hex colour."""
        t = np.clip(val / nec_abs_max, -1.0, 1.0)
        # Piecewise interpolation through 5 stops
        for k in range(len(_RdYlGn_stops) - 1):
            t0, c0 = _RdYlGn_stops[k]
            t1, c1 = _RdYlGn_stops[k + 1]
            if t <= t1 or k == len(_RdYlGn_stops) - 2:
                frac = np.clip((t - t0) / (t1 - t0 + 1e-30), 0.0, 1.0)
                rgb = (1.0 - frac) * c0 + frac * c1
                r, g, b = (np.clip(rgb, 0, 1) * 255).astype(int)
                return f"#{r:02X}{g:02X}{b:02X}"
        return "#FFFFBF"  # fallback: yellow midpoint

    arrows = VGroup()

    for i in range(subsample // 2, Nx, subsample):
        for j in range(subsample // 2, Ny, subsample):
            mag = norm_mag[i, j]
            if mag < min_magnitude:
                continue

            dx, dy = direction_2d[i, j]

            # Map grid indices to scene coordinates
            sx = (i / (Nx - 1) - 0.5) * scene_width
            sy = (j / (Ny - 1) - 0.5) * scene_height

            # Arrow length proportional to normalized magnitude
            length = min(mag * arrow_scale, max_arrow_length)
            end_x = sx + dx * length
            end_y = sy + dy * length

            start = np.array([sx, sy, 0.0])
            end = np.array([end_x, end_y, 0.0])

            # Skip degenerate arrows
            if np.linalg.norm(end - start) < 0.01:
                continue

            # Colour by local NEC margin
            color = _nec_to_hex(float(nec_margin_2d[i, j]))

            # Opacity: stronger arrows more opaque
            opacity = float(0.5 + 0.5 * mag)

            arrow = Arrow(
                start, end,
                buff=0,
                stroke_width=2.0,
                tip_length=0.1,
                color=color,
                stroke_opacity=opacity,
                fill_opacity=opacity,
            )
            arrows.add(arrow)

    return arrows


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


class BoostArrows(Scene):
    """Worst-case boost direction arrow field overlay on NEC margin heatmap.

    A 2D ``Scene`` that displays the observer-robust NEC margin as a
    background heatmap and overlays a subsampled arrow field showing the
    direction and magnitude of the worst-case observer boost at each
    grid point.

    **Key insight communicated:** Arrows align with the propagation
    direction near the bubble wall, explaining why the robust margin
    finds deeper violations than the Eulerian observer.

    Arrows are coloured by local NEC margin (RdYlGn: red = violated,
    green = satisfied) so one can see at a glance *where* violations
    are worst and *which boost* triggers them.

    Static scene for a single velocity (v_s = 0.5 by default) with a
    brief fade-in of arrows.  Holds for 5 seconds total.
    """

    # Configurable class attributes
    metric_name: str = "Alcubierre"
    v_s: float = 0.5
    grid_shape: tuple[int, int, int] = (30, 30, 30)
    bounds: list[tuple[float, float]] = [(-3, 3), (-3, 3), (-3, 3)]
    subsample: int = 4
    image_height: float = 5.5
    image_width: float = 5.5

    def construct(self) -> None:
        # Background
        self.camera.background_color = COLORS_3B1B["background"]

        # ==============================================================
        # Step 1: Compute boost direction field
        # ==============================================================
        frame, dir_eq, mag_eq = _compute_boost_direction_field(
            metric_name=self.metric_name,
            v_s=self.v_s,
            grid_shape=self.grid_shape,
            bounds=self.bounds,
        )

        # ==============================================================
        # Step 2: Background heatmap (NEC margin, dark-midpoint diverging)
        # ==============================================================
        clim = compute_symlog_clim([frame], "nec_margin_sweep")
        rgba = frame_to_rgba(frame, "nec_margin_sweep", clim, cmap_name="dark_diverge")
        heatmap_img = ImageMobject(rgba)
        heatmap_img.height = self.image_height
        heatmap_img.width = self.image_width

        self.add(heatmap_img)

        # ==============================================================
        # Step 3: Contour overlays
        # ==============================================================
        mid_z = frame.grid_shape[2] // 2
        data_2d = frame.scalar_fields["nec_margin_sweep"][:, :, mid_z]
        nec_margin_eq = data_2d  # keep for arrow colouring
        x_1d = frame.x[:, 0, 0]
        y_1d = frame.y[0, :, 0]
        x_range = (float(x_1d[0]), float(x_1d[-1]))
        y_range = (float(y_1d[0]), float(y_1d[-1]))

        # Near-zero contour (margin ≈ 0): white dashed line.
        # The observer-swept NEC margin is everywhere ≤ 0 (worst-case is
        # always non-positive), so level=0.0 sits exactly on the data
        # boundary and yields an empty contour.  Use a small negative
        # threshold to mark the onset of significant violation.
        nec_min = float(np.min(data_2d))
        nec_threshold = nec_min * 1e-2 if nec_min < 0 else -1e-6
        zero_paths = extract_zero_contour(
            data_2d, x_range, y_range,
            level=nec_threshold, scene_width=self.image_width,
        )
        zero_contour = _contour_to_vmobject(
            zero_paths, color=WHITE, stroke_width=2.5, dashed=True,
        )
        self.add(zero_contour)

        # Bubble wall (f = 0.5): cyan solid outline
        bubble_paths = extract_bubble_contour(
            frame, scene_width=self.image_width,
        )
        bubble_contour = _contour_to_vmobject(
            bubble_paths, color="#58C4DD", stroke_width=2.5, dashed=False,
        )
        self.add(bubble_contour)

        # ==============================================================
        # Step 4: Arrow field overlay (coloured by NEC margin)
        # ==============================================================
        arrow_field = _make_arrow_field(
            dir_eq, mag_eq,
            nec_margin_2d=nec_margin_eq,
            x_range=x_range,
            y_range=y_range,
            scene_width=self.image_width,
            scene_height=self.image_height,
            subsample=self.subsample,
            arrow_scale=0.4,
            min_magnitude=0.05,
            max_arrow_length=0.6,
        )

        # ==============================================================
        # Step 5: Title + equation - top center
        # ==============================================================
        title_text = Text(
            "Worst-Case Boost Direction",
            font_size=28, color=WHITE, weight="LIGHT",
        )
        from warpax.visualization.manim._scene_utils import format_metric_equation
        equation = format_metric_equation(self.metric_name)
        equation.scale(0.6)
        header = VGroup(title_text, equation).arrange(DOWN, buff=0.1)
        header.to_edge(UP, buff=0.15)
        self.add(header)

        # ==============================================================
        # Step 6: Parameters - upper-left
        # ==============================================================
        v_label = MathTex(r"v_s", font_size=30, color=WHITE)
        v_eq = MathTex(r"=", font_size=30, color=WHITE)
        v_val = MathTex(f"{self.v_s}", font_size=30, color=YELLOW)
        v_row = VGroup(v_label, v_eq, v_val).arrange(RIGHT, buff=0.08)

        zeta_label = MathTex(r"\zeta_{\max}", font_size=30, color=WHITE)
        zeta_eq = MathTex(r"=", font_size=30, color=WHITE)
        zeta_val = MathTex(r"5", font_size=30, color=YELLOW)
        zeta_row = VGroup(zeta_label, zeta_eq, zeta_val).arrange(RIGHT, buff=0.08)

        param_block = VGroup(v_row, zeta_row).arrange(
            DOWN, buff=0.12, aligned_edge=LEFT,
        )
        param_block.to_corner(UL, buff=0.35)
        param_bg = BackgroundRectangle(
            param_block, fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8, buff=0.1,
        )
        self.add(VGroup(param_bg, param_block))

        # ==============================================================
        # Step 7: Colour legends - upper-right
        # ==============================================================

        # NEC margin heatmap legend (dark-midpoint diverging) --
        bg_colors = ["#3B4CC0", "#2A3377", "#1A1A2E", "#672015", "#B40426"]
        bg_strips = VGroup(*[
            Rectangle(
                width=0.28, height=0.10,
                fill_color=c, fill_opacity=0.95,
                stroke_width=0.3, stroke_color=WHITE,
            ) for c in bg_colors
        ]).arrange(RIGHT, buff=0)
        bg_title = Text(
            "NEC margin (background)",
            font_size=16, color=WHITE, weight="LIGHT",
        )
        bg_lo = MathTex(r"-", font_size=18, color="#3B4CC0")
        bg_hi = MathTex(r"+", font_size=18, color="#B40426")
        bg_bar_row = VGroup(bg_lo, bg_strips, bg_hi).arrange(RIGHT, buff=0.06)
        bg_legend = VGroup(bg_title, bg_bar_row).arrange(
            DOWN, buff=0.08, aligned_edge=LEFT,
        )

        # Arrow colour legend (RdYlGn) --
        arrow_colors = ["#A50026", "#F46D43", "#FFFFBF", "#66BD63", "#006837"]
        arrow_strips = VGroup(*[
            Rectangle(
                width=0.28, height=0.10,
                fill_color=c, fill_opacity=0.95,
                stroke_width=0.3, stroke_color=WHITE,
            ) for c in arrow_colors
        ]).arrange(RIGHT, buff=0)
        arrow_title = Text(
            "Arrow colour (NEC at point)",
            font_size=16, color=WHITE, weight="LIGHT",
        )
        arrow_lo = MathTex(
            r"\text{violated}", font_size=14, color="#A50026",
        )
        arrow_hi = MathTex(
            r"\text{satisfied}", font_size=14, color="#006837",
        )
        arrow_bar_row = VGroup(arrow_lo, arrow_strips, arrow_hi).arrange(
            RIGHT, buff=0.06,
        )
        arrow_legend = VGroup(arrow_title, arrow_bar_row).arrange(
            DOWN, buff=0.08, aligned_edge=LEFT,
        )

        legend_group = VGroup(bg_legend, arrow_legend).arrange(
            DOWN, buff=0.25, aligned_edge=LEFT,
        )
        legend_group.to_corner(UR, buff=0.35)
        legend_bg = BackgroundRectangle(
            legend_group, fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8, buff=0.1,
        )
        self.add(VGroup(legend_bg, legend_group))

        # ==============================================================
        # Step 8: Contour annotation - lower-left
        # ==============================================================
        # Dashed white line sample + label
        dash_sample = DashedVMobject(
            VMobject(color=WHITE, stroke_width=2.0).set_points_as_corners(
                [np.array([-0.3, 0, 0]), np.array([0.3, 0, 0])]
            ),
            num_dashes=6,
        )
        dash_label = Text(
            "NEC ≈ 0 boundary",
            font_size=14, color=WHITE, weight="LIGHT",
        )
        dash_row = VGroup(dash_sample, dash_label).arrange(RIGHT, buff=0.12)

        # Solid cyan line sample + label
        solid_sample = VMobject(color="#58C4DD", stroke_width=2.5)
        solid_sample.set_points_as_corners(
            [np.array([-0.3, 0, 0]), np.array([0.3, 0, 0])]
        )
        solid_label = Text(
            "Bubble wall (f = 0.5)",
            font_size=14, color="#58C4DD", weight="LIGHT",
        )
        solid_row = VGroup(solid_sample, solid_label).arrange(RIGHT, buff=0.12)

        # Arrow sample
        arrow_sample = Arrow(
            np.array([-0.2, 0, 0]), np.array([0.3, 0, 0]),
            buff=0, stroke_width=2.0, tip_length=0.08,
            color="#F46D43",
        )
        arrow_label = Text(
            "Boost direction & magnitude",
            font_size=14, color=WHITE, weight="LIGHT",
        )
        arrow_row = VGroup(arrow_sample, arrow_label).arrange(RIGHT, buff=0.12)

        contour_legend = VGroup(dash_row, solid_row, arrow_row).arrange(
            DOWN, buff=0.1, aligned_edge=LEFT,
        )
        contour_legend.to_corner(DOWN + LEFT, buff=0.3)
        contour_bg = BackgroundRectangle(
            contour_legend, fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8, buff=0.1,
        )
        self.add(VGroup(contour_bg, contour_legend))

        # ==============================================================
        # Step 9: Violation indicator - below params
        # ==============================================================
        nec_min = float(np.min(nec_margin_eq))
        dot_color = (COLORS_3B1B["violation_red"] if nec_min < 0
                     else COLORS_3B1B["safe_green"])
        status_dot = Dot(radius=0.08, color=dot_color)
        status_text = Text(
            "NEC violated" if nec_min < 0 else "NEC satisfied",
            font_size=14, color=dot_color, weight="LIGHT",
        )
        status_row = VGroup(status_dot, status_text).arrange(RIGHT, buff=0.1)
        status_row.next_to(param_block, DOWN, buff=0.2, aligned_edge=LEFT)
        self.add(status_row)

        # ==============================================================
        # Step 10: Fade in arrows + hold
        # ==============================================================
        self.play(FadeIn(arrow_field), run_time=1.5)
        self.wait(5.0)
