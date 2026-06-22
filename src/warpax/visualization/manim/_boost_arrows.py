"""Worst-case direction arrow fields over an energy-condition margin heatmap.

Two physically-distinct scenes share this module:

- ``WorstCaseNullDirections`` -- the Null Energy Condition. The null contraction
  ``T_{ab} k^a k^b`` (with ``k`` normalized to the local Eulerian frame,
  ``k . n_Eul = -1``) is rapidity-independent, so there is no "boost": the arrow
  shows the worst-case null *direction* and its length encodes the depth of the
  violation ``|min(0, T_{ab} k^a k^b)|``.
- ``WorstCaseBoostDirections`` -- the Weak Energy Condition. The worst-case over
  *unbounded* timelike boosts is -inf wherever the NEC is violated, so a
  rapidity-capped value would just report the cutoff. Instead the heatmap is the
  bounded invariant Type-I rest-frame WEC margin ``min(rho, rho+p_i)``, the arrow
  is the closed-form worst-boost direction ``e_{i*}``, and ``zeta_th`` (the
  threshold rapidity at which an observer first sees negative energy) is reported.

Usage:
    manim render -ql --format mp4 \\
        src/warpax/visualization/manim/_boost_arrows.py WorstCaseNullDirections
    manim render -ql --format mp4 \\
        src/warpax/visualization/manim/_boost_arrows.py WorstCaseBoostDirections
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
    Group,
    ImageMobject,
    MathTex,
    Scene,
    Text,
    VGroup,
    VMobject,
)

from matplotlib.colors import LinearSegmentedColormap

from warpax.visualization.manim._image_utils import (
    compute_symlog_clim,
    extract_bubble_contour,
    extract_zero_contour,
    frame_to_rgba,
)
from warpax.visualization.manim._scene_utils import (
    COLORS_3B1B,
    format_metric_equation,
)

# Dark-midpoint diverging colormap: blue -> dark gray -> red.
_DARK_DIVERGE = LinearSegmentedColormap.from_list(
    "dark_diverge",
    ["#3B4CC0", "#1A1A2E", "#B40426"],
    N=256,
)
import matplotlib as _mpl

try:
    _mpl.colormaps.register(_DARK_DIVERGE, name="dark_diverge")
except ValueError:
    pass  # already registered


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


def _worst_direction_field(
    quantity: str,
    *,
    metric_name: str = "Alcubierre",
    v_s: float = 0.5,
    grid_shape: tuple[int, int, int] = (30, 30, 30),
    bounds: list[tuple[float, float]] | None = None,
    n_theta: int = 13,
    n_phi: int = 24,
    n_rapidity: int = 16,
    zeta_max: float = 5.0,
):
    """Worst-case direction + strength per grid point, for NEC or WEC.

    Parameters
    ----------
    quantity : {"nec", "wec"}
        ``"nec"``: worst over the null sphere (theta, phi). The null
        contraction is rapidity-independent, so the encoded *strength* is the
        depth of the violation ``|min(0, T_{ab} k^a k^b)|``.
        ``"wec"``: worst over timelike boosts (zeta, theta, phi). The encoded
        *strength* is the WEC violation depth ``|min(0, T_{ab} u^a u^b)|`` at
        the worst boost, and is zero where WEC is satisfied (so no arrow is
        drawn there). The worst boost saturates at ``zeta_max``, so its
        ``|sinh(zeta*)|`` carries no spatial information - hence depth is used.

    Returns
    -------
    tuple
        ``(frame, dir_eq, strength_eq, field_name)`` where ``dir_eq`` is
        ``(Nx, Ny, 2)`` unit spatial directions on the z-midplane and
        ``strength_eq`` is ``(Nx, Ny)``.
    """
    import jax.numpy as jnp

    from warpax.benchmarks import AlcubierreMetric
    from warpax.energy_conditions.sweep import (
        make_angular_observers,
        sweep_nec_margins,
    )
    from warpax.geometry import GridSpec
    from warpax.geometry.grid import evaluate_curvature_grid
    from warpax.visualization.common._conversion import (
        _oneside_neg_clim,
        eulerian_energy_density_grid,
    )
    from warpax.visualization.common._frame_data import FrameData
    from warpax.visualization.common._physics import _shape_function_grid

    if bounds is None:
        bounds = [(-3, 3), (-3, 3), (-3, 3)]

    grid_spec = GridSpec(bounds=bounds, shape=grid_shape)
    metric = AlcubierreMetric(v_s=v_s)
    result = evaluate_curvature_grid(metric, grid_spec, compute_invariants=True)

    n_points = int(np.prod(grid_shape))
    T_flat = result.stress_energy.reshape(n_points, 4, 4)
    g_flat = result.metric.reshape(n_points, 4, 4)

    # Dense angular set covering the (hemi)sphere of directions.
    angular = np.asarray(make_angular_observers(n_theta=n_theta, n_phi=n_phi))
    thetas, phis = angular[:, 1], angular[:, 2]
    idx = np.arange(n_points)

    extra_fields: dict[str, np.ndarray] = {}
    if quantity == "nec":
        params = jnp.asarray(angular[:, 1:])  # (D, 2) null (theta, phi)
        margins = np.asarray(sweep_nec_margins(T_flat, g_flat, params))  # (N, D)
        worst_k = np.argmin(margins, axis=-1)
        worst_margin = margins[idx, worst_k]
        wt, wp = thetas[worst_k], phis[worst_k]
        dir_x = np.sin(wt) * np.cos(wp)
        dir_y = np.sin(wt) * np.sin(wp)
        strength = np.abs(np.minimum(worst_margin, 0.0))  # violation depth
        field_name = "nec_margin_sweep"
    elif quantity == "wec":
        # Cap-free WEC: the worst-case energy density over *unbounded* boosts is
        # -inf wherever NEC is violated, so a rapidity-capped min carries only the
        # cutoff. The invariant Type-I rest-frame WEC margin would be ideal, but
        # the Alcubierre wall is overwhelmingly Type IV (no matter rest frame), so
        # that margin -- and the threshold rapidity zeta_th -- are undefined there.
        # Plot instead the always-defined Eulerian energy density rho_Eul (the WEC
        # violation seen by the natural observer); show the closed-form worst-boost
        # direction only where the matter is genuinely Type I.
        from warpax.visualization.common._conversion import eulerian_wec_fields

        wf = eulerian_wec_fields(result.stress_energy, result.metric, result.metric_inv)
        rho_eul = eulerian_energy_density_grid(result.stress_energy, result.metric_inv)
        worst_margin = rho_eul.reshape(-1)
        boost = wf["boost_dir"].reshape(-1, 3)
        dir_x, dir_y = boost[:, 0], boost[:, 1]
        # Arrows only where matter is Type I (a well-defined rest frame exists)
        # and the Eulerian observer already sees a violation.
        typeI = wf["he_type"].reshape(-1) == 1.0
        strength = np.where(typeI, np.abs(np.minimum(worst_margin, 0.0)), 0.0)
        field_name = "energy_density"
        extra_fields["zeta_th"] = wf["zeta_th"]
        extra_fields["he_type"] = wf["he_type"]
    else:
        raise ValueError(f"quantity must be 'nec' or 'wec', got {quantity!r}")

    dnorm = np.sqrt(dir_x**2 + dir_y**2 + 1e-30)
    dir_x, dir_y = dir_x / dnorm, dir_y / dnorm

    worst_margin_grid = worst_margin.reshape(grid_shape)
    dir_grid = np.stack([dir_x.reshape(grid_shape), dir_y.reshape(grid_shape)], axis=-1)
    strength_grid = strength.reshape(grid_shape)

    mid_z = grid_shape[2] // 2
    dir_eq = dir_grid[:, :, mid_z, :]
    strength_eq = strength_grid[:, :, mid_z]

    energy_density = eulerian_energy_density_grid(result.stress_energy, result.metric_inv)
    X, Y, Z = grid_spec.meshgrid
    # Both margin fields are <= 0 -> one-sided depth scale (honest, no false
    # "satisfied" half).
    scalar_fields = {
        field_name: worst_margin_grid,
        "energy_density": energy_density,
        **extra_fields,
    }
    colormaps = {field_name: "nec_depth", "energy_density": "nec_depth"}
    clim = {
        field_name: _oneside_neg_clim(worst_margin_grid),
        "energy_density": _oneside_neg_clim(energy_density),
    }

    f_grid = _shape_function_grid(metric, grid_spec, 0.0)
    if f_grid is not None:
        scalar_fields["shape_function"] = f_grid
        colormaps["shape_function"] = "viridis"
        fmin, fmax = float(f_grid.min()), float(f_grid.max())
        clim["shape_function"] = (fmin, fmax if fmax > fmin else fmin + 1.0)

    frame = FrameData(
        x=np.asarray(X),
        y=np.asarray(Y),
        z=np.asarray(Z),
        scalar_fields=scalar_fields,
        metric_name=metric_name,
        v_s=v_s,
        grid_shape=grid_shape,
        t=0.0,
        colormaps=colormaps,
        clim=clim,
    )
    return frame, dir_eq, strength_eq, field_name


def _make_arrow_field(
    direction_2d: np.ndarray,
    strength_2d: np.ndarray,
    scene_width: float,
    scene_height: float,
    *,
    color: str = YELLOW,
    subsample: int = 4,
    max_arrow_length: float = 0.55,
    min_arrow_length: float = 0.15,
    min_frac: float = 0.06,
) -> VGroup:
    """Uniform-color arrow field; length and opacity encode local strength.

    Arrow direction is the worst-case (null or boost) spatial direction and
    length scales with ``strength_2d / max(strength)``. A single accent color
    is used so the arrows never appear to contradict the heatmap colormap
    (a separate diverging scale for the margin field).
    """
    Nx, Ny = direction_2d.shape[:2]
    max_s = float(np.max(strength_2d))
    if max_s < 1e-15:
        return VGroup()

    arrows = VGroup()
    for i in range(subsample // 2, Nx, subsample):
        for j in range(subsample // 2, Ny, subsample):
            s = float(strength_2d[i, j]) / max_s
            if s < min_frac:
                continue
            dx, dy = direction_2d[i, j]
            # Canonical orientation: x-index -> horizontal, y-index -> vertical
            # up (matches frame_to_rgba / extract_zero_contour after B4).
            sx = (i / (Nx - 1) - 0.5) * scene_width
            sy = (j / (Ny - 1) - 0.5) * scene_height
            length = min(
                min_arrow_length + s * (max_arrow_length - min_arrow_length),
                max_arrow_length,
            )
            start = np.array([sx, sy, 0.0])
            end = np.array([sx + dx * length, sy + dy * length, 0.0])
            if np.linalg.norm(end - start) < 0.02:
                continue
            arrows.add(
                Arrow(
                    start,
                    end,
                    buff=0,
                    stroke_width=2.2,
                    max_tip_length_to_length_ratio=0.35,
                    color=color,
                    stroke_opacity=float(0.45 + 0.55 * s),
                    fill_opacity=float(0.45 + 0.55 * s),
                )
            )
    return arrows


class _ArrowFieldScene(Scene):
    """Shared construction for the NEC null-direction / WEC boost scenes."""

    # Overridden by subclasses
    quantity: str = "nec"
    title: str = "Worst-Case Direction"
    ec_label: str = r"\text{NEC}"
    arrow_desc: str = "worst direction; length = strength"

    metric_name: str = "Alcubierre"
    v_s: float = 0.5
    grid_shape: tuple[int, int, int] = (30, 30, 30)
    bounds: list[tuple[float, float]] = [(-3, 3), (-3, 3), (-3, 3)]
    subsample: int = 4
    image_height: float = 5.5
    image_width: float = 5.5

    def construct(self) -> None:
        self.camera.background_color = COLORS_3B1B["background"]

        frame, dir_eq, strength_eq, field_name = _worst_direction_field(
            self.quantity,
            metric_name=self.metric_name,
            v_s=self.v_s,
            grid_shape=self.grid_shape,
            bounds=self.bounds,
        )

        # NEC/WEC worst-case margins are <= 0; use the one-sided sequential
        # ramp so the colour range encodes violation depth honestly.
        clim = compute_symlog_clim([frame], field_name, one_sided=True)
        rgba = frame_to_rgba(frame, field_name, clim, cmap_name="nec_depth")
        heatmap_img = ImageMobject(rgba)
        heatmap_img.height = self.image_height
        heatmap_img.width = self.image_width
        self.add(heatmap_img)

        mid_z = frame.grid_shape[2] // 2
        data_2d = frame.scalar_fields[field_name][:, :, mid_z]
        x_1d = frame.x[:, 0, 0]
        y_1d = frame.y[0, :, 0]
        x_range = (float(x_1d[0]), float(x_1d[-1]))
        y_range = (float(y_1d[0]), float(y_1d[-1]))

        # Margin = 0 boundary. For WEC this is a genuine violation boundary;
        # for NEC the worst-case margin is <= 0 everywhere, so mark the onset
        # of significant violation with a small negative threshold instead.
        margin_min = float(np.min(data_2d))
        if self.quantity == "nec":
            level = margin_min * 1e-2 if margin_min < 0 else -1e-6
            boundary_label = r"\text{NEC} \approx 0"
        else:
            level = 0.0
            boundary_label = r"\text{WEC} = 0"
        zero_paths = extract_zero_contour(
            data_2d,
            x_range,
            y_range,
            level=level,
            scene_width=self.image_width,
            scene_height=self.image_height,
        )
        self.add(
            _contour_to_vmobject(
                zero_paths,
                color=WHITE,
                stroke_width=2.5,
                dashed=True,
            )
        )

        # Real f = 0.5 bubble wall (cyan solid).
        bubble_paths = extract_bubble_contour(
            frame,
            scene_width=self.image_width,
            scene_height=self.image_height,
        )
        self.add(
            _contour_to_vmobject(
                bubble_paths,
                color="#58C4DD",
                stroke_width=2.5,
                dashed=False,
            )
        )

        arrow_field = _make_arrow_field(
            dir_eq,
            strength_eq,
            scene_width=self.image_width,
            scene_height=self.image_height,
            subsample=self.subsample,
        )

        # Header
        title_text = Text(self.title, font_size=28, color=WHITE, weight="LIGHT")
        equation = format_metric_equation(self.metric_name)
        equation.scale(0.6)
        header = VGroup(title_text, equation).arrange(DOWN, buff=0.1)
        header.to_edge(UP, buff=0.15)
        self.add(header)

        # Parameter block
        v_row = VGroup(
            MathTex(r"v_s", font_size=30, color=WHITE),
            MathTex(r"=", font_size=30, color=WHITE),
            MathTex(f"{self.v_s}", font_size=30, color=YELLOW),
        ).arrange(RIGHT, buff=0.08)
        slice_lbl = MathTex(r"z = 0\ \text{plane}", font_size=24, color=WHITE)
        param_block = VGroup(v_row, slice_lbl).arrange(
            DOWN,
            buff=0.12,
            aligned_edge=LEFT,
        )
        param_block.to_corner(UL, buff=0.35)
        self.add(
            VGroup(
                BackgroundRectangle(
                    param_block,
                    fill_color=COLORS_3B1B["background"],
                    fill_opacity=0.8,
                    buff=0.1,
                ),
                param_block,
            )
        )

        # Quantitative margin colorbar (one-sided ramp; negative = violation, 0 = marginal).
        from warpax.visualization.manim._scene_utils import make_colorbar_legend

        margin_title = (
            "NEC margin (worst null)"
            if self.quantity == "nec"
            else "Eulerian energy density ρ_Eul (≤ 0)"
        )
        cbar = make_colorbar_legend(
            "nec_depth",
            clim[0],
            clim[1],
            clim[2],
            margin_title,
        )
        arrow_legend = VGroup(
            Text("Arrow (yellow):", font_size=15, color=WHITE, weight="LIGHT"),
            Text(self.arrow_desc, font_size=13, color=YELLOW, weight="LIGHT"),
        ).arrange(DOWN, buff=0.06, aligned_edge=LEFT)

        legend_group = Group(cbar, arrow_legend).arrange(
            DOWN,
            buff=0.22,
            aligned_edge=LEFT,
        )
        legend_group.to_corner(UR, buff=0.35).shift(DOWN * 0.5)
        self.add(
            Group(
                BackgroundRectangle(
                    legend_group,
                    fill_color=COLORS_3B1B["background"],
                    fill_opacity=0.8,
                    buff=0.1,
                ),
                legend_group,
            )
        )

        # Contour legend (lower-left)
        dash_sample = DashedVMobject(
            VMobject(color=WHITE, stroke_width=2.0).set_points_as_corners(
                [np.array([-0.3, 0, 0]), np.array([0.3, 0, 0])]
            ),
            num_dashes=6,
        )
        dash_row = VGroup(
            dash_sample,
            MathTex(boundary_label, font_size=16, color=WHITE),
        ).arrange(RIGHT, buff=0.12)
        solid_sample = VMobject(color="#58C4DD", stroke_width=2.5)
        solid_sample.set_points_as_corners([np.array([-0.3, 0, 0]), np.array([0.3, 0, 0])])
        solid_row = VGroup(
            solid_sample,
            Text("Bubble wall (f = 0.5)", font_size=14, color="#58C4DD", weight="LIGHT"),
        ).arrange(RIGHT, buff=0.12)
        contour_legend = VGroup(dash_row, solid_row).arrange(
            DOWN,
            buff=0.1,
            aligned_edge=LEFT,
        )
        contour_legend.to_corner(DOWN + LEFT, buff=0.3)
        self.add(
            VGroup(
                BackgroundRectangle(
                    contour_legend,
                    fill_color=COLORS_3B1B["background"],
                    fill_opacity=0.8,
                    buff=0.1,
                ),
                contour_legend,
            )
        )

        # Violation status, evaluated on the rendered z = 0 slice.
        margin_min = float(np.min(data_2d))
        violated = margin_min < 0
        dot_color = COLORS_3B1B["violation_red"] if violated else COLORS_3B1B["safe_green"]
        ec = "NEC" if self.quantity == "nec" else "WEC"
        status_row = VGroup(
            Dot(radius=0.08, color=dot_color),
            Text(
                f"{ec} {'violated' if violated else 'satisfied'} (z=0 slice)",
                font_size=14,
                color=dot_color,
                weight="LIGHT",
            ),
        ).arrange(RIGHT, buff=0.1)
        status_row.next_to(param_block, DOWN, buff=0.2, aligned_edge=LEFT)
        self.add(status_row)

        # The worst boost is unbounded (-> -inf, = NEC), and the
        # wall matter is overwhelmingly Type IV (no rest frame), so the rest-frame
        # WEC margin and threshold rapidity zeta_th are undefined there. Report the
        # Type-IV fraction among the matter (non-vacuum) points only.
        if self.quantity == "wec" and "he_type" in frame.scalar_fields:
            _mz = frame.grid_shape[2] // 2
            he2d = frame.scalar_fields["he_type"][:, :, _mz]
            rho2d = np.abs(frame.scalar_fields["energy_density"][:, :, _mz])
            matter = rho2d > 1e-6 * (float(np.nanmax(rho2d)) + 1e-30)
            t4 = float(np.mean((he2d == 4.0)[matter])) if matter.any() else 0.0
            wec_note = Text(
                "ρ_Eul ≤ 0   ·   worst over unbounded boosts → −∞ (≡ NEC)   ·   "
                f"wall matter ~{t4 * 100:.0f}% Type-IV "
                "(no rest frame ⇒ rest-frame margin & ζ_th undefined)",
                font_size=11,
                color=YELLOW,
                weight="LIGHT",
            )
            wec_note.set_opacity(0.9)
            if wec_note.width > 7.5:
                wec_note.scale_to_fit_width(7.5)
            wec_note.next_to(status_row, DOWN, buff=0.15, aligned_edge=LEFT)
            self.add(
                VGroup(
                    BackgroundRectangle(
                        wec_note,
                        fill_color=COLORS_3B1B["background"],
                        fill_opacity=0.8,
                        buff=0.08,
                    ),
                    wec_note,
                )
            )

        from warpax.visualization.manim._scene_utils import (
            make_conventions_caption,
        )

        caption = make_conventions_caption()
        if caption.width > 7.0:
            caption.scale_to_fit_width(7.0)
        caption.to_edge(DOWN, buff=0.04)
        self.add(
            VGroup(
                BackgroundRectangle(
                    caption,
                    fill_color=COLORS_3B1B["background"],
                    fill_opacity=0.8,
                    buff=0.05,
                ),
                caption,
            )
        )

        self.play(FadeIn(arrow_field), run_time=1.5)
        self.wait(5.0)


class WorstCaseNullDirections(_ArrowFieldScene):
    """Worst-case null direction over the NEC margin (NEC is boost-independent).

    Arrow direction = worst null ray; arrow length = depth of the NEC violation.
    The robust NEC margin is non-positive everywhere for the Alcubierre bubble (0
    in flat regions, strictly negative in the wall), so violation means strictly
    negative; the arrows reveal *which* null direction is worst and *where* the
    violation is deepest. Null rays are normalized to the Eulerian frame
    (k . n_Eul = -1) so the depth is comparable across the grid.
    """

    quantity = "nec"
    title = "Worst-Case Null Direction (NEC)"
    ec_label = r"\text{NEC}"
    arrow_desc = "worst null direction; length = |min T k k|"


class WorstCaseBoostDirections(_ArrowFieldScene):
    """Worst-case timelike boost over the bounded WEC margin.

    Heatmap = invariant Type-I rest-frame WEC margin min(rho, rho+p_i) (bounded,
    cap-free). Arrow direction = closed-form worst-boost direction e_{i*}, drawn
    only where matter is Type I and WEC is violated. The worst case over
    *unbounded* boosts is -inf (= NEC), so the rest-frame margin and the
    threshold rapidity -- not a rapidity-capped value -- carry the physics.
    """

    quantity = "wec"
    title = "Worst-Case Boost (WEC)"
    ec_label = r"\text{WEC}"
    arrow_desc = "worst boost direction (Type-I points only); length = |ρ_Eul|"
