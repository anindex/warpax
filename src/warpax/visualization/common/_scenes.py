"""Showcase scene builders and post-render overlay helpers.

Scene builders produce lists of FrameData for the three signature animations:
- ``scene_bubble_collapse``: Alcubierre v_s ramps down to near-flat spacetime.
- ``scene_velocity_ramp``: EC violations intensify as v_s sweeps 0.1 to 0.99.
- ``scene_observer_sweep``: Per-rapidity EC margins on fixed Alcubierre geometry.

Overlay helpers burn text and watermarks onto RGBA screenshot arrays:
- ``add_text_overlay``: Title + parameter value text at configurable position.
- ``add_watermark``: Subtle semi-transparent branding in bottom-right corner.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Post-render overlay helpers
# ---------------------------------------------------------------------------


def add_text_overlay(
    img: np.ndarray,
    text: str,
    position: str = "upper_left",
    font_color: tuple[int, ...] = (255, 255, 255, 255),
    font_size: int = 24,
) -> np.ndarray:
    """Burn text overlay onto an RGBA screenshot array.

    Parameters
    ----------
    img : np.ndarray
        RGBA image array of shape ``(H, W, 4)`` with dtype ``uint8``.
    text : str
        Text to overlay (supports multi-line via ``\\n``).
    position : str
        Anchor position: ``"upper_left"``, ``"upper_right"``, ``"lower_right"``.
    font_color : tuple[int, ...]
        RGBA color tuple (default white opaque).
    font_size : int
        Approximate font size in pixels (default 24). Uses PIL default font
        scaled via ``font_size`` parameter where available.

    Returns
    -------
    np.ndarray
        New RGBA array with text burned in (input is not mutated).
    """
    result = img.copy()
    pil_img = Image.fromarray(result, mode="RGBA")
    draw = ImageDraw.Draw(pil_img)

    # Use PIL default font always available, no external font files
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        # Older Pillow without size parameter
        from PIL import ImageFont
        font = ImageFont.load_default()

    # Measure text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    h, w = img.shape[:2]
    margin = 20

    if position == "upper_left":
        xy = (margin, margin)
    elif position == "upper_right":
        xy = (w - text_w - margin, margin)
    elif position == "lower_right":
        xy = (w - text_w - margin, h - text_h - margin)
    else:
        raise ValueError(
            f"Unknown position {position!r}. "
            "Use 'upper_left', 'upper_right', or 'lower_right'."
        )

    draw.text(xy, text, fill=font_color, font=font)
    return np.array(pil_img)


def add_watermark(
    img: np.ndarray,
    text: str = "WarpAx",
    alpha: int = 100,
) -> np.ndarray:
    """Add a subtle semi-transparent watermark in the bottom-right corner.

    Parameters
    ----------
    img : np.ndarray
        RGBA image array of shape ``(H, W, 4)`` with dtype ``uint8``.
    text : str
        Watermark text (default ``"WarpAx"``).
    alpha : int
        Transparency level (0 = invisible, 255 = opaque, default 100).

    Returns
    -------
    np.ndarray
        New RGBA array with watermark composited (input is not mutated).
    """
    result = img.copy()
    pil_img = Image.fromarray(result, mode="RGBA")

    # Create transparent overlay for compositing
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Small font for subtle watermark
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=14)
    except TypeError:
        from PIL import ImageFont
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    h, w = img.shape[:2]
    offset = 10
    xy = (w - text_w - offset, h - text_h - offset)

    draw.text(xy, text, fill=(255, 255, 255, alpha), font=font)

    # Alpha-composite overlay onto image
    composited = Image.alpha_composite(pil_img, overlay)
    return np.array(composited)


# ---------------------------------------------------------------------------
# Scene builders (require JAX/Equinox indirectly through build_frame_sequence)
# ---------------------------------------------------------------------------


def scene_bubble_collapse(
    grid_spec,
    *,
    n_frames: int = 90,
    v_max: float = 0.5,
    v_min: float = 0.01,
) -> list:
    """Build frame sequence for Alcubierre bubble collapse.

    The warp bubble velocity ramps down from *v_max* toward flat spacetime,
    with curvature diminishing as v_s approaches zero.

    Parameters
    ----------
    grid_spec : GridSpec
        Spatial grid specification.
    n_frames : int
        Number of animation frames (default 90).
    v_max : float
        Starting velocity (default 0.5).
    v_min : float
        Minimum velocity floor to avoid degenerate zero curvature (default 0.01).

    Returns
    -------
    list[FrameData]
        Frame sequence with decreasing v_s.
    """
    from warpax.benchmarks import AlcubierreMetric

    from ._physics import build_frame_sequence, collapse_profile

    metric = AlcubierreMetric(v_s=v_max)

    t_values = list(np.linspace(0.0, 1.0, n_frames))
    v_values = [
        max(collapse_profile(t, v_max=v_max), v_min)
        for t in t_values
    ]

    return build_frame_sequence(
        metric,
        grid_spec,
        v_s_values=v_values,
        t_values=t_values,
        progress=True,
    )


def scene_velocity_ramp(
    grid_spec,
    *,
    n_frames: int = 90,
    v_start: float = 0.1,
    v_end: float = 0.99,
) -> list:
    """Build EC-enriched frame sequence for velocity ramp-up.

    The warp bubble velocity sweeps from *v_start* to *v_end*, producing
    frames with observer-swept WEC and NEC margins that show violations
    intensifying with speed.

    Parameters
    ----------
    grid_spec : GridSpec
        Spatial grid specification.
    n_frames : int
        Number of animation frames (default 90).
    v_start : float
        Starting velocity (default 0.1).
    v_end : float
        Ending velocity (default 0.99).

    Returns
    -------
    list[FrameData]
        Frame sequence with wec_margin_sweep and nec_margin_sweep fields.
    """
    from warpax.benchmarks import AlcubierreMetric

    from ._physics import build_ec_frame_sequence, make_velocity_sweep

    metric = AlcubierreMetric(v_s=v_start)
    v_values = make_velocity_sweep(v_start, v_end, n_frames)

    return build_ec_frame_sequence(
        metric,
        grid_spec,
        v_s_values=v_values,
        progress=True,
    )


def scene_observer_sweep(
    grid_spec,
    *,
    n_frames: int = 60,
    v_s: float = 0.5,
    zeta_max: float = 5.0,
    n_directions: int = 3,
) -> list:
    """Build per-rapidity EC margin frames on fixed Alcubierre geometry.

    Unlike ``build_ec_frame_sequence`` which takes worst-case across ALL
    observers, this scene shows the WEC/NEC margin landscape for a SINGLE
    rapidity at each frame.  The geometry is computed once and reused,
    revealing how the violation landscape shifts with observer boost.

    Parameters
    ----------
    grid_spec : GridSpec
        Spatial grid specification.
    n_frames : int
        Number of animation frames (default 60).
    v_s : float
        Fixed warp velocity (default 0.5).
    zeta_max : float
        Maximum rapidity, matching the paper's cap (default 5.0).
    n_directions : int
        Number of boost directions per rapidity (default 3: +x, +y, +z).

    Returns
    -------
    list[FrameData]
        Frame sequence where each frame's ``t`` field carries the rapidity
        value for annotation (geometry is identical across frames).
    """
    import jax.numpy as jnp

    from warpax.benchmarks import AlcubierreMetric
    from warpax.energy_conditions.sweep import sweep_nec_margins, sweep_wec_margins
    from warpax.geometry.grid import evaluate_curvature_grid

    from ._conversion import _symmetric_clim
    from ._frame_data import FrameData

    metric = AlcubierreMetric(v_s=v_s)

    # Evaluate geometry ONCE
    result = evaluate_curvature_grid(
        metric, grid_spec,
        compute_invariants=True,
    )

    # Flatten for sweep
    N = int(np.prod(grid_spec.shape))
    T_flat = result.stress_energy.reshape(N, 4, 4)
    g_flat = result.metric.reshape(N, 4, 4)
    energy_density = np.asarray(result.stress_energy[..., 0, 0])

    # Extract grid coordinates as NumPy
    X, Y, Z = grid_spec.meshgrid
    x_np = np.asarray(X)
    y_np = np.asarray(Y)
    z_np = np.asarray(Z)

    # Fixed boost directions: +x, +y, +z
    all_directions = [
        (float(np.pi / 2), 0.0),         # +x
        (float(np.pi / 2), float(np.pi / 2)),  # +y
        (0.0, 0.0),                       # +z
    ]
    directions = all_directions[:n_directions]

    rapidities = np.linspace(0.0, zeta_max, n_frames)

    # Progress bar
    iterator = enumerate(rapidities)
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(list(iterator), desc="Observer sweep", unit="frame")
    except ImportError:
        iterator = list(iterator)

    frames = []
    metric_name = metric.name()

    for i, zeta in iterator:
        # Construct observer params for this single rapidity
        obs_params = jnp.array([
            [float(zeta), theta, phi] for theta, phi in directions
        ])

        # Compute margins for this rapidity
        wec_margins = sweep_wec_margins(T_flat, g_flat, obs_params)  # (N, K)
        nec_params = obs_params[:, 1:]  # (K, 2)
        nec_margins = sweep_nec_margins(T_flat, g_flat, nec_params)  # (N, K)

        # Worst-case across directions at this rapidity
        worst_wec = np.asarray(jnp.min(wec_margins, axis=-1).reshape(grid_spec.shape))
        worst_nec = np.asarray(jnp.min(nec_margins, axis=-1).reshape(grid_spec.shape))

        scalar_fields = {
            "energy_density": energy_density,
            "wec_margin_sweep": worst_wec,
            "nec_margin_sweep": worst_nec,
        }

        colormaps = {
            "energy_density": "RdBu_r",
            "wec_margin_sweep": "RdBu_r",
            "nec_margin_sweep": "RdBu_r",
        }

        clim = {
            "energy_density": _symmetric_clim(energy_density),
            "wec_margin_sweep": _symmetric_clim(worst_wec),
            "nec_margin_sweep": _symmetric_clim(worst_nec),
        }

        frame = FrameData(
            x=x_np,
            y=y_np,
            z=z_np,
            scalar_fields=scalar_fields,
            metric_name=metric_name,
            v_s=v_s,
            grid_shape=grid_spec.shape,
            t=float(zeta),  # Carry rapidity in t field for annotation
            colormaps=colormaps,
            clim=clim,
        )
        frames.append(frame)

    return frames
