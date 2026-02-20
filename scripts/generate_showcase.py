
"""Generate showcase animations for warp drive physics.

Produces four signature animations demonstrating the paper's key physics
results, exported as GIF (social media), MP4 (talks), and PNG frames
(paper supplement).

Usage
-----
All scenes (production quality):
    python scripts/generate_showcase.py

Quick test (fewer frames, lower resolution):
    python scripts/generate_showcase.py --quick

Single scene:
    python scripts/generate_showcase.py --scene collapse
    python scripts/generate_showcase.py --scene ramp
    python scripts/generate_showcase.py --scene observer

Specific formats only:
    python scripts/generate_showcase.py --formats gif mp4

Combine flags:
    python scripts/generate_showcase.py --quick --scene collapse --formats gif
"""
from __future__ import annotations

import argparse
import gc
import os
import time

# Non-interactive backend (before any other matplotlib import)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from warpax.geometry import GridSpec
from warpax.visualization.common import (
    add_text_overlay,
    add_watermark,
    scene_bubble_collapse,
    scene_observer_sweep,
    scene_velocity_ramp,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Production settings
PROD_FRAMES = {"collapse": 90, "ramp": 90, "observer": 60}
PROD_GRID = 40

# Quick settings
QUICK_FRAMES = {"collapse": 15, "ramp": 15, "observer": 10}
QUICK_GRID = 20

# Resolution settings
GIF_RES = (720, 720)       # Square for social media
MP4_RES = (1920, 1080)     # 16:9 for talks
PNG_RES = (1920, 1080)     # 16:9 for supplement

# Grid bounds (standard warp bubble domain)
GRID_BOUNDS = [(-3, 3), (-3, 3), (-3, 3)]

# Static camera position for physics scenes (elevation 30 degrees)
STATIC_CAMERA = [(6, 6, 4), (0, 0, 0), (0, 0, 1)]


# ---------------------------------------------------------------------------
# Embedding setup_frame callback factory
# ---------------------------------------------------------------------------

# NOTE: The PyVista-based 3D embedding rendering was removed.
# Use Manim scenes (scripts/render_all_scenes.py) for animations.
# The scene builders (scene_bubble_collapse, etc.) from the common
# layer still produce FrameData lists usable by any backend.


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _ensure_rgba(img):
    """Convert RGB image to RGBA if needed."""
    if img.ndim == 3 and img.shape[2] == 3:
        alpha = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
        return np.concatenate([img, alpha], axis=2)
    return img


def _apply_overlays(images, title, param_fn, watermark=False):
    """Apply text overlays and optional watermark to screenshot list.

    Parameters
    ----------
    images : list[np.ndarray]
        RGB or RGBA screenshot arrays.
    title : str
        Scene title text.
    param_fn : callable
        ``(idx) -> str`` returning the parameter string for frame *idx*.
    watermark : bool
        Whether to add WarpAx watermark.

    Returns
    -------
    list[np.ndarray]
        Processed RGBA images.
    """
    result = []
    for i, img in enumerate(images):
        img = _ensure_rgba(img)
        text = f"{title}\n{param_fn(i)}"
        out = add_text_overlay(img, text, position="upper_left", font_size=20)
        if watermark:
            out = add_watermark(out, text="WarpAx", alpha=100)
        result.append(out)
    return result


def _write_gif(images, path, fps=20):
    """Write RGBA images to GIF."""
    import imageio.v3 as iio
    duration_ms = 1000.0 / fps
    iio.imwrite(
        path,
        images,
        extension=".gif",
        duration=duration_ms,
        loop=0,
        palettesize=256,
    )


def _write_mp4(images, path, fps=30, quality=5):
    """Write RGB images to MP4 (H.264)."""
    import imageio.v2 as iio_v2
    writer = iio_v2.get_writer(
        str(path),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        quality=quality,
    )
    try:
        for img in images:
            writer.append_data(img[:, :, :3])
    finally:
        writer.close()


def _write_png(images, directory, basename):
    """Write RGBA images as numbered PNG frames."""
    os.makedirs(directory, exist_ok=True)
    import imageio.v3 as iio
    for idx, img in enumerate(images):
        frame_path = os.path.join(directory, f"{basename}_{idx:04d}.png")
        iio.imwrite(frame_path, img)


# ---------------------------------------------------------------------------
# Scene renderers (matplotlib-based 2D slices)
# ---------------------------------------------------------------------------


def _render_2d_frame(frame_data, field="energy_density", title=""):
    """Render a single FrameData as a 2D z=0 slice using matplotlib.

    Returns an RGBA numpy array.
    """
    from warpax.visualization.common._color import resolve_clim, resolve_cmap

    color_field = field if field in frame_data.scalar_fields else "energy_density"
    cmap = resolve_cmap(frame_data, color_field)
    clim = resolve_clim(frame_data, color_field)

    # Extract z=0 mid-slice
    arr = frame_data.scalar_fields[color_field]
    mid_z = arr.shape[2] // 2
    slice_2d = arr[:, :, mid_z].T

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    im = ax.imshow(
        slice_2d,
        origin="lower",
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        extent=[
            float(frame_data.x.min()),
            float(frame_data.x.max()),
            float(frame_data.y.min()),
            float(frame_data.y.max()),
        ],
    )
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.tick_params(colors="white")
    if title:
        ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.yaxis.label.set_color("white")
    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_color("white")
    fig.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame_img = buf.reshape(h, w, 4).copy()
    plt.close(fig)
    return frame_img


def render_scene_collapse(grid_spec, n_frames, output_dir, formats, quick):
    """Render bubble collapse scene: Alcubierre v_s ramps down."""
    print("  Building frames...")
    frames = scene_bubble_collapse(grid_spec, n_frames=n_frames)

    images = []
    for i, fd in enumerate(frames):
        img = _render_2d_frame(fd, field="energy_density",
                               title=f"Bubble Collapse  v_s = {fd.v_s:.3f}")
        images.append(img)

    _export_images(
        images, scene_name="collapse", title="Bubble Collapse",
        param_fn=lambda i: f"v_s = {frames[min(i, len(frames)-1)].v_s:.3f}",
        output_dir=output_dir, formats=formats,
    )


def render_scene_ramp(grid_spec, n_frames, output_dir, formats, quick):
    """Render velocity ramp-up scene: EC violations intensify."""
    print("  Building frames...")
    frames = scene_velocity_ramp(grid_spec, n_frames=n_frames)

    images = []
    for i, fd in enumerate(frames):
        field = "wec_margin_sweep" if "wec_margin_sweep" in fd.scalar_fields else "energy_density"
        img = _render_2d_frame(fd, field=field,
                               title=f"Velocity Ramp  v_s = {fd.v_s:.3f}")
        images.append(img)

    _export_images(
        images, scene_name="ramp", title="Velocity Ramp",
        param_fn=lambda i: f"v_s = {frames[min(i, len(frames)-1)].v_s:.3f}",
        output_dir=output_dir, formats=formats,
    )


def render_scene_observer(grid_spec, n_frames, output_dir, formats, quick):
    """Render observer sweep scene: per-rapidity EC margins."""
    print("  Building frames...")
    frames = scene_observer_sweep(grid_spec, n_frames=n_frames)

    images = []
    for i, fd in enumerate(frames):
        field = "wec_margin_sweep" if "wec_margin_sweep" in fd.scalar_fields else "energy_density"
        img = _render_2d_frame(fd, field=field,
                               title=f"Observer Sweep  \u03b6 = {fd.t:.2f}")
        images.append(img)

    _export_images(
        images, scene_name="observer", title="Observer Sweep",
        param_fn=lambda i: f"\u03b6 = {frames[min(i, len(frames)-1)].t:.2f}",
        output_dir=output_dir, formats=formats,
    )


# ---------------------------------------------------------------------------
# Export dispatcher
# ---------------------------------------------------------------------------


def _export_images(
    images,
    scene_name,
    title,
    param_fn,
    output_dir,
    formats,
    watermark=False,
):
    """Export pre-rendered RGBA images in requested formats.

    GIF: square, dark theme, watermark
    MP4: widescreen, no watermark
    PNG: numbered frames, no watermark
    """
    os.makedirs(output_dir, exist_ok=True)

    if "gif" in formats:
        gif_imgs = _apply_overlays(images, title, param_fn, watermark=watermark)
        gif_path = os.path.join(output_dir, f"{scene_name}.gif")
        _write_gif(gif_imgs, gif_path)
        print(f"    Wrote: {gif_path}")

    if "mp4" in formats:
        mp4_imgs = _apply_overlays(images, title, param_fn, watermark=False)
        mp4_path = os.path.join(output_dir, f"{scene_name}.mp4")
        _write_mp4(mp4_imgs, mp4_path)
        print(f"    Wrote: {mp4_path}")

    if "png" in formats:
        png_imgs = _apply_overlays(images, title, param_fn, watermark=False)
        png_dir = os.path.join(output_dir, scene_name)
        _write_png(png_imgs, png_dir, scene_name)
        print(f"    Wrote: {png_dir}/ ({len(png_imgs)} frames)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate showcase animations for warp drive physics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer frames and lower grid resolution for fast iteration.",
    )
    parser.add_argument(
        "--scene",
        choices=["collapse", "ramp", "observer", "all"],
        default="all",
        help="Which scene to generate (default: all).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["gif", "mp4", "png"],
        default=["gif", "mp4", "png"],
        help="Output formats (default: gif mp4 png).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="animations",
        help="Output directory (default: animations).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Grid resolution per axis (default: 40, quick: 20).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    quick = args.quick
    grid_size = args.grid_size or (QUICK_GRID if quick else PROD_GRID)
    grid_spec = GridSpec(
        bounds=GRID_BOUNDS,
        shape=(grid_size, grid_size, grid_size),
    )
    frames_config = QUICK_FRAMES if quick else PROD_FRAMES

    scenes = (
        ["collapse", "ramp", "observer"]
        if args.scene == "all"
        else [args.scene]
    )

    print(f"Grid: {grid_size}^3 | Quick: {quick} | Formats: {args.formats}")
    print(f"Scenes: {', '.join(scenes)}")
    print(f"Output: {args.output_dir}/")

    render_fns = {
        "collapse": render_scene_collapse,
        "ramp": render_scene_ramp,
        "observer": render_scene_observer,
    }

    t_total = time.time()

    for scene_name in scenes:
        print(f"\n{'=' * 60}")
        print(f"  Generating: {scene_name}")
        print(f"{'=' * 60}\n")
        t0 = time.time()
        n_frames = frames_config[scene_name]
        render_fns[scene_name](
            grid_spec, n_frames, args.output_dir, args.formats, quick,
        )
        elapsed = time.time() - t0
        print(f"\n  {scene_name} complete in {elapsed:.1f}s")
        gc.collect()

    total = time.time() - t_total
    print(f"\nAll scenes complete in {total:.1f}s. Output in: {args.output_dir}/")


if __name__ == "__main__":
    main()
