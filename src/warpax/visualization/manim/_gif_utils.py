"""FFmpeg palette-based GIF conversion utilities.

Provides a two-pass FFmpeg workflow for converting Manim MP4 output to
high-quality optimized GIFs:

1. **palettegen** generates an optimal 256-color palette from the video
2. **paletteuse** encodes the GIF using that palette with Bayer dithering

This approach produces significantly better GIF quality than naive encoding
(especially for smooth gradients in EC heatmaps) while keeping file sizes
reasonable.

"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def mp4_to_gif(
    input_path: str | Path,
    output_path: str | Path | None = None,
    fps: int = 20,
    width: int = 1280,
    dither: str = "bayer",
    bayer_scale: int = 5,
) -> Path:
    """Convert an MP4 file to an optimized GIF using FFmpeg palette workflow.

    Uses a two-pass approach:

    - **Pass 1:** Generate an optimal 256-color palette with
      ``palettegen=stats_mode=diff``.
    - **Pass 2:** Encode the GIF using the palette with the specified
      dither algorithm.

    Parameters
    ----------
    input_path : str or Path
        Path to the input MP4 file.
    output_path : str or Path or None
        Path for the output GIF.  If ``None``, replaces the ``.mp4``
        extension with ``.gif``.
    fps : int
        Target frame rate for the GIF (default 20).
    width : int
        Target width in pixels (height auto-scaled, default 1280).
    dither : str
        FFmpeg dither algorithm (default ``"bayer"``).
    bayer_scale : int
        Bayer dither scale, 1--5 (default 5).  Higher values produce
        smoother gradients.

    Returns
    -------
    Path
        Path to the generated GIF file.

    Raises
    ------
    RuntimeError
        If FFmpeg is not found on PATH or if either pass fails.
    FileNotFoundError
        If *input_path* does not exist.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".gif")
    else:
        output_path = Path(output_path)

    # Check ffmpeg availability
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg not found on PATH. Install FFmpeg to use GIF conversion. "
            "On Ubuntu: sudo apt install ffmpeg"
        )

    tmpdir = tempfile.mkdtemp(prefix="warpax_gif_")
    palette_path = Path(tmpdir) / "palette.png"

    try:
        # Filter chain shared between passes
        scale_filter = f"fps={fps},scale={width}:-1:flags=lanczos"

        # Pass 1: Generate palette
        cmd_palette = [
            "ffmpeg",
            "-i", str(input_path),
            "-vf", f"{scale_filter},palettegen=stats_mode=diff",
            "-y", str(palette_path),
        ]
        logger.debug("FFmpeg palette pass: %s", " ".join(cmd_palette))

        result_palette = subprocess.run(
            cmd_palette,
            capture_output=True,
            text=True,
        )
        if result_palette.returncode != 0:
            logger.error(
                "FFmpeg palette generation failed:\n%s", result_palette.stderr
            )
            raise RuntimeError(
                f"FFmpeg palette generation failed (exit {result_palette.returncode}). "
                f"stderr: {result_palette.stderr[-500:]}"
            )

        # Pass 2: Encode GIF using palette
        lavfi = (
            f"{scale_filter}[x];[x][1:v]paletteuse="
            f"dither={dither}:bayer_scale={bayer_scale}"
        )
        cmd_gif = [
            "ffmpeg",
            "-i", str(input_path),
            "-i", str(palette_path),
            "-lavfi", lavfi,
            "-y", str(output_path),
        ]
        logger.debug("FFmpeg GIF pass: %s", " ".join(cmd_gif))

        result_gif = subprocess.run(
            cmd_gif,
            capture_output=True,
            text=True,
        )
        if result_gif.returncode != 0:
            logger.error(
                "FFmpeg GIF encoding failed:\n%s", result_gif.stderr
            )
            raise RuntimeError(
                f"FFmpeg GIF encoding failed (exit {result_gif.returncode}). "
                f"stderr: {result_gif.stderr[-500:]}"
            )

        logger.info("GIF created: %s", output_path)
        return output_path

    finally:
        # Clean up temporary palette file
        if palette_path.exists():
            palette_path.unlink()
        try:
            Path(tmpdir).rmdir()
        except OSError:
            pass  # directory not empty or already removed


def render_and_convert(
    scene_module_path: str | Path,
    scene_class_name: str,
    quality: str = "h",
    output_dir: str | Path | None = None,
    gif_fps: int = 20,
    gif_width: int = 1280,
) -> tuple[Path, Path]:
    """Render a Manim scene to MP4 and convert to optimized GIF.

    Convenience function that runs ``manim render`` on the specified
    scene module and class, then calls :func:`mp4_to_gif` on the
    output.

    Parameters
    ----------
    scene_module_path : str or Path
        Path to the Python file containing the scene class.
    scene_class_name : str
        Name of the Manim ``Scene`` subclass to render.
    quality : str
        Render quality: ``"l"`` (480p), ``"m"`` (720p), ``"h"``
        (1080p), ``"k"`` (4K).  Default ``"h"``.
    output_dir : str or Path or None
        Directory for output files.  If ``None``, uses Manim's default
        ``media/`` directory.
    gif_fps : int
        GIF frame rate (default 20).
    gif_width : int
        GIF width in pixels (default 1280).

    Returns
    -------
    tuple[Path, Path]
        ``(mp4_path, gif_path)`` paths to the rendered MP4 and the
        converted GIF.

    Raises
    ------
    RuntimeError
        If ``manim`` is not found on PATH, rendering fails, or the
        output MP4 cannot be located.
    """
    scene_module_path = Path(scene_module_path)

    if shutil.which("manim") is None:
        raise RuntimeError(
            "Manim CLI not found on PATH. Install with: pip install manim"
        )

    # Build render command
    cmd = [
        "manim", "render",
        f"-q{quality}",
        "--format", "mp4",
        str(scene_module_path),
        scene_class_name,
    ]

    if output_dir is not None:
        output_dir = Path(output_dir)
        cmd.extend(["--media_dir", str(output_dir)])

    logger.info("Rendering scene: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Manim render failed:\n%s", result.stderr)
        raise RuntimeError(
            f"Manim render failed (exit {result.returncode}). "
            f"stderr: {result.stderr[-500:]}"
        )

    # Locate the output MP4
    # Manim outputs to media/videos/{module_stem}/{quality_dir}/{SceneName}.mp4
    quality_dirs = {
        "l": "480p15",
        "m": "720p30",
        "h": "1080p60",
        "k": "2160p60",
    }
    quality_dir = quality_dirs.get(quality, "1080p60")

    if output_dir is not None:
        search_root = output_dir / "videos"
    else:
        search_root = Path("media") / "videos"

    module_stem = scene_module_path.stem
    expected_mp4 = (
        search_root / module_stem / quality_dir / f"{scene_class_name}.mp4"
    )

    if expected_mp4.exists():
        mp4_path = expected_mp4
    else:
        # Fallback: search for the MP4 in the search root
        mp4_candidates = list(search_root.rglob(f"{scene_class_name}.mp4"))
        if not mp4_candidates:
            raise RuntimeError(
                f"Could not find rendered MP4 for {scene_class_name}. "
                f"Searched in: {search_root}"
            )
        # Use the most recently modified
        mp4_path = max(mp4_candidates, key=lambda p: p.stat().st_mtime)
        logger.info("Found MP4 at non-standard path: %s", mp4_path)

    # Convert to GIF
    gif_path = mp4_to_gif(mp4_path, fps=gif_fps, width=gif_width)

    return mp4_path, gif_path
