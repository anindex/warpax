
"""Batch render all Manim showcase scenes to MP4 and optimized GIF.

Renders all 8 WarpAx Manim scenes (4 x 3D, 4 x 2D heatmap) at the
specified quality, then optionally converts each MP4 to a 720p GIF
using the FFmpeg two-pass palette workflow.

Usage
-----
Render all scenes at 1080p with GIF conversion::

    python scripts/render_all_scenes.py

Render a single scene at low quality for testing::

    python scripts/render_all_scenes.py --scene ECHeatmapContour --quality l

Skip GIF conversion::

    python scripts/render_all_scenes.py --skip-gif

Custom output directory::

    python scripts/render_all_scenes.py --output-dir /tmp/manim_output

Examples
--------
Smoke test (fast, low quality)::

    python scripts/render_all_scenes.py --scene BubbleCollapse --quality l

Full render for publication::

    python scripts/render_all_scenes.py --quality h
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

# Project root (scripts/ is one level below)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIM_DIR = PROJECT_ROOT / "src" / "warpax" / "visualization" / "manim"

# All 8 scenes: (module_path_relative_to_project, class_name, description)
SCENES = [
    # 3D scenes
    ("src/warpax/visualization/manim/_bubble_collapse.py", "BubbleCollapse", "Bubble collapse animation"),
    ("src/warpax/visualization/manim/_velocity_ramp.py", "VelocityRamp", "Velocity ramp showcase"),
    ("src/warpax/visualization/manim/_observer_sweep.py", "ObserverSweep", "Observer sweep comparison"),
    # 2D heatmap scenes
    ("src/warpax/visualization/manim/_heatmap_contour.py", "ECHeatmapContour", "EC heatmap with contours"),
    ("src/warpax/visualization/manim/_split_screen.py", "SplitScreen", "Eulerian vs robust split-screen"),
    ("src/warpax/visualization/manim/_boost_arrows.py", "BoostArrows", "Boost direction arrow field"),
    ("src/warpax/visualization/manim/_expansion_shear.py", "ExpansionShear", "Expansion/shear field"),
]

QUALITY_DIRS = {
    "l": "480p15",
    "m": "720p30",
    "h": "1080p60",
    "k": "2160p60",
}

logger = logging.getLogger(__name__)


def find_mp4(
    scene_class: str,
    module_stem: str,
    quality: str,
    media_dir: Path,
) -> Path | None:
    """Locate the rendered MP4 file for a scene."""
    quality_dir = QUALITY_DIRS.get(quality, "1080p60")
    expected = media_dir / "videos" / module_stem / quality_dir / f"{scene_class}.mp4"
    if expected.exists():
        return expected

    # Fallback: search recursively
    candidates = list((media_dir / "videos").rglob(f"{scene_class}.mp4"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    return None


def render_scene(
    module_path: str,
    scene_class: str,
    quality: str,
    media_dir: Path,
) -> Path | None:
    """Render a single Manim scene and return the MP4 path on success."""
    cmd = [
        sys.executable, "-m", "manim", "render",
        f"-q{quality}",
        "--format", "mp4",
        "--media_dir", str(media_dir),
        str(PROJECT_ROOT / module_path),
        scene_class,
    ]

    logger.info("Rendering %s ...", scene_class)
    logger.debug("Command: %s", " ".join(cmd))

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        logger.error(
            "FAILED: %s (exit %d, %.1fs)\n%s",
            scene_class,
            result.returncode,
            elapsed,
            result.stderr[-1000:] if result.stderr else "(no stderr)",
        )
        return None

    mp4_path = find_mp4(
        scene_class,
        Path(module_path).stem,
        quality,
        media_dir,
    )
    if mp4_path is None:
        logger.error("FAILED: %s MP4 not found after render", scene_class)
        return None

    logger.info("OK: %s (%.1fs) -> %s", scene_class, elapsed, mp4_path)
    return mp4_path


def convert_to_gif(mp4_path: Path) -> Path | None:
    """Convert an MP4 to optimized GIF, returning the GIF path on success."""
    # Import from the project package
    from warpax.visualization.manim._gif_utils import mp4_to_gif

    try:
        gif_path = mp4_to_gif(mp4_path, width=1280, fps=20)
        logger.info("GIF: %s", gif_path)
        return gif_path
    except Exception:
        logger.exception("GIF conversion failed for %s", mp4_path)
        return None


def main() -> None:
    """Entry point for batch rendering."""
    parser = argparse.ArgumentParser(
        description="Batch render all WarpAx Manim showcase scenes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Scene names:\n"
            + "\n".join(f"  {name:<25s} {desc}" for _, name, desc in SCENES)
        ),
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        metavar="NAME",
        help="Render only the named scene (case-sensitive class name).",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="h",
        choices=["l", "m", "h", "k"],
        help="Render quality: l=480p, m=720p, h=1080p (default), k=4K.",
    )
    parser.add_argument(
        "--skip-gif",
        action="store_true",
        help="Skip GIF conversion after rendering.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Output media directory (default: media/ in project root).",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Draft mode: low quality (-ql) and skip GIF conversion.",
    )
    args = parser.parse_args()

    # Draft mode overrides
    if args.draft:
        args.quality = "l"
        args.skip_gif = True

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine output directory
    if args.output_dir is not None:
        media_dir = Path(args.output_dir).resolve()
    else:
        media_dir = PROJECT_ROOT / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    # Filter scenes if --scene is specified
    if args.scene is not None:
        matching = [(m, c, d) for m, c, d in SCENES if c == args.scene]
        if not matching:
            valid = ", ".join(c for _, c, _ in SCENES)
            parser.error(f"Unknown scene '{args.scene}'. Valid names: {valid}")
        scenes = matching
    else:
        scenes = SCENES

    # Render loop
    results: list[tuple[str, str, Path | None, Path | None]] = []
    total = len(scenes)

    print(f"\nRendering {total} scene(s) at quality={args.quality} -> {media_dir}\n")
    t_total_start = time.monotonic()

    for i, (module_path, scene_class, desc) in enumerate(scenes, 1):
        print(f"[{i}/{total}] {scene_class} ({desc})")

        mp4_path = render_scene(module_path, scene_class, args.quality, media_dir)

        gif_path = None
        if mp4_path is not None and not args.skip_gif:
            gif_path = convert_to_gif(mp4_path)

        results.append((scene_class, desc, mp4_path, gif_path))

    t_total = time.monotonic() - t_total_start

    # Summary
    succeeded = sum(1 for _, _, mp4, _ in results if mp4 is not None)
    gifs_ok = sum(1 for _, _, _, gif in results if gif is not None)

    print(f"\n{'=' * 60}")
    print(f"Render Summary: {succeeded}/{total} scenes succeeded ({t_total:.1f}s)")
    if not args.skip_gif:
        print(f"GIF conversion: {gifs_ok}/{succeeded} converted")
    print(f"{'=' * 60}")

    for scene_class, desc, mp4, gif in results:
        status = "OK" if mp4 is not None else "FAILED"
        print(f"  [{status:>6s}] {scene_class}")
        if mp4 is not None:
            print(f"          MP4: {mp4}")
        if gif is not None:
            print(f"          GIF: {gif}")

    print()

    # Exit with error code if any scene failed
    if succeeded < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
