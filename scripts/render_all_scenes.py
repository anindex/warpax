"""Batch render all Manim showcase scenes to MP4 and optimized GIF.

Renders all 8 WarpAx Manim scenes (3 x 3D, 5 x 2D heatmap) at the
specified quality, then optionally converts each MP4 to a 720p GIF
using the FFmpeg two-pass palette workflow.

Usage
-----
Render all scenes at 1080p with GIF conversion::

    python scripts/render_all_scenes.py

Render a single scene at low quality for testing::

    python scripts/render_all_scenes.py --scene NECMargin2D --quality l

Skip GIF conversion::

    python scripts/render_all_scenes.py --skip-gif

Custom output directory::

    python scripts/render_all_scenes.py --output-dir /tmp/manim_output

Examples
--------
Smoke test (fast, low quality)::

    python scripts/render_all_scenes.py --scene WallAndVelocitySweep --quality l

Full render for publication::

    python scripts/render_all_scenes.py --quality h
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Project root (scripts/ is one level below)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIM_DIR = PROJECT_ROOT / "src" / "warpax" / "visualization" / "manim"

# Manim's 3D *Cairo* camera projects every sub-mobject recursively and corrupts
# the C heap on the heavy dual-layer surfaces (segfault/SIGILL mid-render,
# independent of Python version or stack size). The 3D scenes therefore render
# with the GPU OpenGL renderer instead (see ``_OPENGL_SCENES`` /
# ``render_scene``); the 2D scenes keep the stable Cairo renderer.


def _render_env() -> dict:
    """Env for the manim subprocess: constrain JAX/BLAS threads and enable gRPC
    fork handlers so manim's ffmpeg fork doesn't deadlock against JAX's thread
    pool under Python 3.14. Set before the child imports JAX; only fills values
    the caller has not already set.
    """
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("GRPC_ENABLE_FORK_SUPPORT", "1")
    if "xla_cpu_multi_thread_eigen" not in env.get("XLA_FLAGS", ""):
        env["XLA_FLAGS"] = (
            env.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
        ).strip()
    return env


# All 9 scenes: (module_path_relative_to_project, class_name, description)
SCENES = [
    # 3D scenes
    (
        "src/warpax/visualization/manim/_wall_velocity_sweep.py",
        "WallAndVelocitySweep",
        "Alcubierre bubble: rho_Eul + NEC margin (wall-steepness then velocity sweep)",
    ),
    (
        "src/warpax/visualization/manim/_velocity_sweep.py",
        "VelocitySweep",
        "Quasi-static velocity sweep (v_s 0.1->0.99)",
    ),
    (
        "src/warpax/visualization/manim/_boost_rapidity_sweep.py",
        "BoostRapiditySweep",
        "Boosted-observer energy density vs rapidity",
    ),
    # 2D heatmap scenes
    (
        "src/warpax/visualization/manim/_nec_margin.py",
        "NECMargin2D",
        "Observer-robust NEC margin (k.n_Eul = -1)",
    ),
    (
        "src/warpax/visualization/manim/_split_screen.py",
        "EulerianVsWorstCaseNEC",
        "Eulerian 6-null vs worst-case NEC split-screen",
    ),
    (
        "src/warpax/visualization/manim/_boost_arrows.py",
        "WorstCaseNullDirections",
        "Worst-case null direction over the NEC margin",
    ),
    (
        "src/warpax/visualization/manim/_boost_arrows.py",
        "WorstCaseBoostDirections",
        "Worst-case boost over the bounded WEC margin + threshold rapidity",
    ),
    (
        "src/warpax/visualization/manim/_eulerian_kinematics.py",
        "EulerianKinematics2D",
        "Expansion theta = -K and shear sigma^2 of the Eulerian congruence",
    ),
    (
        "src/warpax/visualization/manim/_kretschmann.py",
        "KretschmannInvariant2D",
        "Observer-independent Kretschmann curvature invariant",
    ),
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


# The dual-layer ThreeDScenes overwhelm manim's *Cairo* 3D renderer (its
# recursive camera projection corrupts/overflows the C heap on the heavy
# 32x32 + 48x48 surfaces, segfaulting mid-render regardless of Python version or
# stack size). The GPU OpenGL renderer projects on-device and is stable, so these
# scenes render with ``--renderer=opengl`` under EGL (headless). 2D scenes keep
# the Cairo renderer, whose output is already validated.
_OPENGL_SCENES = frozenset(
    {"WallAndVelocitySweep", "VelocitySweep", "BoostRapiditySweep"}
)


def render_scene(
    module_path: str,
    scene_class: str,
    quality: str,
    media_dir: Path,
) -> Path | None:
    """Render a single Manim scene and return the MP4 path on success."""
    use_opengl = scene_class in _OPENGL_SCENES
    cmd = [
        sys.executable,
        "-m",
        "manim",
        "render",
        f"-q{quality}",
        "--format",
        "mp4",
        "--media_dir",
        str(media_dir),
    ]
    if use_opengl:
        # OpenGL needs an explicit movie write (it otherwise opens a preview).
        cmd += ["--renderer", "opengl", "--write_to_movie"]
    cmd += [str(PROJECT_ROOT / module_path), scene_class]

    logger.info(
        "Rendering %s (%s renderer) ...",
        scene_class,
        "opengl" if use_opengl else "cairo",
    )
    logger.debug("Command: %s", " ".join(cmd))

    env = _render_env()
    if use_opengl:
        # Headless GPU context for the OpenGL renderer (the box has EGL). Note:
        # do NOT raise the stack (``ulimit -s unlimited``) for OpenGL -- an
        # unlimited main-thread stack destabilises the EGL/GL driver threads and
        # segfaults late in the render. The 3D scenes use OpenGL precisely to
        # avoid the Cairo recursion, so no stack raise is needed anywhere.
        env.setdefault("PYOPENGL_PLATFORM", "egl")

    # The OpenGL/EGL renderer occasionally segfaults transiently (driver-side)
    # and succeeds on a clean retry, so give each render up to three attempts.
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        t0 = time.monotonic()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        elapsed = time.monotonic() - t0
        if result.returncode == 0:
            break
        if attempt < max_attempts:
            logger.warning(
                "%s crashed (exit %d, %.1fs); retrying (%d/%d)",
                scene_class,
                result.returncode,
                elapsed,
                attempt + 1,
                max_attempts,
            )

    if result.returncode != 0:
        logger.error(
            "FAILED: %s (exit %d, %.1fs, %d attempts)\n%s",
            scene_class,
            result.returncode,
            elapsed,
            max_attempts,
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
        epilog=("Scene names:\n" + "\n".join(f"  {name:<25s} {desc}" for _, name, desc in SCENES)),
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
