#!/usr/bin/env bash
# render_manim_scenes.sh -- Render all Manim showcase scenes (MP4 + GIF).
#
# Thin wrapper around scripts/render_all_scenes.py, which owns the canonical
# scene list (the two used to drift: this script rendered 3 scenes while the
# Python renderer rendered 7+). Delegating keeps a single source of truth.
#
# Usage: bash scripts/render_manim_scenes.sh [QUALITY] [extra render_all_scenes.py args]
# QUALITY: low (480p test), medium (720p), high (1080p, default), 4k (2160p)

set -euo pipefail

# Manim's 3D Cairo render overflows the default 8 MB main-thread stack under
# Python 3.14 and segfaults. Raise the stack before rendering (best-effort;
# render_all_scenes.py also raises it per-subprocess and the scene modules cap
# the recursion limit, so this is belt-and-suspenders).
ulimit -s unlimited 2>/dev/null || ulimit -s 65536 2>/dev/null || true

# Fork-safety for the 3D scenes: manim spawns ffmpeg while JAX's thread pool is
# live, which deadlocks on fork under Python 3.14. Constrain JAX/BLAS to a small
# thread pool and enable gRPC fork handlers BEFORE python starts (must precede
# any JAX import). Exported so the render_all_scenes.py subprocess inherits them.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export GRPC_ENABLE_FORK_SUPPORT="${GRPC_ENABLE_FORK_SUPPORT:-1}"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false"

QUALITY="${1:-high}"
case "$QUALITY" in
    low) Q="l" ;;
    medium) Q="m" ;;
    high) Q="h" ;;
    4k) Q="k" ;;
    *) echo "Unknown quality: $QUALITY. Use: low, medium, high, 4k"; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Rendering all Manim scenes (quality: $QUALITY) ==="
exec python "${SCRIPT_DIR}/render_all_scenes.py" --quality "$Q" "${@:2}"
