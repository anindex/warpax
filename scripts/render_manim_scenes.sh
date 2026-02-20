
# render_manim_scenes.sh Render all Manim showcase scenes
# Usage: bash scripts/render_manim_scenes.sh [QUALITY]
#   QUALITY: low (480p test), medium (720p), high (1080p), 4k (3840x2160)
#   Default: high (1080p)
#
# Each scene is rendered as MP4 at the requested quality AND as GIF at
# 720p/20fps for lightweight embedding.

set -euo pipefail

QUALITY="${1:-high}"
# Parse quality argument
case "$QUALITY" in
    low)    Q="-ql" ;;
    medium) Q="-qm" ;;
    high)   Q="-qh" ;;
    4k)     Q="-qk" ;;
    *)      echo "Unknown quality: $QUALITY. Use: low, medium, high, 4k"; exit 1 ;;
esac

SCENES=(
    "src/warpax/visualization/manim/_bubble_collapse.py BubbleCollapse"
    "src/warpax/visualization/manim/_velocity_ramp.py VelocityRamp"
    "src/warpax/visualization/manim/_observer_sweep.py ObserverSweep"
)

echo "=== Rendering all Manim scenes ==="
echo "Quality: $QUALITY ($Q)"
echo ""

for scene in "${SCENES[@]}"; do
    FILE=$(echo "$scene" | cut -d' ' -f1)
    CLASS=$(echo "$scene" | cut -d' ' -f2)

    # Skip scenes whose source files do not exist yet
    if [ ! -f "$FILE" ]; then
        echo "--- Skipping $CLASS (source not found: $FILE) ---"
        echo ""
        continue
    fi

    echo "--- Rendering $CLASS ---"

    # MP4 at requested quality / 30fps
    echo "  MP4..."
    manim render $Q --format mp4 --fps 30 "$FILE" "$CLASS"

    # GIF at 720p / 20fps (always medium quality for lightweight embedding)
    echo "  GIF (720p)..."
    manim render -qm --format gif --fps 20 "$FILE" "$CLASS"

    echo "  Done: $CLASS"
    echo ""
done

echo "=== All scenes rendered ==="
echo "Output directory: media/videos/"
