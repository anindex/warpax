
# reproduce_all.sh Regenerate ALL results and figures
#
# Single-command reproducibility script for warpax.
# Deletes all cached results and figures, re-runs every analysis script
# in dependency order, and regenerates all figure PDFs.
#
# Usage:
#   ./reproduce_all.sh              Full regeneration (delete cache + recompute)
#   ./reproduce_all.sh --keep-cache Skip cache deletion (re-run only missing)
#   ./reproduce_all.sh --phase N    Run only phase N (1, 2, or 3)
#
# Phases:
#   1 - Core computation (analysis, convergence, kinematic scalars, geodesics)
#   2 - Ablation studies (C1/C2, N-starts, zeta, Rodal DEC, WarpShell, etc.)
#   3 - Figure generation
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
FIGURES_DIR="${SCRIPT_DIR}/figures"

# Parse flags
KEEP_CACHE=false
PHASE_ONLY=""
for arg in "$@"; do
    case "$arg" in
        --keep-cache) KEEP_CACHE=true ;;
        --phase)      shift; PHASE_ONLY="$1" ;;
        1|2|3)        PHASE_ONLY="$arg" ;;
        -h|--help)
            echo "Usage: $0 [--keep-cache] [--phase N]"
            echo "  --keep-cache  Skip cache deletion (only recompute missing results)"
            echo "  --phase N     Run only phase N (1=core, 2=ablations, 3=figures)"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

PYTHON="${PYTHON:-python}"

# Track elapsed time
SECONDS=0

# ------------------------------------------------------------------
# Step 0: Clear cached results (unless --keep-cache)
# ------------------------------------------------------------------
if [ "$KEEP_CACHE" = false ] && [ -z "$PHASE_ONLY" -o "$PHASE_ONLY" = "1" ]; then
    echo "============================================================"
    echo "  Step 0: Clearing cached results and figures"
    echo "============================================================"
    find "${RESULTS_DIR}" -name '*.npz' -delete 2>/dev/null || true
    find "${RESULTS_DIR}" -name '*.json' ! -name 'paper_constants.json' -delete 2>/dev/null || true
    find "${RESULTS_DIR}" -name '*.tex' -delete 2>/dev/null || true
    find "${FIGURES_DIR}" -name '*.pdf' -delete 2>/dev/null || true
    echo "  Cleared results/ (except paper_constants.json) and figures/*.pdf"
    echo ""
fi

# ------------------------------------------------------------------
# Phase 1: Core computation (independent scripts)
# ------------------------------------------------------------------
run_phase_1() {
    echo "============================================================"
    echo "  Phase 1: Core computation"
    echo "============================================================"

    echo ""
    echo "[1/4] run_analysis.py Full metric analysis sweep"
    $PYTHON "${SCRIPT_DIR}/scripts/run_analysis.py"

    echo ""
    echo "[2/4] run_convergence.py Richardson extrapolation convergence"
    $PYTHON "${SCRIPT_DIR}/scripts/run_convergence.py"

    echo ""
    echo "[3/4] run_kinematic_scalars.py Kinematic scalar fields"
    $PYTHON "${SCRIPT_DIR}/scripts/run_kinematic_scalars.py"

    echo ""
    echo "[4/4] run_geodesics.py Geodesic integration & tidal forces"
    $PYTHON "${SCRIPT_DIR}/scripts/run_geodesics.py"

    echo ""
    echo "  Phase 1 complete."
    echo ""
}

# ------------------------------------------------------------------
# Phase 2: Ablation and supplementary studies
# ------------------------------------------------------------------
run_phase_2() {
    echo "============================================================"
    echo "  Phase 2: Ablation & supplementary studies"
    echo "============================================================"

    echo ""
    echo "[1/8] run_c1_vs_c2_comparison.py C1 vs C2 WarpShell comparison"
    $PYTHON "${SCRIPT_DIR}/scripts/run_c1_vs_c2_comparison.py"

    echo ""
    echo "[2/8] run_nstarts_ablation.py N-starts ablation"
    $PYTHON "${SCRIPT_DIR}/scripts/run_nstarts_ablation.py"

    echo ""
    echo "[3/8] run_zeta_sensitivity.py Zeta sensitivity"
    $PYTHON "${SCRIPT_DIR}/scripts/run_zeta_sensitivity.py"

    echo ""
    echo "[4/8] rodal_dec_ablation.py Rodal DEC ablation"
    $PYTHON "${SCRIPT_DIR}/scripts/rodal_dec_ablation.py"

    echo ""
    echo "[5/8] run_warpshell_convergence.py WarpShell convergence"
    $PYTHON "${SCRIPT_DIR}/scripts/run_warpshell_convergence.py"

    echo ""
    echo "[6/8] run_wall_resolution.py Wall resolution analysis"
    $PYTHON "${SCRIPT_DIR}/scripts/run_wall_resolution.py"

    echo ""
    echo "[7/8] run_sampling_comparison.py Fibonacci vs BFGS comparison"
    $PYTHON "${SCRIPT_DIR}/scripts/run_sampling_comparison.py"

    echo ""
    echo "[8/8] run_smoothwidth_ablation.py Smooth width ablation"
    $PYTHON "${SCRIPT_DIR}/scripts/run_smoothwidth_ablation.py"

    echo ""
    echo "[+] run_worst_observer_alignment.py Worst observer alignment"
    $PYTHON "${SCRIPT_DIR}/scripts/run_worst_observer_alignment.py"

    echo ""
    echo "[+] run_missed_detection_comparison.py Missed detection comparison"
    $PYTHON "${SCRIPT_DIR}/scripts/run_missed_detection_comparison.py"

    echo ""
    echo "  Phase 2 complete."
    echo ""
}

# ------------------------------------------------------------------
# Phase 3: Figure generation
# ------------------------------------------------------------------
run_phase_3() {
    echo "============================================================"
    echo "  Phase 3: Figure generation"
    echo "============================================================"

    echo ""
    echo "[1/1] reproduce_figures.py Generate all figure PDFs"
    $PYTHON "${SCRIPT_DIR}/scripts/reproduce_figures.py" \
        --figures-dir "${FIGURES_DIR}/" \
        --results-dir "${RESULTS_DIR}/"

    echo ""
    echo "  Phase 3 complete."
    echo ""
}

# ------------------------------------------------------------------
# Execute requested phases
# ------------------------------------------------------------------
case "${PHASE_ONLY}" in
    1) run_phase_1 ;;
    2) run_phase_2 ;;
    3) run_phase_3 ;;
    "")
        run_phase_1
        run_phase_2
        run_phase_3
        ;;
esac

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
ELAPSED=$SECONDS
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

echo "============================================================"
echo "  Reproduction complete!"
echo "  Total time: ${MINS}m ${SECS}s"
echo ""
echo "  Results:  $(find "${RESULTS_DIR}" -name '*.json' -o -name '*.npz' | wc -l) files"
echo "  Figures:  $(find "${FIGURES_DIR}" -name '*.pdf' 2>/dev/null | wc -l) PDFs"
echo "============================================================"
