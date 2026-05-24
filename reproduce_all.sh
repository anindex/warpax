#!/usr/bin/env bash
# reproduce_all.sh - Regenerate all warpax results and figures
#
# Deletes cached results and figures, re-runs every analysis script in
# dependency order, and regenerates all figure PDFs.
#
# Usage:
#   ./reproduce_all.sh                    Full regeneration
#   ./reproduce_all.sh --keep-cache       Re-run only missing results
#   ./reproduce_all.sh --stage NAME       Run only one stage
#
# Stages: core (analysis, convergence, scalars, geodesics),
#         ablation (ablation + supplementary studies),
#         figures (figure generation).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
FIGURES_DIR="${SCRIPT_DIR}/figures"

KEEP_CACHE=false
STAGE_ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --keep-cache) KEEP_CACHE=true ;;
        --stage)
            shift
            if [ $# -eq 0 ]; then
                echo "Error: --stage requires an argument (core, ablation, figures)" >&2
                exit 1
            fi
            STAGE_ONLY="$1"
            ;;
        core|ablation|figures) STAGE_ONLY="$1" ;;
        -h|--help)
            echo "Usage: $0 [--keep-cache] [--stage core|ablation|figures]"
            echo "  --keep-cache   Skip cache deletion (only recompute missing results)"
            echo "  --stage NAME   Run only one stage (core, ablation, or figures)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

PYTHON="${PYTHON:-python}"

export PYTHONPATH="${SCRIPT_DIR:-$PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Pin JAX backend to CPU by default for deterministic reproduction.
# Blackwell sm_120 with jax[cuda12]==0.10.0 crashes in two paths
# (cuBLAS LT autotuner; cuSolver DN handle creation), so the committed cache
# is CPU-provenanced. Override with JAX_PLATFORMS=gpu (non-deterministic).
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
echo "[reproduce_all.sh] JAX backend pinned: JAX_PLATFORMS=${JAX_PLATFORMS}" >&2
if [ "${JAX_PLATFORMS}" != "cpu" ]; then
    echo "[reproduce_all.sh] WARNING: non-CPU backend selected." >&2
    echo "[reproduce_all.sh] GPU runs are non-deterministic; CPU is canonical." >&2
fi

cd "${SCRIPT_DIR}"

SECONDS=0

if [ "$KEEP_CACHE" = false ] && [ -z "$STAGE_ONLY" -o "$STAGE_ONLY" = "core" ]; then
    echo "============================================================"
    echo " Step 0: Clearing cached results and figures"
    echo "============================================================"
    find "${RESULTS_DIR}" -name '*.npz' -delete 2>/dev/null || true
    find "${RESULTS_DIR}" -name '*.json' -delete 2>/dev/null || true
    find "${RESULTS_DIR}" -name '*.tex' -delete 2>/dev/null || true
    find "${FIGURES_DIR}" -name '*.pdf' -delete 2>/dev/null || true
    echo " Cleared results/ and figures/*.pdf"
    echo ""
fi

run_core() {
    echo "============================================================"
    echo " Stage: Core computation"
    echo "============================================================"

    echo ""
    echo "[1/5] run_analysis.py Full metric analysis sweep"
    $PYTHON "${SCRIPT_DIR}/scripts/run_analysis.py"

    echo ""
    echo "[2/5] run_convergence.py Richardson extrapolation convergence"
    $PYTHON "${SCRIPT_DIR}/scripts/run_convergence.py"

    echo ""
    echo "[3/5] run_kinematic_scalars.py Kinematic scalar fields"
    $PYTHON "${SCRIPT_DIR}/scripts/run_kinematic_scalars.py"

    echo ""
    echo "[4/5] run_geodesics.py Geodesic integration & tidal forces"
    $PYTHON "${SCRIPT_DIR}/scripts/run_geodesics.py"

    echo ""
    echo "[5/5] run_clustered_convergence.py Wall-clustered convergence"
    $PYTHON "${SCRIPT_DIR}/scripts/run_clustered_convergence.py" \
        --resolutions 25 50 100 \
        --include-rodal-matched \
        --n-starts 8

    echo ""
    echo " Core stage complete."
    echo ""
}

run_ablation() {
    echo "============================================================"
    echo " Stage: Ablation & supplementary studies"
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
    echo "[+] run_superluminal_investigation.py Superluminal characterization"
    $PYTHON "${SCRIPT_DIR}/scripts/run_superluminal_investigation.py"

    echo ""
    echo "[+] run_rodal_matched_resolution.py Rodal matched-param feasibility"
    $PYTHON "${SCRIPT_DIR}/scripts/run_rodal_matched_resolution.py"

    echo ""
    echo "[+] run_lentz_wall_assessment.py Lentz wall resolution assessment"
    $PYTHON "${SCRIPT_DIR}/scripts/run_lentz_wall_assessment.py"

    echo ""
    echo "[+] run_wall_restricted_analysis.py Wall-restricted Type-IV analysis"
    $PYTHON "${SCRIPT_DIR}/scripts/run_wall_restricted_analysis.py"

    echo ""
    echo " Ablation stage complete."
    echo ""
}

run_figures() {
    echo "============================================================"
    echo " Stage: Figure generation"
    echo "============================================================"

    echo ""
    echo "[1/2] reproduce_figures.py Generate all figure PDFs"
    $PYTHON "${SCRIPT_DIR}/scripts/reproduce_figures.py" \
        --figures-dir "${FIGURES_DIR}/" \
        --results-dir "${RESULTS_DIR}/"

    echo ""
    echo "[2/2] generate_vdb_comparison_figures.py VdB NEC/WEC/SEC/DEC panels"
    $PYTHON "${SCRIPT_DIR}/scripts/generate_vdb_comparison_figures.py"

    echo ""
    echo " Figures stage complete."
    echo ""
}

case "${STAGE_ONLY}" in
    core) run_core ;;
    ablation) run_ablation ;;
    figures) run_figures ;;
    "")
        run_core
        run_ablation
        run_figures
        ;;
esac

ELAPSED=$SECONDS
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

echo "============================================================"
echo " Reproduction complete!"
echo " Total time: ${MINS}m ${SECS}s"
echo ""
echo " Results: $(find "${RESULTS_DIR}" -name '*.json' -o -name '*.npz' | wc -l) files"
echo " Figures: $(find "${FIGURES_DIR}" -name '*.pdf' 2>/dev/null | wc -l) PDFs"
echo "============================================================"
