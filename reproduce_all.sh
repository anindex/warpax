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
#         figures (figure generation),
#         gate (prose/table consistency check; runs last, needs everything above),
#         enclosures (certified interval branch-and-bound; hours, opt-in only).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
FIGURES_DIR="${SCRIPT_DIR}/figures"
PAPER_FIGURES_DIR="${SCRIPT_DIR}/../warpax_arxiv/figures"

# Figures are generated into the codebase (FIGURES_DIR), then copied to the
# paper folder. This keeps a single source of truth in the repo and a synced
# copy under warpax_arxiv/ for the manuscript build.
sync_figures_to_paper() {
    echo "  Copying figures: ${FIGURES_DIR}/*.pdf -> ${PAPER_FIGURES_DIR}/"
    mkdir -p "${PAPER_FIGURES_DIR}"
    cp -f "${FIGURES_DIR}"/*.pdf "${PAPER_FIGURES_DIR}/" 2>/dev/null || true
}

KEEP_CACHE=false
STAGE_ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --keep-cache) KEEP_CACHE=true ;;
        --stage)
            shift
            if [ $# -eq 0 ]; then
                echo "Error: --stage requires an argument (core, ablation, figures, enclosures)" >&2
                exit 1
            fi
            STAGE_ONLY="$1"
            ;;
        core|ablation|figures|enclosures|gate) STAGE_ONLY="$1" ;;
        -h|--help)
            echo "Usage: $0 [--keep-cache] [--stage core|ablation|figures|enclosures]"
            echo "  --keep-cache   Skip cache deletion (only recompute missing results)"
            echo "  --stage NAME   Run only one stage (core, ablation, figures, gate, enclosures)"
            echo "                 'enclosures' is hours of interval branch-and-bound;"
            echo "                 it is deliberately not part of 'all'."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

PYTHON="${PYTHON:-python}"

# Check the interpreter BEFORE step 0 deletes results/. A bare `python` is not on
# PATH under uv-managed environments, so the default sent one run straight through
# the cache wipe and into "command not found", destroying every artifact it was
# about to regenerate. Fail here instead, while the cache is still intact.
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "[reproduce_all.sh] PYTHON='${PYTHON}' is not executable." >&2
    echo "[reproduce_all.sh] Nothing was deleted. Re-run with, e.g.:" >&2
    echo "[reproduce_all.sh]   PYTHON=./.venv/bin/python $0 $*" >&2
    exit 1
fi
export PYTHONPATH="${SCRIPT_DIR:-$PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! "${PYTHON}" -c 'import warpax' >/dev/null 2>&1; then
    echo "[reproduce_all.sh] PYTHON='${PYTHON}' cannot import warpax." >&2
    echo "[reproduce_all.sh] Nothing was deleted. Use the project venv:" >&2
    echo "[reproduce_all.sh]   PYTHON=./.venv/bin/python $0 $*" >&2
    exit 1
fi

# Pin JAX backend to CPU by default for deterministic reproduction.
# Blackwell sm_120 with jax[cuda12]==0.10.0 crashes in two paths
# (cuBLAS LT autotuner; cuSolver DN handle creation), so the committed cache
# is CPU-provenanced. Override with JAX_PLATFORMS=gpu (non-deterministic).
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
# Persistent XLA compilation cache: each stage is a fresh python process,
# so without this every stage repays full compile cost. Numerics-safe
# (cached artifacts are the same XLA programs). Disable with =0.
export WARPAX_JIT_CACHE="${WARPAX_JIT_CACHE:-1}"
# JAX's 1s default floor skips most of our kernels; 0.05s captures them.
export WARPAX_JIT_CACHE_MIN_COMPILE_TIME_SECS="${WARPAX_JIT_CACHE_MIN_COMPILE_TIME_SECS:-0.05}"
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
    echo "[K1] run_velocity_sweep.py Velocity-resolved Hawking-Ellis type map"
    $PYTHON "${SCRIPT_DIR}/scripts/run_velocity_sweep.py"

    echo ""
    echo "[K2] run_invariant_verification.py Invariant all-observer benchmark"
    $PYTHON "${SCRIPT_DIR}/scripts/run_invariant_verification.py"

    echo ""
    echo "[K3] validate_superluminal_classification.py Type-IV 3-solver/50-digit gate"
    $PYTHON "${SCRIPT_DIR}/scripts/validate_superluminal_classification.py"

    echo ""
    echo "[K4] run_matched_benchmark.py Matched wall-resolved benchmark + convergence"
    $PYTHON "${SCRIPT_DIR}/scripts/run_matched_benchmark.py"

    echo ""
    echo "[K5] run_shift_vorticity.py Shift vorticity controls the type (reads K1)"
    $PYTHON "${SCRIPT_DIR}/scripts/run_shift_vorticity.py"

    echo ""
    echo "[K6] run_anec_retained.py Averaged null energy along null rays"
    $PYTHON "${SCRIPT_DIR}/scripts/run_anec_retained.py"

    echo ""
    echo "[K6b] run_anec_symplectic.py Rigorous geodesic-integrated ANEC (symplectic + witness)"
    $PYTHON "${SCRIPT_DIR}/scripts/run_anec_symplectic.py"

    echo ""
    echo "[K7] run_quantum_inequality.py Ford-Roman quantum inequality (reads K6)"
    $PYTHON "${SCRIPT_DIR}/scripts/run_quantum_inequality.py"

    echo ""
    echo "[K8] run_construction_verification.py Cross-construction all-observer verification"
    $PYTHON "${SCRIPT_DIR}/scripts/run_construction_verification.py"

    echo ""
    echo "[K9] run_exoticity_ranking.py Boost-invariant exoticity ranking + v_s scaling laws (reads K1, K6b)"
    $PYTHON "${SCRIPT_DIR}/scripts/run_exoticity_ranking.py"

    echo ""
    echo "[K10] derive_vorticity_type.py Vorticity -> Type-IV mechanism (f = kappa omega)"
    $PYTHON "${SCRIPT_DIR}/scripts/derive_vorticity_type.py"

    echo ""
    echo "[K11] run_curvature_scaling.py Universal v_s scaling of wall curvature invariants"
    $PYTHON "${SCRIPT_DIR}/scripts/run_curvature_scaling.py"

    echo ""
    echo "[K12] run_ssv_bound.py SSV NEC lower-bound saturation (reads K1)"
    $PYTHON "${SCRIPT_DIR}/scripts/run_ssv_bound.py"

    echo ""
    echo "[K13] run_delta_crosscheck.py Momentum discriminant vs eigensolver agreement"
    $PYTHON "${SCRIPT_DIR}/scripts/run_delta_crosscheck.py"

    echo ""
    echo "[K14] run_integrated_negative_energy.py Slice-integrated negative energy"
    $PYTHON "${SCRIPT_DIR}/scripts/run_integrated_negative_energy.py"

    echo ""
    echo "[K15] run_rodal_sigma_resolved.py Wall-resolved Rodal sigma sweep (item A5)"
    $PYTHON "${SCRIPT_DIR}/scripts/run_rodal_sigma_resolved.py"

    echo ""
    echo "[K16] run_classifier_audit.py Jordan displacement limit + LMI label audit"
    $PYTHON "${SCRIPT_DIR}/scripts/run_classifier_audit.py"

    echo ""
    echo "[K16b] run_type_transition_audit.py LMI across the Type-II locus (analytic)"
    $PYTHON "${SCRIPT_DIR}/scripts/run_type_transition_audit.py"

    echo ""
    echo "[K16c] run_lmi_audit.py Type-free LMI vs the type-based route, at every grid point"
    $PYTHON "${SCRIPT_DIR}/scripts/run_lmi_audit.py"
    echo ""

    echo "[K17] emit_paper_numbers.py Emit the auto-sourced macros"
    $PYTHON "${SCRIPT_DIR}/scripts/emit_paper_numbers.py"

    echo ""
    echo "[1/8] run_analysis.py Full metric analysis sweep"
    $PYTHON "${SCRIPT_DIR}/scripts/run_analysis.py"

    echo ""
    echo "[2/8] run_convergence.py Richardson extrapolation convergence"
    $PYTHON "${SCRIPT_DIR}/scripts/run_convergence.py"

    echo ""
    echo "[3/8] run_kinematic_scalars.py Kinematic scalar fields"
    $PYTHON "${SCRIPT_DIR}/scripts/run_kinematic_scalars.py"

    echo ""
    echo "[4/8] run_geodesics.py Geodesic integration & tidal forces"
    $PYTHON "${SCRIPT_DIR}/scripts/run_geodesics.py"

    echo ""
    echo "[5/8] run_clustered_convergence.py Wall-clustered convergence"
    $PYTHON "${SCRIPT_DIR}/scripts/run_clustered_convergence.py" \
        --resolutions 25 50 100 \
        --include-rodal-matched \
        --n-starts 8

    echo ""
    echo "[6/8] run_diagnostic_convergence.py Per-diagnostic wall-resolved certification"
    $PYTHON "${SCRIPT_DIR}/scripts/run_diagnostic_convergence.py"

    echo ""
    echo "[7/8] run_extra_convergence.py Exoticity index + ANEC-minimum convergence"
    $PYTHON "${SCRIPT_DIR}/scripts/run_extra_convergence.py"

    echo ""
    echo "[8/8] run_curvature_convergence.py Curvature-exponent resolution stability"
    # No --velocities override: the default is the scaling table's own window.
    $PYTHON "${SCRIPT_DIR}/scripts/run_curvature_convergence.py"

    echo ""
    echo " Core stage complete."
    echo ""
}

run_enclosures() {
    echo ""
    echo "[E1] run_enclosures.py Certified global enclosures of the wall null deficit"
    echo "     (interval branch-and-bound; hours, not minutes)"
    $PYTHON "${SCRIPT_DIR}/scripts/run_enclosures.py" --max-boxes 120000
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
    echo "[+] run_rodal_native_resolution.py Rodal native-param resolution stability"
    $PYTHON "${SCRIPT_DIR}/scripts/run_rodal_native_resolution.py"

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
    echo "[1/3] reproduce_figures.py Generate all figure PDFs"
    $PYTHON "${SCRIPT_DIR}/scripts/reproduce_figures.py" \
        --figures-dir "${FIGURES_DIR}/" \
        --results-dir "${RESULTS_DIR}/"

    echo ""
    echo "[2/3] generate_vdb_comparison_figures.py VdB NEC/WEC/SEC/DEC panels"
    $PYTHON "${SCRIPT_DIR}/scripts/generate_vdb_comparison_figures.py"

    echo ""
    echo "[3/3] emit_diagnostic_tables.py Diagnostic/ablation LaTeX tables"
    $PYTHON "${SCRIPT_DIR}/scripts/emit_diagnostic_tables.py"

    echo ""
    echo "[manifest] write_manifest.py Integrity record for the cached grids"
    $PYTHON "${SCRIPT_DIR}/scripts/write_manifest.py"

    echo ""
    echo "[sync] Copy generated figures into the paper folder"
    sync_figures_to_paper

    echo ""
    echo " Figures stage complete."
    echo ""
}

# The consistency gate belongs AFTER everything, not inside core. It used to run as
# K17, roughly two thirds of the way through the core stage -- before run_analysis.py
# had written a single .npz, so write_manifest.check() reported all 41 cached grids
# absent, the gate exited 1, and `set -e` tore the run down before the steps that
# would have produced them. A gate placed where its inputs do not exist yet does not
# check the pipeline; it just stops it.
run_gate() {
    echo "============================================================"
    echo " Stage: Prose/table consistency gate"
    echo "============================================================"
    echo ""
    $PYTHON "${SCRIPT_DIR}/scripts/check_paper_numbers.py"
}

case "${STAGE_ONLY}" in
    core) run_core ;;
    ablation) run_ablation ;;
    figures) run_figures ;;
    gate) run_gate ;;
    "")
        run_core
        run_ablation
        run_figures
        run_gate
        ;;
    enclosures)
        run_enclosures
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
