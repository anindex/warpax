"""GPU xfail registry: tests expected to fail on Blackwell sm_120.

Materialised from the baseline pytest log capturing the cuBLAS LT
autotuner and cuSolver failure modes on sm_120.

Count
-----
The log (source of truth) shows **79** unique failing test IDs on the
workaround-XLA-no-cublaslt configuration. Any change to this count
requires a CHANGELOG entry.
"""

from __future__ import annotations

__all__ = ["EXPECTED_GPU_FAILURES", "EXPECTED_GPU_FAILURES_COUNT_TARGET"]

# Count locked from log materialisation (2026-04-19).
# See module docstring §"Count deviation" for the 79-vs-117 rationale.
EXPECTED_GPU_FAILURES_COUNT_TARGET: int = 79


EXPECTED_GPU_FAILURES: frozenset[str] = frozenset(
    {
        # --- cuBLAS LT autotuner / cuSolver eig (Mode 1/3) - classifier stack ---
        "tests/test_ec_classification.py::TestJITAndVmap::test_classify_mixed_tensor",
        "tests/test_ec_classification.py::TestJITAndVmap::test_jit_compilation",
        "tests/test_ec_classification.py::TestJITAndVmap::test_vmap_batch",
        "tests/test_ec_classification.py::TestScaleAwareImaginaryTolerance::test_eigenvalues_imag_near_zero_for_diagonal",
        "tests/test_ec_classification.py::TestScaleAwareImaginaryTolerance::test_genuine_type_iv_large_imaginary",
        "tests/test_ec_classification.py::TestScaleAwareImaginaryTolerance::test_large_eigenvalues_tiny_relative_imag_is_type_i",
        "tests/test_ec_classification.py::TestTypeIClassification::test_anisotropic_pressures",
        "tests/test_ec_classification.py::TestTypeIClassification::test_large_eigenvalue_type_i",
        "tests/test_ec_classification.py::TestTypeIClassification::test_near_degenerate",
        "tests/test_ec_classification.py::TestTypeIClassification::test_perfect_fluid_dust",
        "tests/test_ec_classification.py::TestTypeIClassification::test_perfect_fluid_with_pressure",
        "tests/test_ec_classification.py::TestTypeIIClassification::test_null_eigenvector",
        "tests/test_ec_classification.py::TestTypeIIIClassification::test_maximally_degenerate_null_with_relaxed_tol",
        "tests/test_ec_classification.py::TestTypeIIIClassification::test_type_iii_vs_type_i_at_default_tol",
        "tests/test_ec_classification.py::TestTypeIIISyntheticBenchmark::test_type_iii_3x3_block",
        "tests/test_ec_classification.py::TestTypeIIISyntheticBenchmark::test_type_iii_at_large_scale",
        "tests/test_ec_classification.py::TestTypeIIISyntheticBenchmark::test_type_iii_at_small_scale",
        "tests/test_ec_classification.py::TestTypeIIISyntheticBenchmark::test_type_iii_eigenvalues_returned_real",
        "tests/test_ec_classification.py::TestTypeIIISyntheticBenchmark::test_type_iii_rho_and_pressures_nan",
        "tests/test_ec_classification.py::TestTypeIINullDustBenchmark::test_null_dust_classification",
        "tests/test_ec_classification.py::TestTypeIVClassification::test_complex_eigenvalues",
        # --- cuBLAS LT (Mode 1) - mpmath classifier bridge ---
        "tests/test_ec_classification_mpmath.py::TestMpmathClassifier::test_perfect_fluid_mpmath_agrees_with_float64",
        "tests/test_ec_classification_mpmath.py::TestMpmathClassifier::test_type_iv_that_float64_misclassifies",
        # --- cuSolver eig (Mode 3) - verifier / EC pipeline ---
        "tests/test_ec_verifier.py::TestAlcubierreGrid::test_grid_ec_runs",
        "tests/test_ec_verifier.py::TestAlcubierreGrid::test_nec_violation_detected",
        "tests/test_ec_verifier.py::TestAlcubierreGrid::test_some_type_i_points",
        "tests/test_ec_verifier.py::TestAlcubierreGrid::test_summary_fraction_violated",
        "tests/test_ec_verifier.py::TestAlcubierreGrid::test_wec_violation_detected",
        "tests/test_ec_verifier.py::TestDECFutureDirectedness::test_verify_point_dec_catches_past_directed",
        "tests/test_ec_verifier.py::TestDustVerifyPoint::test_all_conditions_satisfied",
        "tests/test_ec_verifier.py::TestDustVerifyPoint::test_he_type_is_one",
        "tests/test_ec_verifier.py::TestDustVerifyPoint::test_result_is_namedtuple",
        "tests/test_ec_verifier.py::TestDustVerifyPoint::test_rho_and_pressures",
        "tests/test_ec_verifier.py::TestDustVerifyPoint::test_wec_margin_value",
        "tests/test_ec_verifier.py::TestECGridResultNewFields::test_classification_stats_present",
        "tests/test_ec_verifier.py::TestECGridResultNewFields::test_dust_all_type_i",
        "tests/test_ec_verifier.py::TestECGridResultNewFields::test_opt_margins_present",
        "tests/test_ec_verifier.py::TestEulerianGridComparison::test_eulerian_grid_runs",
        "tests/test_ec_verifier.py::TestEulerianVsObserverRobust::test_observer_robust_finds_worse_than_eulerian",
        "tests/test_ec_verifier.py::TestEulerianVsObserverRobust::test_observer_robust_wec_violated",
        "tests/test_ec_verifier.py::TestFloat64Dtype::test_verify_point_dtypes",
        "tests/test_ec_verifier.py::TestSummaryStatistics::test_dust_no_violations",
        "tests/test_ec_verifier.py::TestSummaryStatistics::test_summary_finite",
        "tests/test_ec_verifier.py::TestSummaryStatistics::test_summary_types",
        "tests/test_ec_verifier.py::TestWECViolationVerifyPoint::test_wec_violated",
        "tests/test_ec_verifier.py::TestWECViolationVerifyPoint::test_worst_observer_returned",
        "tests/test_ec_verifier.py::TestWorstObserver::test_worst_observer_is_timelike",
        "tests/test_ec_verifier.py::TestWorstObserver::test_worst_params_ranges",
        "tests/test_ec_verifier.py::TestWorstObserver::test_worst_params_shape",
        # --- cuSolver eig (Mode 3) - edge cases ---
        "tests/test_edge_cases.py::TestClassifierNearDegenerateInputs::test_exactly_zero_tensor_classifies_as_type_i",
        "tests/test_edge_cases.py::TestClassifierNearDegenerateInputs::test_large_scale_eigenvalues_with_tiny_imaginary",
        "tests/test_edge_cases.py::TestClassifierNearDegenerateInputs::test_near_degenerate_eigenvalues",
        "tests/test_edge_cases.py::TestClassifierNearDegenerateInputs::test_near_vacuum_classifies_as_type_i",
        "tests/test_edge_cases.py::TestClassifierNearDegenerateInputs::test_type_i_iv_boundary_above_threshold",
        "tests/test_edge_cases.py::TestClassifierNearDegenerateInputs::test_type_i_iv_boundary_below_threshold",
        "tests/test_edge_cases.py::TestDeterminantGuardBoundary::test_superluminal_g00_positive_detected",
        "tests/test_edge_cases.py::TestNaNPropagationAtSharpWalls::test_classifier_with_nan_input_sanitizes",
        "tests/test_edge_cases.py::TestOptimizerConvergenceNearWalls::test_alcubierre_wall_center_converges",
        "tests/test_edge_cases.py::TestOptimizerConvergenceNearWalls::test_rodal_wall_center_converges",
        "tests/test_edge_cases.py::TestOptimizerConvergenceNearWalls::test_sharp_wall_optimizer_finite_margins",
        # --- cuSolver eig (Mode 3) - geodesic observables ---
        "tests/test_geodesic_observables.py::TestSchwarzschildTidalEigenvalues::test_schwarzschild_tidal_eigenvalues",
        "tests/test_geodesic_observables.py::TestTidalTensorFlatSpace::test_tidal_tensor_flat_space",
        # --- cuSolver eig (Mode 3) - physics validation suite ---
        "tests/test_physics_validation/test_geodesics_ec.py::TestSyntheticECVerification::test_dec_violated_large_momentum_flux",
        "tests/test_physics_validation/test_geodesics_ec.py::TestSyntheticECVerification::test_dust_all_satisfied",
        "tests/test_physics_validation/test_geodesics_ec.py::TestSyntheticECVerification::test_eigenvalue_check_consistency",
        "tests/test_physics_validation/test_geodesics_ec.py::TestSyntheticECVerification::test_nec_violated_rho_plus_p_negative",
        "tests/test_physics_validation/test_geodesics_ec.py::TestSyntheticECVerification::test_negative_energy_wec_violated",
        "tests/test_physics_validation/test_geodesics_ec.py::TestSyntheticECVerification::test_radiation_all_satisfied",
        "tests/test_physics_validation/test_geodesics_ec.py::TestSyntheticECVerification::test_sec_violated_but_wec_satisfied",
        "tests/test_physics_validation/test_geodesics_ec.py::TestSyntheticECVerification::test_vacuum_all_trivially_satisfied",
        "tests/test_physics_validation/test_geodesics_ec.py::TestTidalEigenvaluesMultiRadius::test_tidal_eigenvalues_multi_radius[10.0]",
        "tests/test_physics_validation/test_geodesics_ec.py::TestTidalEigenvaluesMultiRadius::test_tidal_eigenvalues_multi_radius[20.0]",
        "tests/test_physics_validation/test_geodesics_ec.py::TestTidalEigenvaluesMultiRadius::test_tidal_eigenvalues_multi_radius[5.0]",
        "tests/test_physics_validation/test_geodesics_ec.py::TestTidalEigenvaluesMultiRadius::test_tidal_eigenvalues_zero_in_flat_space",
        "tests/test_physics_validation/test_warp_metrics.py::TestRodalTypeI::test_rodal_globally_type_i",
        # --- Other (boundary / golden / analysis stragglers) ---
        "tests/test_analysis_integration.py::test_analysis_float64",
        "tests/test_ec_optimization.py::TestBFGSBoundaryStall::test_bfgs_boundary_stall_detected",
        "tests/test_optimizer_prng_golden.py::test_golden_starter_pool_at_default_config",
        "tests/scripts/test_example_07_smoke.py::TestExample07Smoke::test_single_point_runs",
        # TOTAL: 79 entries (see EXPECTED_GPU_FAILURES_COUNT_TARGET)
    }
)
