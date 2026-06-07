"""Observer-robust energy-condition verification (pure JAX).

Two-tier strategy: Hawking-Ellis classification plus exact eigenvalue
margins for Type I points, Optimistix BFGS over the observer manifold
for all four conditions. Eulerian-frame margins are available
separately for clean single-frame comparisons.
"""

from .classification import (
    classify_hawking_ellis,
    classify_mixed_tensor,
    classify_with_solver,
)
from .classification_mpmath import (
    classify_hawking_ellis_mpmath,
    eigenvalues_mpmath,
    verify_classification_at_points,
)
from .eigenvalue_checks import (
    check_all,
    check_dec,
    check_dec_typeI_eigenvalue_bound,
    check_nec,
    check_sec,
    check_wec,
)
from .filtering import (
    compute_wall_restricted_stats,
    determinant_guard_mask,
    frobenius_norm_mask,
    shape_function_mask,
)
from .frame_free import (
    certify_grid_frame_free,
    certify_point_frame_free,
    type_fractions,
    typeI_min_margins,
)
from .observer import (
    boost_vector_to_params,
    bounded_param,
    compute_orthonormal_tetrad,
    null_from_angles,
    null_from_stereo,
    stereo_to_params,
    timelike_from_boost_vector,
    timelike_from_rapidity,
)
from .optimization import (
    OptimizationResult,
    optimize_dec,
    optimize_nec,
    optimize_point,
    optimize_sec,
    optimize_wec,
)
from .sweep import (
    cross_validate_sweep,
    make_angular_observers,
    make_rapidity_observers,
    sweep_all_margins,
    sweep_nec_margins,
    sweep_wec_margins,
)
from .types import (
    ClassificationResult,
    ECGridResult,
    ECPointResult,
    ECSummary,
    FrameFreeGridResult,
    WallRestrictedStats,
)
from .worst_observer_analytic import (
    boosted_energy_density,
    worst_observer_typeI,
)
from .verifier import (
    anec_integrand,
    compute_eulerian_ec,
    verify_grid,
    verify_point,
)

__all__ = [
    # Types
    "ClassificationResult",
    "ECGridResult",
    "ECPointResult",
    "ECSummary",
    "FrameFreeGridResult",
    "OptimizationResult",
    "WallRestrictedStats",
    # Classification
    "classify_hawking_ellis",
    "classify_hawking_ellis_mpmath",
    "classify_mixed_tensor",
    "classify_with_solver",
    "eigenvalues_mpmath",
    "verify_classification_at_points",
    # Eigenvalue checks
    "check_all",
    "check_dec",
    "check_dec_typeI_eigenvalue_bound",
    "check_nec",
    "check_sec",
    "check_wec",
    # Observer
    "boost_vector_to_params",
    "bounded_param",
    "compute_orthonormal_tetrad",
    "null_from_angles",
    "null_from_stereo",
    "stereo_to_params",
    "timelike_from_boost_vector",
    "timelike_from_rapidity",
    # Optimization
    "optimize_dec",
    "optimize_nec",
    "optimize_point",
    "optimize_sec",
    "optimize_wec",
    # Observer sweep
    "make_rapidity_observers",
    "make_angular_observers",
    "sweep_wec_margins",
    "sweep_nec_margins",
    "sweep_all_margins",
    "cross_validate_sweep",
    # Verifier
    "anec_integrand",
    "compute_eulerian_ec",
    "verify_grid",
    "verify_point",
    # Filtering
    "compute_wall_restricted_stats",
    "determinant_guard_mask",
    "frobenius_norm_mask",
    "shape_function_mask",
    # Frame-free (all-velocity) certification
    "certify_grid_frame_free",
    "certify_point_frame_free",
    "type_fractions",
    "typeI_min_margins",
    # Analytic worst observer
    "boosted_energy_density",
    "worst_observer_typeI",
]
