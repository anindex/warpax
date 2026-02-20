"""Observer-robust energy condition verification (pure JAX).

Two-tier strategy: Hawking-Ellis classification + eigenvalue checks
for Type I points, Optimistix BFGS optimization over observer space
for all types.  Eulerian-frame comparison available separately.
"""

# Types
from .types import (
    ClassificationResult,
    ECGridResult,
    ECPointResult,
    ECSummary,
)

# Classification
from .classification import classify_hawking_ellis, classify_mixed_tensor

# Eigenvalue checks
from .eigenvalue_checks import (
    check_all,
    check_dec,
    check_nec,
    check_sec,
    check_wec,
)

# Observer parameterization
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

# Optimisation
from .optimization import (
    OptimizationResult,
    optimize_dec,
    optimize_nec,
    optimize_point,
    optimize_sec,
    optimize_wec,
)

# Observer sweep
from .sweep import (
    cross_validate_sweep,
    make_angular_observers,
    make_rapidity_observers,
    sweep_all_margins,
    sweep_nec_margins,
    sweep_wec_margins,
)

# Verifier orchestrator
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
    "OptimizationResult",
    # Classification
    "classify_hawking_ellis",
    "classify_mixed_tensor",
    # Eigenvalue checks
    "check_all",
    "check_dec",
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
    # Optimisation
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
]
