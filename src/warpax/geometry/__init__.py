"""Differential geometry and tensor calculus on Lorentzian manifolds.

"""

from .geometry import (
    CurvatureResult,
    christoffel_symbols,
    compute_curvature_chain,
    einstein_tensor,
    ricci_scalar,
    ricci_tensor,
    riemann_tensor,
    stress_energy_tensor,
)
from .metric import (
    ADMMetric,
    MetricSpecification,
    SymbolicMetric,
    adm_to_full_metric,
    sympy_metric_inverse_to_jax,
    sympy_metric_to_jax,
)
from .grid import (
    GridCurvatureResult,
    build_coord_batch,
    evaluate_curvature_grid,
)
from .invariants import (
    compute_invariants,
    kretschner_scalar,
    ricci_squared,
    weyl_squared,
)
from .transitions import smoothstep, smoothstep_c1, smoothstep_c2
from .types import GridSpec, TensorField

__all__ = [
    "ADMMetric",
    "CurvatureResult",
    "GridCurvatureResult",
    "GridSpec",
    "MetricSpecification",
    "SymbolicMetric",
    "TensorField",
    "adm_to_full_metric",
    "build_coord_batch",
    "christoffel_symbols",
    "compute_curvature_chain",
    "compute_invariants",
    "einstein_tensor",
    "evaluate_curvature_grid",
    "kretschner_scalar",
    "ricci_scalar",
    "ricci_squared",
    "ricci_tensor",
    "riemann_tensor",
    "smoothstep",
    "smoothstep_c1",
    "smoothstep_c2",
    "stress_energy_tensor",
    "sympy_metric_inverse_to_jax",
    "sympy_metric_to_jax",
    "weyl_squared",
]
