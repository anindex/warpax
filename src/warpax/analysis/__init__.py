"""Analysis library for Eulerian vs robust energy condition comparison.

Provides three submodules:
- **comparison**: Eulerian vs robust EC comparison with missed-flag logic
- **convergence**: Richardson extrapolation for grid convergence validation
- **kinematic_scalars**: Expansion, shear, vorticity for Eulerian congruence
"""
from __future__ import annotations

from .comparison import ComparisonResult, build_comparison_table, compare_eulerian_vs_robust
from .convergence import compute_convergence_quantity, richardson_extrapolation
from .kinematic_scalars import compute_kinematic_scalars, compute_kinematic_scalars_grid

__all__ = [
    "ComparisonResult",
    "build_comparison_table",
    "compare_eulerian_vs_robust",
    "compute_convergence_quantity",
    "compute_kinematic_scalars",
    "compute_kinematic_scalars_grid",
    "richardson_extrapolation",
]
