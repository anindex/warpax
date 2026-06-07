"""Analysis library for Eulerian vs robust energy condition comparison.

Provides four submodules:
- **comparison**: Eulerian vs robust EC comparison with missed-flag logic
- **convergence**: Richardson extrapolation for grid convergence validation
- **kinematic_scalars**: Expansion, shear, vorticity for the Eulerian congruence
- **shift_kinematics**: Expansion, shear, vorticity of the ADM shift vector field
"""
from __future__ import annotations

from .comparison import ComparisonResult, build_comparison_table, compare_eulerian_vs_robust
from .convergence import compute_convergence_quantity, richardson_extrapolation
from .invariant_verification import (
    integrated_exotic_content,
    peak_proper_energy_deficit,
    reduction_factors,
    single_frame_miss,
)
from .kinematic_scalars import compute_kinematic_scalars, compute_kinematic_scalars_grid
from .shift_kinematics import (
    compute_shift_kinematics,
    compute_shift_kinematics_grid,
    rotationality,
)

__all__ = [
    "ComparisonResult",
    "build_comparison_table",
    "compare_eulerian_vs_robust",
    "compute_convergence_quantity",
    "compute_kinematic_scalars",
    "compute_kinematic_scalars_grid",
    "compute_shift_kinematics",
    "compute_shift_kinematics_grid",
    "integrated_exotic_content",
    "peak_proper_energy_deficit",
    "reduction_factors",
    "richardson_extrapolation",
    "rotationality",
    "single_frame_miss",
]
