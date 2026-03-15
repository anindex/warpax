"""Differentiable shape-function metric design.

Public API:

- :class:`ShapeFunction` - differentiable basis library
  (cubic B-spline via interpax + pure-JAX Bernstein + Gaussian-mixture-model).
- :class:`ShapeFunctionMetric` - :class:`ADMMetric`
  subclass wrapping a :class:`ShapeFunction` with a construction-time
  ``verify_physical`` gate.
- :func:`bubble_size_constraint`, :func:`velocity_constraint`,
  :func:`boundedness_constraint`, :class:`ConstraintResult`,
  :data:`CONSTRAINT_REGISTRY` - signed-margin constraint library consumed
  by the optimizer.
- :func:`ec_margin_objective`, :func:`averaged_objective`,
  :func:`quantum_objective`, :data:`OBJECTIVE_REGISTRY` - EC-margin / ANEC /
  Ford-Roman objective library consumed by the optimizer.
- :func:`design_metric`, :class:`OptimizationReport` - constrained-BFGS
  shape-function optimizer with sigmoid reparameterization.
"""
from __future__ import annotations

from .constraints import (
    CONSTRAINT_REGISTRY,
    ConstraintResult,
    all_constraints_satisfied,
    boundedness_constraint,
    bubble_size_constraint,
    velocity_constraint,
)
from .metrics import (
    PhysicalityVerdict,
    ShapeFunctionMetric,
    UnphysicalMetricError,
    UnphysicalMetricWarning,
)
from .objectives import (
    OBJECTIVE_REGISTRY,
    averaged_objective,
    ec_margin_objective,
    quantum_objective,
)
from .optimizer import OptimizationReport, design_metric
from .shape_functions import ShapeFunction

__all__ = [
    "CONSTRAINT_REGISTRY",
    "ConstraintResult",
    "OBJECTIVE_REGISTRY",
    "OptimizationReport",
    "PhysicalityVerdict",
    "ShapeFunction",
    "ShapeFunctionMetric",
    "UnphysicalMetricError",
    "UnphysicalMetricWarning",
    "all_constraints_satisfied",
    "averaged_objective",
    "boundedness_constraint",
    "bubble_size_constraint",
    "design_metric",
    "ec_margin_objective",
    "quantum_objective",
    "velocity_constraint",
]
