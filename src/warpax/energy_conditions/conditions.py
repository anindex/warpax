"""Energy condition verifier orchestrator.

Two-tier strategy:
1. Classify T^a_b via Hawking-Ellis at each grid point.
2. For Type I: use fast eigenvalue algebraic checks (O(1)/point).
3. For non-Type-I (or as validation): fall back to optimization over observer space.

Returns margin fields (negative = violated), worst-observer directions, and satisfaction booleans.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .eigenvalue_method import check_eigenvalue_grid
from .optimization_method import (
    check_dec_optimization,
    check_nec_optimization,
    check_sec_optimization,
    check_wec_optimization,
)
from .type_classification import HawkingEllisType, classify_point


@dataclass
class EnergyConditionResult:
    """Complete energy condition analysis at a grid point or across a grid."""

    margins: dict[str, NDArray]  # condition_name -> margin field (*grid_shape,)
    satisfied: dict[str, bool]  # condition_name -> global satisfaction
    worst_observer_params: dict[str, NDArray | None] = field(default_factory=dict)
    method_used: NDArray | None = None  # 'eigenvalue' or 'optimization' at each point


class EnergyConditionVerifier:
    """Orchestrator for observer-robust energy condition verification.

    Parameters
    ----------
    optimization_fallback : bool
        If True, always run optimization even for Type-I points (for validation).
    n_starts : int
        Number of multi-start points for optimization.
    zeta_max : float
        Maximum rapidity for optimization bounds.
    """

    def __init__(
        self,
        optimization_fallback: bool = False,
        n_starts: int = 8,
        zeta_max: float = 5.0,
    ):
        self.optimization_fallback = optimization_fallback
        self.n_starts = n_starts
        self.zeta_max = zeta_max

    def verify_point(
        self,
        T_ab: NDArray,
        g_ab: NDArray,
        use_optimization: bool = False,
    ) -> dict[str, dict]:
        """Verify all energy conditions at a single spacetime point.

        Parameters
        ----------
        T_ab : NDArray
            Stress-energy tensor T_{ab}, shape (4, 4).
        g_ab : NDArray
            Metric tensor g_{ab}, shape (4, 4).
        use_optimization : bool
            Force optimization method even for Type-I.

        Returns
        -------
        dict
            Maps condition name to dict with 'satisfied', 'margin', and optionally
            'worst_params', 'worst_vector'.
        """
        g_inv = np.linalg.inv(g_ab)
        T_mixed = g_inv @ T_ab

        classification = classify_point(T_mixed, g_ab)

        if classification.type == HawkingEllisType.TYPE_I and not use_optimization:
            from .eigenvalue_method import check_all_eigenvalue

            result = check_all_eigenvalue(classification.rho, classification.pressures)
            return {
                name: {"satisfied": r.satisfied, "margin": r.margin, "method": "eigenvalue"}
                for name, r in result.items()
            }

        # Optimization path
        results = {}
        for name, check_fn in [
            ("WEC", check_wec_optimization),
            ("NEC", check_nec_optimization),
            ("DEC", check_dec_optimization),
            ("SEC", check_sec_optimization),
        ]:
            kwargs = {"T_ab": T_ab, "g_ab": g_ab, "n_starts": self.n_starts}
            if name != "NEC":
                kwargs["zeta_max"] = self.zeta_max
            r = check_fn(**kwargs)
            results[name] = {
                "satisfied": r.satisfied,
                "margin": r.margin,
                "worst_params": r.worst_params,
                "worst_vector": r.worst_vector,
                "method": "optimization",
            }

        return results

    def verify_grid(
        self, T_field: NDArray, g_field: NDArray, g_inv_field: NDArray
    ) -> EnergyConditionResult:
        """Verify energy conditions across an entire grid.

        Uses the two-tier strategy: eigenvalue method for Type-I points,
        optimization for non-Type-I points.

        Parameters
        ----------
        T_field : NDArray
            Stress-energy T_{ab}, shape (*grid_shape, 4, 4).
        g_field : NDArray
            Metric g_{ab}, shape (*grid_shape, 4, 4).
        g_inv_field : NDArray
            Inverse metric g^{ab}, shape (*grid_shape, 4, 4).

        Returns
        -------
        EnergyConditionResult
        """
        grid_shape = T_field.shape[:-2]
        n_points = int(np.prod(grid_shape))

        flat_T = T_field.reshape(-1, 4, 4)
        flat_g = g_field.reshape(-1, 4, 4)
        flat_g_inv = g_inv_field.reshape(-1, 4, 4)

        # Initialize margin fields
        margins = {name: np.full(n_points, np.inf) for name in ["WEC", "NEC", "DEC", "SEC"]}
        method_used = np.empty(n_points, dtype="U12")

        # Classify all points and collect Type-I eigenvalues
        rho_field = np.zeros(n_points)
        pressure_field = np.zeros((n_points, 3))
        is_type_i = np.zeros(n_points, dtype=bool)

        for i in range(n_points):
            T_mixed = flat_g_inv[i] @ flat_T[i]
            classification = classify_point(T_mixed, flat_g[i])

            if classification.type == HawkingEllisType.TYPE_I:
                is_type_i[i] = True
                rho_field[i] = classification.rho
                pressure_field[i] = classification.pressures

        # Batch eigenvalue check for Type-I points
        if np.any(is_type_i):
            idx = np.where(is_type_i)[0]
            eigen_margins = check_eigenvalue_grid(rho_field[idx], pressure_field[idx])
            for name, m in eigen_margins.items():
                margins[name][idx] = m
            method_used[idx] = "eigenvalue"

        # Optimization for non-Type-I points
        non_type_i_idx = np.where(~is_type_i)[0]
        for i in non_type_i_idx:
            point_results = self.verify_point(flat_T[i], flat_g[i], use_optimization=True)
            for name, r in point_results.items():
                margins[name][i] = r["margin"]
            method_used[i] = "optimization"

        # Also run optimization on Type-I points if fallback requested
        if self.optimization_fallback:
            for i in np.where(is_type_i)[0]:
                point_results = self.verify_point(flat_T[i], flat_g[i], use_optimization=True)
                for name, r in point_results.items():
                    # Take the worse (smaller) margin
                    margins[name][i] = min(margins[name][i], r["margin"])

        # Reshape margins back to grid shape
        margin_fields = {
            name: m.reshape(grid_shape) for name, m in margins.items()
        }
        satisfied = {name: bool(np.all(m >= -1e-12)) for name, m in margin_fields.items()}

        return EnergyConditionResult(
            margins=margin_fields,
            satisfied=satisfied,
            method_used=method_used.reshape(grid_shape),
        )
