"""Domain-specific exceptions for source-consistent warp-shell analysis."""


class WarpAXError(Exception):
    """Base class for all warpax-specific errors."""


class ConstraintViolationError(WarpAXError):
    """ADM constraint residual exceeds acceptable threshold."""

    def __init__(self, residual_type: str, max_residual: float, location: str) -> None:
        self.residual_type = residual_type
        self.max_residual = max_residual
        self.location = location
        super().__init__(
            f"{residual_type} constraint violated: "
            f"max_residual={max_residual:.6e} at {location}"
        )


class TOVInconsistencyError(WarpAXError):
    """Anisotropic TOV equilibrium equation not satisfied."""

    def __init__(self, max_residual: float, r_location: float) -> None:
        self.max_residual = max_residual
        self.r_location = r_location
        super().__init__(
            f"TOV inconsistency: residual={max_residual:.6e} at r={r_location:.6e}"
        )


class JunctionDiscontinuityError(WarpAXError):
    """Metric lacks required smoothness at junction surface."""

    def __init__(self, surface_label: str, jump_magnitude: float) -> None:
        self.surface_label = surface_label
        self.jump_magnitude = jump_magnitude
        super().__init__(
            f"Junction discontinuity at {surface_label}: "
            f"jump={jump_magnitude:.6e}"
        )


class AsymptoticFalloffError(WarpAXError):
    """Metric does not decay sufficiently at spatial infinity."""

    def __init__(self, measured_order: float, required_order: int) -> None:
        self.measured_order = measured_order
        self.required_order = required_order
        super().__init__(
            f"Asymptotic falloff insufficient: "
            f"measured O(1/r^{measured_order:.2f}), "
            f"required O(1/r^{required_order})"
        )


class TransportUndefinedError(WarpAXError):
    """Transport diagnostic could not be computed (e.g., geodesic did not converge)."""

    def __init__(self, geodesic_id: int, termination_reason: str) -> None:
        self.geodesic_id = geodesic_id
        self.termination_reason = termination_reason
        super().__init__(
            f"Transport undefined for geodesic {geodesic_id}: {termination_reason}"
        )
