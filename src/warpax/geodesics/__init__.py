"""Geodesic integration for Lorentzian spacetimes.

Provides timelike and null geodesic integration through arbitrary spacetimes
using Diffrax (JAX-native adaptive ODE solver). Supports batched geodesic
families via vmap, event-based termination, initial condition helpers,
Jacobi geodesic deviation (tidal forces), and physical observables
(blueshift, conservation monitoring, proper time).
"""
from __future__ import annotations

from .deviation import (
    DeviationResult,
    geodesic_deviation_vector_field,
    integrate_geodesic_with_deviation,
    tidal_eigenvalues,
    tidal_tensor,
)
from .initial_conditions import (
    circular_orbit_ic,
    null_ic,
    radial_infall_ic,
    timelike_ic,
)
from .integrator import (
    GeodesicResult,
    bounding_box_event,
    geodesic_vector_field,
    horizon_event,
    integrate_geodesic,
    integrate_geodesic_family,
    make_event,
)
from .observables import (
    blueshift_along_trajectory,
    compute_blueshift,
    monitor_conservation,
    proper_time_elapsed,
    velocity_norm,
)

__all__ = [
    # Core integrator
    "GeodesicResult",
    "bounding_box_event",
    "circular_orbit_ic",
    "geodesic_vector_field",
    "horizon_event",
    "integrate_geodesic",
    "integrate_geodesic_family",
    "make_event",
    "null_ic",
    "radial_infall_ic",
    "timelike_ic",
    # Jacobi deviation
    "DeviationResult",
    "geodesic_deviation_vector_field",
    "integrate_geodesic_with_deviation",
    "tidal_eigenvalues",
    "tidal_tensor",
    # Observables
    "blueshift_along_trajectory",
    "compute_blueshift",
    "monitor_conservation",
    "proper_time_elapsed",
    "velocity_norm",
]
