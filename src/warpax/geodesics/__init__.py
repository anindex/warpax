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
    eulerian_affine_scale,
    eulerian_frequency,
    killing_energy,
    null_ic,
    null_ic_eulerian_normalized,
    null_ic_killing_normalized,
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
from .symplectic import (
    SymplecticGeodesicResult,
    integrate_geodesic_symplectic,
    integrate_geodesic_symplectic_family,
    null_ic_canonical,
    timelike_ic_canonical,
)

__all__ = [
    # Jacobi deviation
    "DeviationResult",
    # Core integrator
    "GeodesicResult",
    # Symplectic (structure-preserving) integrator
    "SymplecticGeodesicResult",
    # Observables
    "blueshift_along_trajectory",
    "bounding_box_event",
    "circular_orbit_ic",
    "compute_blueshift",
    "eulerian_affine_scale",
    "eulerian_frequency",
    "geodesic_deviation_vector_field",
    "geodesic_vector_field",
    "horizon_event",
    "integrate_geodesic",
    "integrate_geodesic_family",
    "integrate_geodesic_symplectic",
    "integrate_geodesic_symplectic_family",
    "integrate_geodesic_with_deviation",
    "killing_energy",
    "make_event",
    "monitor_conservation",
    "null_ic",
    "null_ic_canonical",
    "null_ic_eulerian_normalized",
    "null_ic_killing_normalized",
    "proper_time_elapsed",
    "radial_infall_ic",
    "tidal_eigenvalues",
    "tidal_tensor",
    "timelike_ic",
    "timelike_ic_canonical",
    "velocity_norm",
]
