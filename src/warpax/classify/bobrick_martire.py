"""Bobrick-Martire taxonomy classifier for warp drive metrics .

Citation: Bobrick, A. & Martire, G. (2021). "Introducing physical warp
drives." *Classical and Quantum Gravity* 38, 105009. arXiv:2102.08443.

The classifier returns `ClassifiedMetric` with the integer class label
(1 / 2 / 3 / 0=unclassified) plus three diagnostic flags capturing the
intermediate test outcomes. Applied pointwise via a multi-probe cascade.

Textbook
physics; Bobrick-Martire 2021 already cited in paper §2.1.

Class taxonomy (per Bobrick & Martire 2021 §3):

- **Class I** - Vacuum or trivial-fluid metrics with a timelike Killing
  vector. Includes Minkowski (flat vacuum) and Schwarzschild (static
  vacuum). ``T_{ab} ≈ 0`` everywhere (or pure-vacuum gravitational
  field) and ``d g/d t ≈ 0``.
- **Class II** - Alcubierre-family metrics with a smooth bubble-shaped
  shape function that drives a non-trivial stress-energy profile.
  Matter is distributed through a bounded bubble wall region spanning
  a range of radii. Includes Alcubierre, Rodal, Natario, Van den
  Broeck, Lentz.
- **Class III** - Matter-shell / junction-structured metrics whose
  stress-energy is localized inside a finite matter shell (``R_1..R_2``)
  and vanishes for ``r < R_1``. Includes WarpShell.

Dispatch uses a multi-probe sampling cascade:

1. ``stationary`` - ``max_{a,b} |d g_{ab} / d t| < tol_t`` at an
   off-axis mid-range probe.
2. ``shape_function_supported`` -
   ``max_{a,b,c in {x,y,z}} |d g_{ab} / d x^c| > tol_s`` at an
   off-axis mid-range probe.
3. ``has_matter`` - adaptive radius sweep finds non-zero Eulerian
   energy density at some probe; returns the peak radius.
4. ``comoving_fluid`` - ``rho = -T^0{}_0 > 0`` at the matter-peak
   probe.
5. ``is_shell_structured`` - ``rho`` is zero at a small inner radius
   ``r < 0.5`` relative to the matter peak, signalling a bounded
   matter shell (Class III signature) rather than a bubble-like
   profile (Class II).

Class dispatch:

- Class I: NOT has_matter (vacuum) OR pure-vacuum stationary metric
  (``rho`` zero at all probes).
- Class III: has_matter AND is_shell_structured.
- Class II: has_matter AND NOT is_shell_structured AND
  shape_function_supported.
- else: ``0`` (unclassified).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geometry.geometry import compute_curvature_chain
from ..geometry.metric import MetricSpecification


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class ClassifiedMetric(NamedTuple):
    """Result of Bobrick-Martire classification.

    Attributes
    ----------
    bobrick_class : int
        1 = Class I (Killing-field structure, trivial shape function);
        2 = Class II (Alcubierre-family shape-function-supported bubble);
        3 = Class III (matter-shell / junction-structured);
        0 = unclassified.
    stationary : bool
        True if the metric admits a timelike Killing vector field
        (time-derivatives of all metric components vanish within the
        time tolerance at the probe).
    comoving_fluid : bool
        True if positive energy density is present at the matter peak
        (Eulerian ``rho = -T^0{}_0 > 0``), indicating a comoving
        perfect-fluid description.
    shape_function_supported : bool
        True if the metric's components vary non-trivially in the
        spatial coordinates at the probe - indicates a localized
        shape-function structure.
    """

    bobrick_class: int
    stationary: bool
    comoving_fluid: bool
    shape_function_supported: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Default radial probe ladder; covers Alcubierre (R=1), Rodal (R=100),
# WarpShell (R_1=10, R_2=20) and Schwarzschild at a safe range outside
# horizon. Off-axis to avoid y=z=0 degeneracies. Spans ~3 decades on
# logarithmic scale.
_DEFAULT_RADIAL_PROBES: tuple[float, ...] = (
    0.3, 0.6, 1.0, 1.5, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0,
    75.0, 100.0, 120.0, 150.0,
)


def _probe_point(r: float) -> Float[Array, "4"]:
    """Off-axis probe at radius ``r``; avoids y=z=0 degeneracies."""
    return jnp.array([0.0, r * 0.6, r * 0.6, r * 0.529150262], dtype=jnp.float64)


def _metric_jacobian(
    metric: MetricSpecification,
    coords: Float[Array, "4"],
) -> Float[Array, "4 4 4"]:
    """Forward-mode Jacobian ``d g_{ab} / d x^c`` (shape ``(4, 4, 4)``)."""
    return jax.jacfwd(lambda c: metric(c))(coords)


def _energy_density_mixed(
    metric: MetricSpecification,
    coords: Float[Array, "4"],
) -> Float[Array, ""]:
    """Compute the Eulerian energy density ``rho = -T^0{}_0`` at ``coords``."""
    curv = compute_curvature_chain(metric, coords)
    T_mixed = curv.metric_inv @ curv.stress_energy
    return -T_mixed[0, 0]


def _scan_matter(
    metric: MetricSpecification,
    tol_T: float,
) -> tuple[float, Float[Array, "4"]]:
    """Sweep radial probes; return (peak_rho_magnitude, peak_coords)."""
    peak_mag = 0.0
    peak_coords = _probe_point(1.0)
    for r in _DEFAULT_RADIAL_PROBES:
        coords = _probe_point(r)
        rho = _energy_density_mixed(metric, coords)
        mag = float(jnp.abs(rho))
        if not jnp.isfinite(rho):
            continue
        if mag > peak_mag:
            peak_mag = mag
            peak_coords = coords
    return peak_mag, peak_coords


def _scan_inner_vacuum(
    metric: MetricSpecification,
    peak_mag: float,
    peak_coords: Float[Array, "4"],
) -> bool:
    """Return True if ``rho`` is small at an inner radius deep below the
    matter peak - signature of a shell-structured metric.

    A metric is deemed shell-structured when ``|rho(r_inner)|`` drops
    below ``1e-6 * peak_mag`` at ``r_inner = 0.2 * peak_r``. For smooth
    bubble profiles (Alcubierre-family), the inner density stays within
    a few decades of the peak - ratio > 1e-4 typically.
    """
    peak_r = float(jnp.sqrt(jnp.sum(peak_coords[1:] ** 2)))
    inner_r = 0.2 * peak_r
    if inner_r < 0.05:
        return False  # peak too close to origin; shell test unreliable
    coords = _probe_point(inner_r)
    rho = _energy_density_mixed(metric, coords)
    if not jnp.isfinite(rho):
        return False
    ratio = float(jnp.abs(rho)) / max(peak_mag, 1e-30)
    return ratio < 1e-6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def bobrick_martire(
    metric: MetricSpecification,
    probe_coords: Float[Array, "4"] | None = None,
    tol: float = 1e-10,
) -> ClassifiedMetric:
    """Classify a metric against the Bobrick-Martire Class I / II / III taxonomy.

    Parameters
    ----------
    metric : MetricSpecification
        The warp-drive spacetime to classify.
    probe_coords : Float[Array, "4"], optional
        Coordinates at which to probe stationarity + shape-function
        support (default: radial sweep across a ladder of probes that
        covers unit-radius bubbles (Alcubierre, Rodal) and matter shells
        at R_1..R_2 up to ~20 (WarpShell)).
    tol : float
        Baseline tolerance. The following scale-aware thresholds derive
        from it:

        - ``tol_t`` (stationary) = ``max(1e-6, 1e3 * tol)`` -
          accommodates ``|d g / d t| ~ 1e-7`` in Rodal-family metrics
          at small velocities; calibrated against v_s=0.5 bubbles.
        - ``tol_s`` (shape-function) = ``1e-6`` - strict floor for
          non-trivial spatial curvature.
        - ``tol_T`` (matter detection) = ``1e-8`` - vacuum/matter
          discriminator; several orders above curvature-chain noise.

    Returns
    -------
    ClassifiedMetric
        NamedTuple with ``bobrick_class``, ``stationary``,
        ``comoving_fluid``, ``shape_function_supported`` fields.

    Notes
    -----
    Dispatch (see module-level docstring for full reasoning):

    - Class I: vacuum (no matter anywhere on the probe ladder).
    - Class II: has_matter + smooth bubble profile (matter present at
      multiple radii; no inner-vacuum shell signature).
    - Class III: has_matter + shell-structured (matter bounded within
      R_1..R_2; inner-vacuum signature present).
    - otherwise: 0 (unclassified).

    Because the class decision branches on Python booleans, this
    function is NOT JIT-traceable as-is; deterministic under repeated
    calls at the same probe ladder.
    """
    tol_t = max(1e-6, 1e3 * tol)  # stationary threshold
    tol_s = 1e-6  # shape-function threshold
    tol_T = 1e-8  # matter detection floor

    # Matter sweep: find peak radius (and whether any matter exists)
    peak_mag, peak_coords = _scan_matter(metric, tol_T)
    has_matter = peak_mag > tol_T

    # Time and spatial derivative probes - use the peak radius (if
    # matter is present) else the user-supplied ``probe_coords`` or a
    # default off-axis mid-range point.
    if probe_coords is None:
        diag_coords = peak_coords if has_matter else _probe_point(2.0)
    else:
        diag_coords = probe_coords

    # Test 1: stationary (|d g_{ab} / d t| < tol_t) at diag_coords
    jac = _metric_jacobian(metric, diag_coords)  # shape (4, 4, 4)
    dg_dt = jac[:, :, 0]
    stationary = bool(jnp.all(jnp.abs(dg_dt) < tol_t))

    # Test 2: shape-function supported (spatial derivatives non-trivial)
    dg_dspatial = jac[:, :, 1:]
    shape_function_supported = bool(jnp.any(jnp.abs(dg_dspatial) > tol_s))

    # Test 3: comoving fluid (positive rho at peak)
    rho_peak = _energy_density_mixed(metric, peak_coords)
    comoving_fluid = has_matter and bool(
        jnp.isfinite(rho_peak) & (rho_peak > 0.0)
    )

    # Shell structure test: matter vanishes deep inside the peak radius
    is_shell_structured = has_matter and _scan_inner_vacuum(
        metric, peak_mag, peak_coords
    )

    # Class dispatch
    if not has_matter:
        # Vacuum (Minkowski, Schwarzschild) - Class I by default. We
        # require stationary as well; otherwise return unclassified.
        bobrick_class = 1 if stationary else 0
    elif is_shell_structured:
        # Matter bounded within a shell - Class III (WarpShell).
        bobrick_class = 3
    elif shape_function_supported:
        # Bubble-like matter profile - Class II.
        bobrick_class = 2
    else:
        bobrick_class = 0

    return ClassifiedMetric(
        bobrick_class=int(bobrick_class),
        stationary=stationary,
        comoving_fluid=comoving_fluid,
        shape_function_supported=shape_function_supported,
    )
