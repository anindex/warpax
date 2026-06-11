"""Bobrick-Martire taxonomy classifier for warp drive metrics.

Citation: Bobrick & Martire 2021, *Classical and Quantum Gravity* 38,
105009. arXiv:2102.06824.

Class taxonomy (per §3):

- **Class I** -- vacuum or trivial-fluid metric with a timelike
  Killing vector (Minkowski, Schwarzschild).
- **Class II** -- Alcubierre-family bubble metrics whose stress-energy
  is supported on a smooth bubble wall (Alcubierre, Rodal, Natario,
  Van den Broeck, Lentz).
- **Class III** -- matter-shell / junction-structured metrics with
  stress-energy localized inside a finite shell ``R_1..R_2``
  (WarpShell).

Dispatch is the multi-probe cascade ``stationary ->
shape_function_supported -> has_matter -> comoving_fluid ->
is_shell_structured``. Class III: has_matter and is_shell_structured;
Class II: has_matter and not is_shell_structured and
shape_function_supported; Class I: no matter or pure-vacuum stationary
metric; everything else -> 0 (unclassified).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geometry.geometry import compute_curvature_chain
from ..geometry.metric import MetricSpecification


class ClassifiedMetric(NamedTuple):
    """Result of Bobrick-Martire classification.

    ``bobrick_class``: 1=I (vacuum/Killing), 2=II (Alcubierre-family bubble),
    3=III (matter shell), 0=unclassified. Booleans report the underlying
    probe verdicts (timelike Killing field, positive ``rho`` at the matter
    peak, non-trivial spatial variation).
    """

    bobrick_class: int
    stationary: bool
    comoving_fluid: bool
    shape_function_supported: bool


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
    """Compute the Eulerian energy density ``rho = -T^0{}_0`` at ``coords``.

    Raises the first index of ``T_{ab}`` via ``T^a_{\\;b} = g^{ac} T_{cb}``
    so the result is the mixed-tensor entry ``T^0{}_0`` even for
    non-symmetric ``T``.
    """
    curv = compute_curvature_chain(metric, coords)
    T_mixed = jnp.einsum("ac,cb->ab", curv.metric_inv, curv.stress_energy)
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
    Branches on Python booleans; not JIT-traceable. Deterministic for
    a fixed probe ladder.
    """
    tol_t = max(1e-6, 1e3 * tol)  # stationary threshold
    tol_s = 1e-6  # shape-function threshold
    tol_T = 1e-8  # matter detection floor

    peak_mag, peak_coords = _scan_matter(metric, tol_T)
    has_matter = peak_mag > tol_T

    # Probe at the matter peak (if any), else the supplied or default point.
    if probe_coords is None:
        diag_coords = peak_coords if has_matter else _probe_point(2.0)
    else:
        diag_coords = probe_coords

    jac = _metric_jacobian(metric, diag_coords)  # shape (4, 4, 4)
    dg_dt = jac[:, :, 0]
    stationary = bool(jnp.all(jnp.abs(dg_dt) < tol_t))

    dg_dspatial = jac[:, :, 1:]
    shape_function_supported = bool(jnp.any(jnp.abs(dg_dspatial) > tol_s))

    rho_peak = _energy_density_mixed(metric, peak_coords)
    comoving_fluid = has_matter and bool(
        jnp.isfinite(rho_peak) & (rho_peak > 0.0)
    )

    is_shell_structured = has_matter and _scan_inner_vacuum(
        metric, peak_mag, peak_coords
    )

    if not has_matter:
        # Vacuum (Minkowski, Schwarzschild): Class I when stationary,
        # otherwise unclassified.
        bobrick_class = 1 if stationary else 0
    elif is_shell_structured:
        bobrick_class = 3  # matter bounded within a shell (WarpShell)
    elif shape_function_supported:
        bobrick_class = 2  # bubble-like matter profile
    else:
        bobrick_class = 0

    return ClassifiedMetric(
        bobrick_class=int(bobrick_class),
        stationary=stationary,
        comoving_fluid=comoving_fluid,
        shape_function_supported=shape_function_supported,
    )
