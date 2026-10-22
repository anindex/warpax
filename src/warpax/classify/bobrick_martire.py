"""Bobrick-Martire taxonomy classifier for warp drive metrics.

Citation: Bobrick & Martire 2021, *Classical and Quantum Gravity* 38,
105009. arXiv:2102.06824.

Class taxonomy (per §3):

- **Class I**, vacuum or trivial-fluid metric with a timelike
  Killing vector (Minkowski, Schwarzschild).
- **Class II**, Alcubierre-family bubble metrics whose stress-energy
  is supported on a smooth bubble wall (Alcubierre, Rodal, Natario,
  Van den Broeck, Lentz).
- **Class III**, matter-shell / junction-structured metrics with
  stress-energy localized inside a finite shell ``R_1..R_2``
  (WarpShell).

Dispatch is the multi-probe cascade ``has_matter ->
is_shell_structured -> shape_function_supported``, with ``stationary``
selecting Class I in the no-matter branch; ``comoving_fluid`` is
reported as a separate probe verdict and does not affect the class.
Class III: has_matter and is_shell_structured;
Class II: has_matter and not is_shell_structured and
shape_function_supported; Class I: no matter or pure-vacuum stationary
metric; everything else -> 0 (unclassified).

The default radial probe ladder spans about three decades, covering
Alcubierre (R=1), Rodal (R=100), WarpShell (R_1=10, R_2=20) and
Schwarzschild at a safe range outside the horizon, and sits off-axis to
avoid ``y = z = 0`` degeneracies.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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


_DEFAULT_RADIAL_PROBES: tuple[float, ...] = (
    0.3,
    0.6,
    1.0,
    1.5,
    2.5,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    50.0,
    75.0,
    100.0,
    120.0,
    150.0,
)


def _probe_point(r: float | Float[Array, ""]) -> Float[Array, "4"]:
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


@eqx.filter_jit
def _probe_densities(metric: MetricSpecification) -> tuple[Float[Array, "P 4"], Float[Array, "P"]]:
    """Probe coordinates and ``rho`` at every radius of the default ladder."""
    coords = jax.vmap(_probe_point)(jnp.asarray(_DEFAULT_RADIAL_PROBES))
    return coords, jax.vmap(lambda c: _energy_density_mixed(metric, c))(coords)


def _scan_matter(
    metric: MetricSpecification,
) -> tuple[float, Float[Array, "4"], float]:
    """Sweep radial probes; return (peak_rho_magnitude, peak_coords, peak_rho).

    The peak is the first strict maximum of ``|rho|`` over finite probes;
    with no finite nonzero probe it is ``(0.0, probe(1.0), 0.0)``.
    """
    coords, rho = _probe_densities(metric)
    rho = np.asarray(rho)
    mags = np.where(np.isfinite(rho), np.abs(rho), 0.0)
    idx = int(np.argmax(mags))
    if mags[idx] <= 0.0:
        return 0.0, _probe_point(1.0), 0.0
    return float(mags[idx]), coords[idx], float(rho[idx])


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
    a few decades of the peak - ratio > 1e-4 typically. Returns False when
    ``r_inner < 0.05``: the peak is too close to the origin for the shell
    test to be reliable.
    """
    peak_r = float(jnp.sqrt(jnp.sum(peak_coords[1:] ** 2)))
    inner_r = 0.2 * peak_r
    if inner_r < 0.05:
        return False
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
    tol_t = max(1e-6, 1e3 * tol)
    tol_s = 1e-6
    tol_T = 1e-8

    peak_mag, peak_coords, rho_peak = _scan_matter(metric)
    has_matter = peak_mag > tol_T

    if probe_coords is None:
        diag_coords = peak_coords if has_matter else _probe_point(2.0)
    else:
        diag_coords = probe_coords

    jac = _metric_jacobian(metric, diag_coords)
    dg_dt = jac[:, :, 0]
    stationary = bool(jnp.all(jnp.abs(dg_dt) < tol_t))

    dg_dspatial = jac[:, :, 1:]
    shape_function_supported = bool(jnp.any(jnp.abs(dg_dspatial) > tol_s))

    comoving_fluid = has_matter and rho_peak > 0.0

    is_shell_structured = has_matter and _scan_inner_vacuum(metric, peak_mag, peak_coords)

    if not has_matter:
        bobrick_class = 1 if stationary else 0
    elif is_shell_structured:
        bobrick_class = 3
    elif shape_function_supported:
        bobrick_class = 2
    else:
        bobrick_class = 0

    return ClassifiedMetric(
        bobrick_class=int(bobrick_class),
        stationary=stationary,
        comoving_fluid=comoving_fluid,
        shape_function_supported=shape_function_supported,
    )
