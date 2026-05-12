"""ADM mass and asymptotic falloff computations.

Implements the ADM mass surface integral:

    M_ADM = (1/16pi) oint_S (dj gamma_{ij} - di gamma_{jj}) n^i dA

evaluated on coordinate spheres at large radii using angular quadrature
and JAX autodiff for spatial derivatives.

Also provides Richardson extrapolation for convergence verification.
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from warpax.geometry.metric import MetricSpecification


def _spatial_metric_derivs(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "4"],
) -> Float[Array, "3 3 3"]:
    """Compute dk gamma_{ij} at a single point via autodiff.

    Returns
    -------
    dgamma[i, j, k] = d_ gamma_{ij} / d_ x^k
    """
    t = coords[0]
    spatial = coords[1:]

    def spatial_metric(xyz: Float[Array, "3"]) -> Float[Array, "3 3"]:
        full = jnp.concatenate([jnp.array([t], dtype=coords.dtype), xyz])
        g = metric_fn(full)
        return g[1:, 1:]

    return jax.jacfwd(spatial_metric)(spatial)


def adm_mass(
    metric: MetricSpecification,
    r_surface: float = 100.0,
    n_theta: int = 16,
    n_phi: int = 32,
) -> Float[Array, ""]:
    """Compute ADM mass via surface integral on a coordinate sphere.

    M_ADM = (1/16pi) oint_S (dj gamma_{ij} - di tr(gamma)) n^i dA

    Uses Gauss-Legendre quadrature in theta and uniform quadrature in phi.

    Parameters
    ----------
    metric : MetricSpecification
    r_surface : radius of evaluation sphere (should be large)
    n_theta : number of theta quadrature points
    n_phi : number of phi quadrature points

    Returns
    -------
    M_ADM : ADM mass scalar
    """
    r = jnp.float64(r_surface)

    # Gauss-Legendre quadrature for theta (mapped to [0, pi])
    # Use numpy for quadrature nodes (compile-time constant)
    import numpy as np
    cos_nodes, weights_theta = np.polynomial.legendre.leggauss(n_theta)
    cos_nodes = jnp.array(cos_nodes, dtype=jnp.float64)
    weights_theta = jnp.array(weights_theta, dtype=jnp.float64)
    theta_nodes = jnp.arccos(cos_nodes)  # theta in [0, pi]

    # Uniform quadrature for phi
    phi_nodes = jnp.linspace(0, 2 * jnp.pi, n_phi, endpoint=False, dtype=jnp.float64)
    dphi = 2 * jnp.pi / n_phi

    def integrand_at_angle(theta: Float[Array, ""], phi: Float[Array, ""]) -> Float[Array, ""]:
        """ADM integrand at a single angular point on the sphere."""
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)
        sin_phi = jnp.sin(phi)
        cos_phi = jnp.cos(phi)

        # Cartesian coords on sphere
        x = r * sin_theta * cos_phi
        y = r * sin_theta * sin_phi
        z = r * cos_theta
        coords = jnp.array([0.0, x, y, z], dtype=jnp.float64)

        # Unit outward normal in Cartesian
        n_hat = jnp.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])

        # Spatial metric derivatives
        dgamma = _spatial_metric_derivs(metric, coords)

        # ADM integrand: (dj gamma_{ij} - di tr(gamma)) n^i
        # dj gamma_{ij} = dgamma[i, j, j] summed over j
        div_gamma = jnp.einsum("ijj->i", dgamma)  # dj gamma_{ij}

        # di tr(gamma) = di gamma_{jj} = dgamma[j, j, i] summed over j
        grad_trace = jnp.einsum("jji->i", dgamma)  # di gamma_{jj}

        # Contract with normal
        f = jnp.dot(div_gamma - grad_trace, n_hat)

        # dA = r^2 sin(theta) (the sin(theta) is handled by the Legendre weight change of variable)
        return f * r**2

    # Double quadrature
    total = jnp.float64(0.0)
    # Vectorize over phi for each theta node
    for i_theta in range(n_theta):
        theta_val = theta_nodes[i_theta]
        w_theta = weights_theta[i_theta]

        # For Gauss-Legendre on [-1,1] mapped to [0,pi]:
        # int__0^pi f(theta) sin(theta) dtheta = int__{-1}^{1} f(arccos(u)) du
        # The sin(theta) factor is absorbed into the change of variable.

        def phi_integrand(phi_val):
            return integrand_at_angle(theta_val, phi_val)

        phi_values = jax.vmap(phi_integrand)(phi_nodes)
        phi_sum = jnp.sum(phi_values) * dphi
        total = total + w_theta * phi_sum

    M_ADM = total / (16.0 * jnp.pi)
    return M_ADM


def adm_mass_richardson(
    metric: MetricSpecification,
    radii: list[float] | None = None,
    n_theta: int = 16,
    n_phi: int = 32,
) -> dict[str, object]:
    """ADM mass with Richardson extrapolation for convergence verification.

    Computes M_ADM at multiple radii and extrapolates to r -> inf.

    Parameters
    ----------
    metric : MetricSpecification
    radii : list of evaluation radii (default: [50, 100, 200, 400])
    n_theta, n_phi : angular quadrature points

    Returns
    -------
    dict with 'M_values' (mass at each radius), 'radii', 'M_extrapolated',
    'convergence_order'
    """
    if radii is None:
        radii = [50.0, 100.0, 200.0, 400.0]

    M_values = []
    for r in radii:
        M = adm_mass(metric, r_surface=r, n_theta=n_theta, n_phi=n_phi)
        M_values.append(float(M))

    # Richardson extrapolation: M(r) ~ M_inf + C/r^n
    # Use last two points for simple estimate
    if len(radii) >= 2:
        r1, r2 = radii[-2], radii[-1]
        M1, M2 = M_values[-2], M_values[-1]
        # Assuming O(1/r) correction: M(r) = M_inf + C/r
        # M1 = M_inf + C/r1, M2 = M_inf + C/r2
        # M_inf = (M2 r2 - M1 r1) / (r2 - r1)
        M_extrap = (M2 * r2 - M1 * r1) / (r2 - r1)

        # Convergence order from first 3 points
        if len(radii) >= 3:
            M0 = M_values[-3]
            # n = log((M0-M1)/(M1-M2)) / log(r1/r0) approximately
            dM01 = abs(M0 - M1)
            dM12 = abs(M1 - M2)
            if dM12 > 1e-15 and dM01 > 1e-15:
                conv_order = float(jnp.log(dM01 / dM12) / jnp.log(r2 / r1))
            else:
                conv_order = float("inf")  # Already converged
        else:
            conv_order = float("nan")
    else:
        M_extrap = M_values[-1]
        conv_order = float("nan")

    return {
        "M_values": M_values,
        "radii": radii,
        "M_extrapolated": M_extrap,
        "convergence_order": conv_order,
    }


def falloff_check(
    metric: MetricSpecification,
    r_test: float = 100.0,
    expected_order: int = 1,
    tol: float = 1e-2,
) -> dict[str, bool]:
    """Check asymptotic falloff of metric components.

    For asymptotically flat metrics, g_{munu} - eta_{munu} ~ O(1/r^n).
    Tests by comparing deviations at r and r/2.

    Parameters
    ----------
    metric : MetricSpecification
    r_test : radius to evaluate falloff
    expected_order : expected power-law decay (1 for standard asymptotic flatness)
    tol : tolerance for falloff verification

    Returns
    -------
    dict mapping component names to bool (True if falloff is acceptable)
    """
    coords_far = jnp.array([0.0, r_test, 0.0, 0.0], dtype=jnp.float64)
    coords_near = jnp.array([0.0, r_test / 2.0, 0.0, 0.0], dtype=jnp.float64)

    g_far = metric(coords_far)
    g_near = metric(coords_near)
    eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0], dtype=jnp.float64))

    deviation_far = jnp.abs(g_far - eta)
    deviation_near = jnp.abs(g_near - eta)

    eps = 1e-30
    ratio = (deviation_far + eps) / (deviation_near + eps)
    measured_order = -jnp.log(ratio) / jnp.log(2.0)

    component_names = ["g_tt", "g_xx", "g_yy", "g_zz"]
    results = {}
    for i, name in enumerate(component_names):
        dev = deviation_far[i, i]
        is_flat = dev <= 1e-15
        results[name] = bool(is_flat or (measured_order[i, i] >= expected_order - tol))

    return results


def asymptotic_flatness_report(
    metric: MetricSpecification,
    radii: list[float] | None = None,
    expected_order: int = 1,
    tol: float = 0.1,
) -> dict[str, object]:
    """Comprehensive asymptotic flatness diagnostic.

    Evaluates metric deviations from Minkowski at multiple radii,
    fits power-law falloff exponents for each component (including
    off-diagonal shift terms), and assesses overall asymptotic flatness.

    For a Schwarzschild-like metric, diagonal deviations should fall off
    as 1/r. For warp metrics with compact-support shift, g_{0i} should
    fall to zero at finite radius (faster than any power law).

    Parameters
    ----------
    metric : MetricSpecification
    radii : list of evaluation radii (default: [50, 100, 200, 400])
    expected_order : expected diagonal falloff power (default: 1 for 1/r)
    tol : tolerance on measured exponent (default: 0.1)

    Returns
    -------
    dict with keys:
        'diagonal' : dict per component with 'passed', 'measured_order',
                      'deviations'
        'shift' : dict per shift component with 'passed', 'deviations'
        'is_asymptotically_flat' : bool (all components pass)
        'radii' : list of radii used
    """
    if radii is None:
        radii = [50.0, 100.0, 200.0, 400.0]

    eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0], dtype=jnp.float64))

    # Evaluate metric at each radius along x-axis
    g_at_r = []
    for r in radii:
        coords = jnp.array([0.0, r, 0.0, 0.0], dtype=jnp.float64)
        g_at_r.append(metric(coords))

    # Diagonal falloff analysis
    diagonal_names = ["g_tt", "g_xx", "g_yy", "g_zz"]
    diagonal_indices = [(0, 0), (1, 1), (2, 2), (3, 3)]
    diagonal_results = {}

    for name, (i, j) in zip(diagonal_names, diagonal_indices):
        deviations = [float(jnp.abs(g[i, j] - eta[i, j])) for g in g_at_r]

        # Fit power-law: deviation ~ C / r^n
        # Use last two points: n = log(dev1/dev2) / log(r2/r1)
        if len(radii) >= 2 and deviations[-2] > 1e-15 and deviations[-1] > 1e-15:
            measured = float(
                jnp.log(deviations[-2] / deviations[-1])
                / jnp.log(radii[-1] / radii[-2])
            )
        elif all(d < 1e-15 for d in deviations):
            measured = float("inf")  # Exactly flat
        else:
            measured = 0.0

        is_flat = all(d < 1e-12 for d in deviations)
        passed = is_flat or (measured >= expected_order - tol)

        diagonal_results[name] = {
            "passed": bool(passed),
            "measured_order": measured,
            "deviations": deviations,
        }

    # Off-diagonal shift falloff (g_{0i})
    shift_names = ["g_tx", "g_ty", "g_tz"]
    shift_indices = [(0, 1), (0, 2), (0, 3)]
    shift_results = {}

    for name, (i, j) in zip(shift_names, shift_indices):
        deviations = [float(jnp.abs(g[i, j])) for g in g_at_r]
        # Shift should be zero (compact support) at all test radii
        all_zero = all(d < 1e-10 for d in deviations)
        shift_results[name] = {
            "passed": bool(all_zero),
            "deviations": deviations,
        }

    # Overall assessment
    all_diag_pass = all(r["passed"] for r in diagonal_results.values())
    all_shift_pass = all(r["passed"] for r in shift_results.values())

    return {
        "diagonal": diagonal_results,
        "shift": shift_results,
        "is_asymptotically_flat": all_diag_pass and all_shift_pass,
        "radii": radii,
    }

