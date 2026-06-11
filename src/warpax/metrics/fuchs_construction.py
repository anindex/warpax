"""Fuchs et al. constant-velocity subluminal warp shell metric.

Implements the construction pipeline from Fuchs et al. (CQG 2024,
arXiv:2405.02709, Section 3) for the constant-velocity subluminal warp shell:

    1. Constant-density shell between R_1 and R_2 with total mass M.
    2. Solve the TOV equation for isotropic pressure P'(r), BC P'(R_2)=0.
    3. Apply iterative Gaussian-kernel smoothing to density and pressure
       with differential kernel widths (sigma_rho / sigma_P ~ 1.72).
    4. Recompute cumulative mass from smoothed density.
    5. Solve metric functions a(r) and b(r) from Carroll Eqs. 5.143/5.152.

The original paper uses MATLAB ``smooth()`` (a moving-average lowpass
filter). We substitute a Gaussian kernel convolution, which provides
equivalent boundary regularization without the moving average's spectral sidelobes.
The kernel width is matched as sigma_gauss = span_MA / sqrt(12).
See Weickert (1998) and Getreuer (2013) for the equivalence.

References
----------
Fuchs, Helmerich, Bobrick, Sellers, Melcher, Martire (2024).
    CQG 41, DOI: 10.1088/1361-6382/ad26aa.  arXiv: 2405.02709.
Carroll, S. M. (2004). Spacetime and Geometry. Eqs. 5.143, 5.152.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry.metric import ADMMetric
from ..geometry.transitions import smoothstep


def _gaussian_smooth(
    values: Float[Array, "N"],
    r_grid: Float[Array, "N"],
    sigma: float,
) -> Float[Array, "N"]:
    """Gaussian-kernel smoothing on a uniform radial grid.

    Convolves *values* with a normalized Gaussian kernel of width *sigma*.
    Boundary handling uses the ``reflect`` convention (mirror padding),
    which preserves the integral and avoids boundary artifacts.

    Substitutes the moving average used in [Fuchs2024] (see module
    docstring); kernel widths matched as:

        sigma_gauss = span_MA / sqrt(12)

    Parameters
    ----------
    values : profile to smooth.
    r_grid : uniform radial grid.
    sigma : Gaussian kernel width (standard deviation).
    """
    n = values.shape[0]
    dr = r_grid[1] - r_grid[0]
    # Kernel radius in grid points (truncate at 4 sigma)
    k_radius = int(jnp.ceil(4.0 * sigma / dr))
    k_radius = max(k_radius, 1)
    k_radius = min(k_radius, n // 2)

    offsets = jnp.arange(-k_radius, k_radius + 1, dtype=jnp.float64)
    kernel = jnp.exp(-0.5 * (offsets * dr / sigma) ** 2)
    kernel = kernel / jnp.sum(kernel)

    # Reflect-pad the signal
    padded = jnp.concatenate([
        jnp.flip(values[1:k_radius + 1]),
        values,
        jnp.flip(values[-k_radius - 1:-1]),
    ])

    result = jnp.convolve(padded, kernel, mode="valid")
    return result[:n]


def _moving_average_smooth(
    values: Float[Array, "N"],
    r_grid: Float[Array, "N"],
    sigma: float,
) -> Float[Array, "N"]:
    """Boxcar moving-average smoothing, matching the original MATLAB ``smooth()``.

    The original Fuchs construction uses MATLAB's ``smooth()`` (an unweighted
    moving average over a span of grid points). We expose it here for an
    exact-kernel reproduction of the published pipeline. The span is matched
    to the Gaussian width as

        span = sigma * sqrt(12),

    so the two kernels have the same second moment. Boundary handling uses the
    same reflect (mirror) padding as :func:`_gaussian_smooth`.

    Parameters
    ----------
    values : profile to smooth.
    r_grid : uniform radial grid.
    sigma : Gaussian-equivalent width; the boxcar span is ``sigma*sqrt(12)``.
    """
    n = values.shape[0]
    dr = r_grid[1] - r_grid[0]
    span = sigma * jnp.sqrt(12.0)
    half = int(jnp.ceil(0.5 * span / dr))
    half = max(half, 1)
    half = min(half, n // 2)

    window = 2 * half + 1
    kernel = jnp.ones(window, dtype=jnp.float64) / window

    padded = jnp.concatenate([
        jnp.flip(values[1:half + 1]),
        values,
        jnp.flip(values[-half - 1:-1]),
    ])
    result = jnp.convolve(padded, kernel, mode="valid")
    return result[:n]


def _iterative_smooth(
    values: Float[Array, "N"],
    r_grid: Float[Array, "N"],
    sigma: float,
    n_iter: int = 4,
    kernel_type: str = "gaussian",
) -> Float[Array, "N"]:
    """Apply smoothing iteratively (``n_iter`` passes; Fuchs Section 3.2).

    ``kernel_type`` selects the smoother: ``"gaussian"`` (default, the
    spectrally clean substitute) or ``"moving_average"`` (the original
    MATLAB ``smooth()`` boxcar, for exact-pipeline reproduction).
    """
    smooth_fn = (
        _moving_average_smooth if kernel_type == "moving_average"
        else _gaussian_smooth
    )
    result = values
    for _ in range(n_iter):
        result = smooth_fn(result, r_grid, sigma)
    return result


def _solve_tov_inward(
    rho_grid: Float[Array, "N"],
    m_grid: Float[Array, "N"],
    r_grid: Float[Array, "N"],
    R_1: float,
) -> Float[Array, "N"]:
    """Solve the TOV equation inward from the outer boundary.

    .. math::
        \\frac{dp_r}{dr} = -\\frac{(\\rho + p_r)(m + 4\\pi r^3 p_r)}{r(r - 2m)}

    BC: ``p_r = 0`` at the outermost grid point. Integrates inward with
    classical 4-stage Runge-Kutta on the uniform grid. Mid-step density
    and mass are obtained by linear interpolation between adjacent grid
    samples (consistent for trapezoidal-rule input integrals).

    Parameters
    ----------
    rho_grid : density values on the radial grid.
    m_grid : cumulative mass values.
    r_grid : radial grid (ascending order).
    R_1 : inner shell radius (p_r = 0 for r < R_1).
    """
    # Reverse grid for inward integration; r_rev[0] is the outer boundary.
    r_rev = jnp.flip(r_grid)
    rho_rev = jnp.flip(rho_grid)
    m_rev = jnp.flip(m_grid)
    h = r_rev[1] - r_rev[0]  # Negative (integrating inward)

    # Linearly interpolated midpoint samples (i + 1/2).
    rho_mid = 0.5 * (rho_rev[:-1] + rho_rev[1:])
    m_mid = 0.5 * (m_rev[:-1] + m_rev[1:])

    def tov_rhs(r, p, rho, m):
        r_safe = jnp.maximum(jnp.abs(r), 1e-30)
        denom = r_safe * (r_safe - 2.0 * m)
        denom_safe = jnp.where(
            jnp.abs(denom) < 1e-30,
            jnp.where(denom >= 0.0, 1e-30, -1e-30),
            denom,
        )
        numer = -(rho + p) * (m + 4.0 * jnp.pi * r_safe ** 3 * p)
        return numer / denom_safe

    def scan_step(p_current, inputs):
        r_a, rho_a, m_a, r_b, rho_b, m_b, rho_m, m_m = inputs
        r_mid = r_a + 0.5 * h
        k1 = tov_rhs(r_a,   p_current,                rho_a, m_a)
        k2 = tov_rhs(r_mid, p_current + 0.5 * h * k1, rho_m, m_m)
        k3 = tov_rhs(r_mid, p_current + 0.5 * h * k2, rho_m, m_m)
        k4 = tov_rhs(r_b,   p_current + h * k3,       rho_b, m_b)
        dp = (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        p_next = jnp.maximum(p_current + dp, 0.0)
        return p_next, p_next

    _, p_scan = jax.lax.scan(
        scan_step,
        jnp.float64(0.0),
        (r_rev[:-1], rho_rev[:-1], m_rev[:-1],
         r_rev[1:],  rho_rev[1:],  m_rev[1:],
         rho_mid,    m_mid),
    )

    p_rev = jnp.concatenate([jnp.array([0.0]), p_scan])
    p_grid = jnp.flip(p_rev)

    # Zero out pressure outside the shell
    p_grid = jnp.where(r_grid < R_1, 0.0, p_grid)
    return p_grid


def _compute_metric_functions(
    rho_tilde: Float[Array, "N"],
    P_tilde: Float[Array, "N"],
    m_tilde: Float[Array, "N"],
    r_grid: Float[Array, "N"],
    R_2: float,
) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
    """Compute metric potentials a(r) and b(r) per Carroll Eqs. 5.143/5.152.

    e^{2b} = 1 / (1 - 2m/r)           -- Eq. 5.143
    da/dr = (m + 4pi r^3 P_tilde) / (r(r - 2m))  -- Eq. 5.152

    with Schwarzschild boundary: e^{2a(r>>R_2)} = e^{-2b(r>>R_2)}.
    """
    r_safe = jnp.maximum(r_grid, 1e-30)
    compactness = 2.0 * m_tilde / r_safe
    compactness_safe = jnp.minimum(compactness, 1.0 - 1e-12)

    # b(r): e^{2b} = 1 / (1 - 2m/r)
    b_grid = -0.5 * jnp.log(1.0 - compactness_safe)

    # da/dr
    numer = m_tilde + 4.0 * jnp.pi * r_safe ** 3 * P_tilde
    denom = r_safe * (r_safe - 2.0 * m_tilde)
    denom_safe = jnp.where(
        jnp.abs(denom) < 1e-30,
        jnp.where(denom >= 0.0, 1e-30, -1e-30),
        denom,
    )
    da_dr = numer / denom_safe
    # Zero out in vacuum interior
    da_dr = jnp.where(r_grid < r_grid[0] * 0.5, 0.0, da_dr)

    # Integrate da/dr from the outer boundary inward
    dr = r_grid[1] - r_grid[0]
    forward_integral = jnp.concatenate([
        jnp.array([0.0]),
        jnp.cumsum(0.5 * (da_dr[:-1] + da_dr[1:]) * dr),
    ])

    # Schwarzschild boundary: a(r_max) = -b(r_max)
    total_mass = float(m_tilde[-1])
    a_boundary = 0.5 * jnp.log(jnp.maximum(1.0 - 2.0 * total_mass / r_grid[-1], 1e-30))
    a_grid = a_boundary - (forward_integral[-1] - forward_integral)

    return a_grid, b_grid


def _fuchs_shift_transition(
    r: Float[Array, "..."],
    R_1: float,
    R_2: float,
    R_b: float,
) -> Float[Array, "..."]:
    """Compact sigmoid S_warp(r) from Fuchs Eq. 31-32.

    S_warp transitions from 1 inside the shell to 0 outside, with
    buffer R_b ensuring derivatives stay interior to the bubble.

    Parameters
    ----------
    r : radial distance.
    R_1 : inner shell radius.
    R_2 : outer shell radius.
    R_b : buffer region width.
    """
    R_inner = R_1 + R_b
    R_outer = R_2 - R_b
    # Floor inner radius at 10% of the shell width below R_outer so the
    # denominator stays well-conditioned when 2*R_b approaches (R_2 - R_1).
    R_inner = jnp.minimum(R_inner, R_outer - 0.1 * jnp.maximum(R_2 - R_1, 1e-12))
    t = jnp.clip((r - R_inner) / jnp.maximum(R_outer - R_inner, 1e-12), 0.0, 1.0)
    return 1.0 - smoothstep(t, order=2)


class FuchsConstructionResult(NamedTuple):
    """Pre-solved radial grids from the Fuchs construction.

    Attributes
    ----------
    r_grid : radial grid points.
    a_grid : lapse potential a(r), alpha = e^{a(r)}.
    b_grid : spatial potential b(r), gamma_rr = e^{2b(r)}.
    m_grid : cumulative mass m(r).
    rho_smoothed : Gaussian-smoothed density profile.
    P_smoothed : Gaussian-smoothed isotropic pressure.
    total_mass : total shell mass.
    """
    r_grid: Float[Array, "N"]
    a_grid: Float[Array, "N"]
    b_grid: Float[Array, "N"]
    m_grid: Float[Array, "N"]
    rho_smoothed: Float[Array, "N"]
    P_smoothed: Float[Array, "N"]
    total_mass: float


def build_fuchs_construction(
    R_1: float = 10.0,
    R_2: float = 20.0,
    r_s_param: float = 5.0,
    n_grid: int = 2048,
    sigma_rho_factor: float = 0.06,
    sigma_ratio: float = 1.72,
    n_smooth: int = 4,
    r_pad_factor: float = 1.5,
    kernel_type: str = "gaussian",
) -> FuchsConstructionResult:
    """Build the Fuchs shell via iterative smoothing.

    Parameters
    ----------
    R_1, R_2 : inner/outer shell radii.
    r_s_param : Schwarzschild radius parameter (2M in geometric units).
    n_grid : radial grid resolution (higher = better TOV fidelity).
    sigma_rho_factor : Gaussian kernel width for density as a fraction
        of (R_2 - R_1). Matched to the Fuchs paper's moving-average
        span via sigma = span / sqrt(12).
    sigma_ratio : ratio s_rho / s_P ~ 1.72 from Fuchs Section 3.2.
    n_smooth : number of smoothing iterations (4 in the paper).
    r_pad_factor : extend grid to r_pad_factor * R_2.
    kernel_type : ``"gaussian"`` (default) or ``"moving_average"`` (the
        original MATLAB ``smooth()`` boxcar, variance-matched via
        span = sigma*sqrt(12)) for exact-pipeline reproduction.
    """
    from ..numerics import assert_uniform_grid

    M_total = r_s_param / 2.0
    shell_vol = R_2 ** 3 - R_1 ** 3
    rho_0 = 3.0 * M_total / (4.0 * jnp.pi * shell_vol)

    r_max = r_pad_factor * R_2
    r_grid = jnp.linspace(1e-6, r_max, n_grid)
    assert_uniform_grid(r_grid, name="fuchs_construction.r_grid")

    # Step 1: Constant density
    in_shell = (r_grid >= R_1) & (r_grid <= R_2)
    rho_initial = jnp.where(in_shell, rho_0, 0.0)

    # Step 2: Cumulative mass from initial density
    dr = r_grid[1] - r_grid[0]
    integrand_m = 4.0 * jnp.pi * rho_initial * r_grid ** 2
    m_initial = jnp.concatenate([
        jnp.array([0.0]),
        jnp.cumsum(0.5 * (integrand_m[:-1] + integrand_m[1:]) * dr),
    ])

    # Step 2: TOV for initial isotropic pressure
    P_initial = _solve_tov_inward(rho_initial, m_initial, r_grid, R_1)

    # Step 3: Iterative Gaussian smoothing
    sigma_rho = sigma_rho_factor * (R_2 - R_1)
    sigma_P = sigma_rho / sigma_ratio

    rho_smoothed = _iterative_smooth(
        rho_initial, r_grid, sigma_rho, n_smooth, kernel_type=kernel_type)
    P_smoothed = _iterative_smooth(
        P_initial, r_grid, sigma_P, n_smooth, kernel_type=kernel_type)

    # Ensure non-negative after smoothing
    rho_smoothed = jnp.maximum(rho_smoothed, 0.0)
    P_smoothed = jnp.maximum(P_smoothed, 0.0)

    # Step 4: Recompute mass from smoothed density
    integrand_m_smooth = 4.0 * jnp.pi * rho_smoothed * r_grid ** 2
    m_smoothed = jnp.concatenate([
        jnp.array([0.0]),
        jnp.cumsum(0.5 * (integrand_m_smooth[:-1] + integrand_m_smooth[1:]) * dr),
    ])
    total_mass = float(m_smoothed[-1])

    # Step 5: Metric functions from smoothed profiles
    a_grid, b_grid = _compute_metric_functions(
        rho_smoothed, P_smoothed, m_smoothed, r_grid, R_2,
    )

    return FuchsConstructionResult(
        r_grid=r_grid,
        a_grid=a_grid,
        b_grid=b_grid,
        m_grid=m_smoothed,
        rho_smoothed=rho_smoothed,
        P_smoothed=P_smoothed,
        total_mass=total_mass,
    )


class FuchsMetric(ADMMetric):
    """Fuchs warp shell metric with iteratively-smoothed profiles.

    Uses pre-solved radial grids from ``build_fuchs_construction``, with
    cubic interpolation at evaluation time (same approach as ``TShellMetric``).

    Parameters
    ----------
    _r_grid, _a_grid, _b_grid : pre-solved radial grids.
    v_s : shift magnitude (beta_warp from Fuchs Eq. 30).
    R_1, R_2, R_b : shell radii and buffer zone.
    total_mass : total shell mass.
    """
    _r_grid: Float[Array, "N"]
    _a_grid: Float[Array, "N"]
    _b_grid: Float[Array, "N"]

    v_s: float
    R_1: float
    R_2: float
    R_b: float
    total_mass: float

    def _interp(
        self, r: Float[Array, ""], grid_vals: Float[Array, "N"],
    ) -> Float[Array, ""]:
        """Cubic interpolation on the stored grid."""
        import interpax
        r_clamped = jnp.clip(r, self._r_grid[0], self._r_grid[-1])
        return interpax.interp1d(r_clamped, self._r_grid, grid_vals, method="cubic")

    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Lapse alpha = e^{a(r)}, smoothly interpolated from grid."""
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = jnp.sqrt(x_rel ** 2 + y ** 2 + z ** 2 + 1e-60)
        a_val = self._interp(r, self._a_grid)
        return jnp.maximum(jnp.exp(a_val), 1e-12)

    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        """Shift beta^x = -S_warp(r) * v_s."""
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = jnp.sqrt(x_rel ** 2 + y ** 2 + z ** 2 + 1e-60)
        S_warp = _fuchs_shift_transition(r, self.R_1, self.R_2, self.R_b)
        return jnp.array([-S_warp * self.v_s, 0.0, 0.0])

    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        """Spatial metric: delta_{ij} + (e^{2b} - 1) n_i n_j."""
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = jnp.sqrt(x_rel ** 2 + y ** 2 + z ** 2 + 1e-60)

        b_val = self._interp(r, self._b_grid)
        gamma_rr = jnp.exp(2.0 * b_val)

        x_vec = jnp.array([x_rel, y, z])
        n_hat = x_vec / r
        gamma = jnp.eye(3) + (gamma_rr - 1.0) * jnp.outer(n_hat, n_hat)
        return jnp.where(r < 1e-10, jnp.eye(3), gamma)

    def shape_function_value(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Warp transition function S_warp(r)."""
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = jnp.sqrt(x_rel ** 2 + y ** 2 + z ** 2 + 1e-60)
        return _fuchs_shift_transition(r, self.R_1, self.R_2, self.R_b)

    def symbolic(self):
        """Symbolic placeholder (profiles are numerical).

        Builds the spatial part as the full radial dyad
        ``gamma_ij = delta_ij + (exp(2 b(r)) - 1) n_i n_j`` so the
        symbolic and numerical forms agree off-axis.  Only the
        ``x``-axis entry is x-rel-dependent because we evaluate the
        outer product symbolically.
        """
        import sympy as sp
        from ..geometry.metric import SymbolicMetric

        t, x, y, z = sp.symbols("t x y z")
        a = sp.Function("a")
        b = sp.Function("b")
        beta = sp.Function("S_warp")
        v_s = sp.Symbol("v_s")
        x_rel = x - v_s * t
        r = sp.sqrt(x_rel ** 2 + y ** 2 + z ** 2)

        gamma_rr = sp.exp(2 * b(r))
        delta = sp.eye(3)
        n = sp.Matrix([x_rel / r, y / r, z / r])
        nnT = n * n.T
        spatial_metric = delta + (gamma_rr - 1) * nnT

        g = sp.Matrix.zeros(4, 4)
        g[0, 0] = -sp.exp(2 * a(r)) + (v_s * beta(r)) ** 2
        g[0, 1] = -v_s * beta(r)
        g[1, 0] = -v_s * beta(r)
        for i in range(3):
            for j in range(3):
                g[i + 1, j + 1] = spatial_metric[i, j]
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "Fuchs"


def fuchs_default(
    v_s: float = 0.02,
    R_1: float = 10.0,
    R_2: float = 20.0,
    R_b: float = 1.0,
    r_s_param: float = 5.0,
    n_grid: int = 2048,
    kernel_type: str = "gaussian",
) -> FuchsMetric:
    """Factory for the Fuchs metric with paper-matched parameters.

    Parameters match Section 4 of arXiv:2405.02709:
        v_s = 0.02 (beta_warp)
        R_1 = 10 (inner shell radius)
        R_2 = 20 (outer shell radius)
        r_s_param = 5.0 (Schwarzschild radius parameter)

    Uses Gaussian-kernel smoothing (Section 3) with sigma_rho/sigma_P ~ 1.72,
    applied 4 times, matching the paper's iterative construction procedure.
    """
    construction = build_fuchs_construction(
        R_1=R_1, R_2=R_2, r_s_param=r_s_param, n_grid=n_grid,
        kernel_type=kernel_type,
    )

    return FuchsMetric(
        _r_grid=construction.r_grid,
        _a_grid=construction.a_grid,
        _b_grid=construction.b_grid,
        v_s=v_s,
        R_1=R_1,
        R_2=R_2,
        R_b=R_b,
        total_mass=construction.total_mass,
    )
