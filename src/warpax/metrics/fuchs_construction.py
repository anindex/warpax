"""Fuchs et al. constant-velocity subluminal warp shell metric.

Implements the construction pipeline from Fuchs et al. (CQG 2024,
arXiv:2405.02709, Section 3) for the constant-velocity subluminal warp shell:

    1. Constant-density shell between R_1 and R_2 with total mass M.
    2. Solve the TOV equation for isotropic pressure P'(r), BC P'(R_2)=0.
    3. Apply iterative kernel smoothing to density and pressure
       with differential kernel widths (sigma_rho / sigma_P ~ 1.72).
    4. Recompute cumulative mass from smoothed density.
    5. Solve metric functions a(r) and b(r) from Carroll Eqs. 5.143/5.152.

The default factory uses the published moving-average kernel family, with
fixed physical span across resolutions. The sigmoid's natural-endpoint
extension, C2 radial spline and fixed-width interior/exterior joins are this
manuscript's regularization, not a literal reproduction of the buffered clamps.
The metric is stationary in the published comoving coordinates.
References
----------
Fuchs, Helmerich, Bobrick, Sellers, Melcher, Martire (2024).
    CQG 41, DOI: 10.1088/1361-6382/ad26aa.  arXiv: 2405.02709.
Carroll, S. M. (2004). Spacetime and Geometry. Eqs. 5.143, 5.152.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry.metric import ADMMetric
from ..geometry.transitions import smoothstep_c2
from ._tov_scan import integrate_tov_inward


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
    padded = jnp.concatenate(
        [
            jnp.flip(values[1 : k_radius + 1]),
            values,
            jnp.flip(values[-k_radius - 1 : -1]),
        ]
    )

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
    reproduction of the published kernel family. Fractional endpoint-cell
    weights hold its physical span fixed during resolution refinement. The span is matched
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
    half = int(jnp.ceil(0.5 * span / dr + 0.5))
    half = max(half, 1)
    half = min(half, n // 2)

    # Cell-integrated boxcar weights keep the physical span fixed as dr changes.
    offsets = jnp.arange(-half, half + 1, dtype=jnp.float64) * dr
    kernel = jnp.maximum(
        0.0,
        jnp.minimum(offsets + dr / 2, span / 2) - jnp.maximum(offsets - dr / 2, -span / 2),
    )
    kernel = kernel / jnp.sum(kernel)

    padded = jnp.concatenate(
        [
            jnp.flip(values[1 : half + 1]),
            values,
            jnp.flip(values[-half - 1 : -1]),
        ]
    )
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
    smooth_fn = _moving_average_smooth if kernel_type == "moving_average" else _gaussian_smooth
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
    # Descending grid for inward integration; r_rev[0] is the outer boundary.
    p_rev = integrate_tov_inward(jnp.flip(r_grid), jnp.flip(rho_grid), jnp.flip(m_grid))
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

    e^{2b} = 1 / (1 - 2m/r)          , Eq. 5.143
    da/dr = (m + 4pi r^3 P_tilde) / (r(r - 2m)) , Eq. 5.152

    with Schwarzschild boundary: e^{2a(r>>R_2)} = e^{-2b(r>>R_2)}.
    """
    r_safe = jnp.maximum(r_grid, 1e-30)
    compactness = 2.0 * m_tilde / r_safe
    compactness_safe = jnp.minimum(compactness, 1.0 - 1e-12)

    # b(r): e^{2b} = 1 / (1 - 2m/r)
    b_grid = -0.5 * jnp.log(1.0 - compactness_safe)

    # da/dr
    numer = m_tilde + 4.0 * jnp.pi * r_safe**3 * P_tilde
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
    forward_integral = jnp.concatenate(
        [
            jnp.array([0.0]),
            jnp.cumsum(0.5 * (da_dr[:-1] + da_dr[1:]) * dr),
        ]
    )

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
    r"""Manuscript regularization of Fuchs Eqs. (27)--(28), arXiv v1.

    On R_1 < r < R_2, S=sigmoid((R_2-R_1)(1/(r-R_2)+1/(r-R_1))).
    Extend S by 1 at/below R_1 and 0 at/above R_2. The exponential
    approach makes every derivative vanish at both natural endpoints.
    R_b is retained for constructor compatibility but no longer truncates S.
    """
    # Outside the open shell evaluate a harmless interior argument; the constant
    # branch supplies the exact endpoint extension without reciprocal poles.
    inside = (r > R_1) & (r < R_2)
    r_in = jnp.where(inside, r, 0.5 * (R_1 + R_2))
    arg = (R_2 - R_1) * (1.0 / (r_in - R_2) + 1.0 / (r_in - R_1))
    S = jax.nn.sigmoid(arg)
    return jnp.where(r <= R_1, 1.0, jnp.where(r >= R_2, 0.0, S))


class FuchsConstructionResult(NamedTuple):
    """Pre-solved radial grids from the Fuchs construction.

    Attributes
    ----------
    r_grid : radial grid points.
    a_grid : time potential a(r), g00 = -e^{2a(r)}.
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
    r_s_param: float = 6.668692,
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
        span = sigma*sqrt(12)) with the manuscript's fixed-span discretization.
    """
    from ..numerics import assert_uniform_grid

    M_total = r_s_param / 2.0
    shell_vol = R_2**3 - R_1**3
    rho_0 = 3.0 * M_total / (4.0 * jnp.pi * shell_vol)

    r_max = r_pad_factor * R_2
    r_grid = jnp.linspace(1e-6, r_max, n_grid)
    assert_uniform_grid(r_grid, name="fuchs_construction.r_grid")

    # Step 1: Constant density
    in_shell = (r_grid >= R_1) & (r_grid <= R_2)
    rho_initial = jnp.where(in_shell, rho_0, 0.0)

    # Cumulative mass from initial density (input to the TOV solve)
    dr = r_grid[1] - r_grid[0]
    integrand_m = 4.0 * jnp.pi * rho_initial * r_grid**2
    m_initial = jnp.concatenate(
        [
            jnp.array([0.0]),
            jnp.cumsum(0.5 * (integrand_m[:-1] + integrand_m[1:]) * dr),
        ]
    )

    # Step 2: TOV for initial isotropic pressure
    P_initial = _solve_tov_inward(rho_initial, m_initial, r_grid, R_1)

    # Step 3: Iterative Gaussian smoothing
    sigma_rho = sigma_rho_factor * (R_2 - R_1)
    sigma_P = sigma_rho / sigma_ratio

    rho_smoothed = _iterative_smooth(
        rho_initial, r_grid, sigma_rho, n_smooth, kernel_type=kernel_type
    )
    P_smoothed = _iterative_smooth(P_initial, r_grid, sigma_P, n_smooth, kernel_type=kernel_type)

    # Ensure non-negative after smoothing
    rho_smoothed = jnp.maximum(rho_smoothed, 0.0)
    P_smoothed = jnp.maximum(P_smoothed, 0.0)

    # Step 4: Recompute mass from smoothed density
    integrand_m_smooth = 4.0 * jnp.pi * rho_smoothed * r_grid**2
    m_smoothed = jnp.concatenate(
        [
            jnp.array([0.0]),
            jnp.cumsum(0.5 * (integrand_m_smooth[:-1] + integrand_m_smooth[1:]) * dr),
        ]
    )
    total_mass = float(m_smoothed[-1])

    # Step 5: Metric functions from smoothed profiles
    a_grid, b_grid = _compute_metric_functions(
        rho_smoothed,
        P_smoothed,
        m_smoothed,
        r_grid,
        R_2,
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

    The covariant metric has g00=-exp(2a), g01=g10=-v_s*S and g0y=g0z=0.
    All spatial components are those of the spherical shell. ADM lapse and
    contravariant shift are derived from this metric, not prescribed separately.
    v_s denotes the paper's beta_warp in its stationary comoving coordinates.

    Radial profiles use a C2 cubic spline. Quintic blends attach a constant
    interior on [R_1/4,R_1/2] and a Schwarzschild exterior on
    [(R_2+r_max)/2,r_max]. These fixed physical intervals are part of this
    manuscript's regularization and remain fixed during grid refinement.
    The finite, strictly increasing radial grid must begin in [0,R_1/4],
    so its lower clipping point lies within the constant interior.
    R_b remains accepted but does not clamp the sigmoid.
    """

    _r_grid: Float[Array, "N"]
    _a_grid: Float[Array, "N"]
    _b_grid: Float[Array, "N"]
    v_s: float
    R_1: float
    R_2: float
    R_b: float
    total_mass: float
    _a_slope: Float[Array, "N"] = eqx.field(init=False)
    _b_slope: Float[Array, "N"] = eqx.field(init=False)

    def __post_init__(self):
        from interpax import approx_df

        profiles = (self._r_grid, self._a_grid, self._b_grid)
        if not all(p.ndim == 1 and p.shape == self._r_grid.shape for p in profiles):
            raise ValueError("Fuchs requires matching one-dimensional radial profiles")
        if len(self._r_grid) < 2 or not all(bool(jnp.all(jnp.isfinite(p))) for p in profiles):
            raise ValueError(
                "Fuchs requires at least two finite radial knots and finite potentials"
            )
        if not bool(jnp.all(jnp.diff(self._r_grid) > 0)):
            raise ValueError("Fuchs radial knots must be strictly increasing")
        if not (0 < self.R_1 < self.R_2 < float(self._r_grid[-1])):
            raise ValueError("Fuchs requires 0 < R_1 < R_2 < r_max")
        if not (0 <= float(self._r_grid[0]) <= self.R_1 / 4):
            raise ValueError("Fuchs radial grid must begin in [0, R_1/4] for a C2 inner join")
        if not (0 <= self.total_mass < self.R_2 / 2):
            raise ValueError("Fuchs exterior requires 0 <= 2M < R_2")
        # Cache the spline solve; evaluating curvature only needs local coefficients.
        self._a_slope = approx_df(self._r_grid, self._a_grid, method="cubic2", axis=0)
        self._b_slope = approx_df(self._r_grid, self._b_grid, method="cubic2", axis=0)

    def _potentials(self, r: Float[Array, ""]) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """C2 radial potentials with constant interior and Schwarzschild exterior."""
        from interpax import interp1d

        r_edge = self._r_grid[-1]
        r_eval = jnp.clip(r, self._r_grid[0], r_edge)
        a_in = interp1d(r_eval, self._r_grid, self._a_grid, method="cubic2", fx=self._a_slope)
        b_in = interp1d(r_eval, self._r_grid, self._b_grid, method="cubic2", fx=self._b_slope)
        inner = smoothstep_c2((r - self.R_1 / 4) / (self.R_1 / 4))
        a_in = self._a_grid[0] + inner * (a_in - self._a_grid[0])
        b_in = inner * b_in
        outer = smoothstep_c2((r - (self.R_2 + r_edge) / 2) / ((r_edge - self.R_2) / 2))
        half_log = 0.5 * jnp.log1p(-2 * self.total_mass / jnp.maximum(r, self.R_2))
        return a_in + outer * (half_log - a_in), b_in + outer * (-half_log - b_in)

    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """alpha^2=exp(2a)+v_s^2*S^2*gamma^{xx}, preserving covariant g00."""
        r = jnp.sqrt(jnp.sum(coords[1:] ** 2) + 1e-60)
        a_val, _ = self._potentials(r)
        beta_low = -self.v_s * self.shape_function_value(coords)
        return jnp.sqrt(jnp.exp(2 * a_val) + beta_low * self.shift(coords)[0])

    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        """beta^i=gamma^{ix}*(-v_s*S), including its transverse components."""
        beta_low = jnp.array([-self.v_s * self.shape_function_value(coords), 0.0, 0.0])
        return jnp.linalg.solve(self.spatial_metric(coords), beta_low)

    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        """gamma_ij=delta_ij+(exp(2b)-1)n_i*n_j in stationary coordinates."""
        position = coords[1:]
        r = jnp.sqrt(jnp.sum(position**2) + 1e-60)
        _, b_val = self._potentials(r)
        n_hat = position / r
        return jnp.eye(3) + jnp.expm1(2 * b_val) * jnp.outer(n_hat, n_hat)

    def __call__(self, coords: Float[Array, "4"]) -> Float[Array, "4 4"]:
        """Published covariant-component modification, with regularized profiles."""
        r = jnp.sqrt(jnp.sum(coords[1:] ** 2) + 1e-60)
        a_val, _ = self._potentials(r)
        beta_low = -self.v_s * self.shape_function_value(coords)
        g = jnp.zeros((4, 4))
        g = g.at[0, 0].set(-jnp.exp(2 * a_val))
        g = g.at[0, 1].set(beta_low).at[1, 0].set(beta_low)
        return g.at[1:, 1:].set(self.spatial_metric(coords))

    def shape_function_value(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Natural-endpoint sigmoid in the stationary comoving radial coordinate."""
        r = jnp.sqrt(jnp.sum(coords[1:] ** 2) + 1e-60)
        return _fuchs_shift_transition(r, self.R_1, self.R_2, self.R_b)

    def symbolic(self):
        """Full covariant tensor with abstract regularized radial potentials."""
        import sympy as sp

        from ..geometry.metric import SymbolicMetric

        t, x, y, z = sp.symbols("t x y z")
        a = sp.Function("a")
        b = sp.Function("b")
        beta = sp.Function("S_warp")
        v_s = sp.Symbol("v_s")
        x_rel = x
        r = sp.sqrt(x**2 + y**2 + z**2)

        gamma_rr = sp.exp(2 * b(r))
        delta = sp.eye(3)
        n = sp.Matrix([x_rel / r, y / r, z / r])
        nnT = n * n.T
        spatial_metric = delta + (gamma_rr - 1) * nnT

        beta_low = -v_s * beta(r)
        g = sp.Matrix.zeros(4, 4)
        g[0, 0] = -sp.exp(2 * a(r))
        g[0, 1] = beta_low
        g[1, 0] = beta_low
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
    r_s_param: float = 6.668692,
    n_grid: int = 2048,
    kernel_type: str = "moving_average",
) -> FuchsMetric:
    """Factory for the Fuchs metric with paper-matched parameters.

    Parameters match Section 4 of arXiv:2405.02709:
        v_s = 0.02 (beta_warp)
        R_1 = 10 (inner shell radius)
        R_2 = 20 (outer shell radius)
        r_s_param = 6.668692, the Schwarzschild radius 2GM/c^2 of the published
            M = 4.49e27 kg at R_1 = 10 m, R_2 = 20 m, so 2M/R_2 = 0.3334

    Smoothing defaults to the published boxcar (MATLAB ``smooth()``) with
    ``sigma_rho/sigma_P ~ 1.72``, applied 4 times, matching the paper's
    iterative construction procedure. Pass ``kernel_type="gaussian"`` for the
    spectrally cleaner variance-matched substitute.
    """
    construction = build_fuchs_construction(
        R_1=R_1,
        R_2=R_2,
        r_s_param=r_s_param,
        n_grid=n_grid,
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
