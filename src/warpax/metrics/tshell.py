"""T-shell source-first warp shell metric with tilted matter flow.

Static spherically symmetric shell with matter 4-velocity tilted
relative to the hypersurface normal. The shift vector is derived from
the momentum constraint, not prescribed.

Frame: the shell is **static at the coordinate origin** (no
``v_s * t`` co-moving offset). All radial coordinates use
``r = sqrt(x^2 + y^2 + z^2)`` directly; this is intentional and
distinguishes the T-shell construction from Alcubierre/WarpShell where
the bubble center translates with the lab frame.

Line element:

    ds^2 = -alpha^2 dt^2 + 2 beta_i dx^i dt + gamma_{ij} dx^i dx^j

    alpha = e^{Phi(r)}                                     (TOV/lapse ODE)
    gamma_{ij} = delta_{ij} + (e^{2Lambda(r)} - 1) n_i n_j (Hamiltonian)
    beta^x = beta^x(r)                                     (momentum constraint)

Regions:
    r < R_1:  flat spatial, redshifted lapse, decaying shift
    R_1..R_2: curved shell with tilted flow
    r > R_2:  Schwarzschild exterior with decaying shift
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..constraints.tshell_solver import TShellPotentials, solve_tshell_potentials
from ..geometry.metric import ADMMetric
from .tshell_profiles import (
    TShellSourceProfiles,
    constant_velocity_profiles,
)


class TShellMetric(ADMMetric):
    """Source-first T-shell metric with constraint-derived shift.

    Pre-solved potential grids are stored as pytree leaves. The
    constraint solver runs at construction time; evaluation cost
    is cubic interpolation.

    Parameters
    ----------
    _r_grid, _Phi_grid, _Lambda_grid, _m_grid, _beta_x_grid : array
        Pre-solved potential grids.
    R_1, R_2 : float
        Inner/outer shell radii.
    total_mass : Float[Array, ""]
        Total shell mass M = m(R_2), stored as a jnp scalar array leaf
        (a Python float here would be partitioned as static by
        ``eqx.filter_jit`` and force a retrace per mass value).
    """

    _r_grid: Float[Array, "N"]
    _Phi_grid: Float[Array, "N"]
    _Lambda_grid: Float[Array, "N"]
    _m_grid: Float[Array, "N"]
    _beta_x_grid: Float[Array, "N"]

    R_1: float
    R_2: float
    total_mass: Float[Array, ""]

    def _interp(
        self, r: Float[Array, ""], grid_vals: Float[Array, "N"]
    ) -> Float[Array, ""]:
        """Cubic interpolation on the stored grid."""
        import interpax
        r_clamped = jnp.clip(r, self._r_grid[0], self._r_grid[-1])
        return interpax.interp1d(r_clamped, self._r_grid, grid_vals, method="cubic")

    def _Phi(self, r: Float[Array, ""]) -> Float[Array, ""]:
        return self._interp(r, self._Phi_grid)

    def _Lambda(self, r: Float[Array, ""]) -> Float[Array, ""]:
        return self._interp(r, self._Lambda_grid)

    def _beta_x(self, r: Float[Array, ""]) -> Float[Array, ""]:
        return self._interp(r, self._beta_x_grid)

    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Lapse alpha = e^{Phi(r)}."""
        t, x, y, z = coords
        r = jnp.sqrt(x**2 + y**2 + z**2 + 1e-60)
        return jnp.maximum(jnp.exp(self._Phi(r)), 1e-12)

    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        """Shift beta^i from momentum constraint."""
        t, x, y, z = coords
        r = jnp.sqrt(x**2 + y**2 + z**2 + 1e-60)
        return jnp.array([self._beta_x(r), 0.0, 0.0])

    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        """Spatial metric: delta_{ij} + (e^{2Lambda} - 1) n_i n_j."""
        t, x, y, z = coords
        r = jnp.sqrt(x**2 + y**2 + z**2 + 1e-60)

        gamma_rr = jnp.exp(2.0 * self._Lambda(r))
        x_vec = jnp.array([x, y, z])
        n_hat = x_vec / r
        gamma = jnp.eye(3) + (gamma_rr - 1.0) * jnp.outer(n_hat, n_hat)
        return jnp.where(r < 1e-10, jnp.eye(3), gamma)

    def shape_function_value(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Shell region indicator based on shift magnitude."""
        t, x, y, z = coords
        r = jnp.sqrt(x**2 + y**2 + z**2 + 1e-60)
        beta_x = self._beta_x(r)
        beta_max = jnp.max(jnp.abs(self._beta_x_grid))
        return jnp.where(
            beta_max > 1e-30,
            jnp.abs(beta_x) / beta_max,
            jnp.zeros_like(r),
        )

    def symbolic(self):
        """Symbolic placeholder (potentials are numerical)."""
        import sympy as sp
        from ..geometry.metric import SymbolicMetric

        t, x, y, z = sp.symbols("t x y z")
        Phi = sp.Function("Phi")
        Lambda = sp.Function("Lambda")
        beta = sp.Function("beta_x")
        r = sp.sqrt(x**2 + y**2 + z**2)

        g = sp.Matrix([
            [-sp.exp(2 * Phi(r)) + beta(r)**2, beta(r), 0, 0],
            [beta(r), sp.exp(2 * Lambda(r)), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "TShell"


def tshell_from_potentials(
    potentials: TShellPotentials,
    R_1: float,
    R_2: float,
) -> TShellMetric:
    """Construct a TShellMetric from pre-solved potentials."""
    return TShellMetric(
        _r_grid=potentials.r_grid,
        _Phi_grid=potentials.Phi_grid,
        _Lambda_grid=potentials.Lambda_grid,
        _m_grid=potentials.m_grid,
        _beta_x_grid=potentials.beta_x_grid,
        R_1=R_1,
        R_2=R_2,
        # jnp.asarray: keep total_mass an array pytree leaf (retrace fix);
        # the T-shell solver still reports a Python float.
        total_mass=jnp.asarray(potentials.total_mass),
    )


def tshell_from_profiles(
    profiles: TShellSourceProfiles,
    n_grid: int = 1024,
) -> TShellMetric:
    """Construct a TShellMetric from source profiles.

    Solves the constraint equations at construction time.
    """
    potentials = solve_tshell_potentials(
        rho=profiles.density,
        p_r=profiles.radial_pressure,
        v_x=profiles.velocity_x,
        R_1=profiles.R_1,
        R_2=profiles.R_2,
        n_grid=n_grid,
    )
    return tshell_from_potentials(
        potentials=potentials,
        R_1=profiles.R_1,
        R_2=profiles.R_2,
    )


def tshell_default(v_0: float = 0.1) -> TShellMetric:
    """Default T-shell: R_1=10, R_2=20, rho_0=1e-4, v_0=0.1."""
    profiles = constant_velocity_profiles(
        R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=v_0,
    )
    return tshell_from_profiles(profiles)
