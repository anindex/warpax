"""S-shell (Class I) source-first warp shell metric.

Static, spherically symmetric, flow-orthogonal shell where the spacetime
geometry is derived from matter source profiles via the Einstein constraint
equations. The line element:

    ds^2 = -e^{2Phi(r)} dt^2 + 2 beta_i dx^i dt
           + (delta_{ij} + (e^{2Lambda(r)} - 1) n_i n_j) dx^i dx^j

where Phi(r) and Lambda(r) are solved from rho(r) and p_r(r) via the
Hamiltonian constraint and TOV/lapse ODE. An optional shift beta^x
produces a nonzero transport observable.

Regions:
    r < R_1:  flat spatial metric, constant redshifted lapse, uniform shift
    R_1..R_2: constraint-derived curved region
    r > R_2:  Schwarzschild exterior (+ shift decay if v_s > 0)
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..constraints.constraint_solver import SShellPotentials, solve_sshell_potentials
from ..geometry.metric import ADMMetric
from ..geometry.transitions import smoothstep_c2
from .sshell_profiles import SShellSourceProfiles, constant_density_profiles


def _warp_transition_c2(
    r: Float[Array, "..."],
    R_inner: float,
    R_outer: float,
) -> Float[Array, "..."]:
    """C2-smooth transition: 1 for r <= R_inner, 0 for r >= R_outer."""
    t = jnp.clip((r - R_inner) / (R_outer - R_inner), 0.0, 1.0)
    return 1.0 - smoothstep_c2(t)


class SShellMetric(ADMMetric):
    """Source-first S-shell metric via ADM 3+1 decomposition.

    Stores pre-solved metric potential grids as pytree leaves. The
    constraint solver runs at construction time (via factory functions);
    per-evaluation cost is a cubic interpolation lookup.

    Parameters
    ----------
    _r_grid, _Phi_grid, _Lambda_grid, _m_grid : array
        Pre-solved potential grids from ``solve_sshell_potentials``.
    v_s : float
        Shift magnitude (default: 0.0).
    R_1, R_2 : float
        Inner/outer shell radii.
    smooth_width : float
        Transition width for the shift profile.
    total_mass : float
        Total shell mass M = m(R_2).
    """

    _r_grid: Float[Array, "N"]
    _Phi_grid: Float[Array, "N"]
    _Lambda_grid: Float[Array, "N"]
    _m_grid: Float[Array, "N"]

    v_s: float
    R_1: float
    R_2: float
    smooth_width: float
    total_mass: float

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

    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Lapse alpha = e^{Phi(r)} from constraint solver."""
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = jnp.sqrt(x_rel**2 + y**2 + z**2 + 1e-60)
        return jnp.maximum(jnp.exp(self._Phi(r)), 1e-12)

    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        """Shift beta^x = -v_s * S_warp(r)."""
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = jnp.sqrt(x_rel**2 + y**2 + z**2 + 1e-60)
        S_warp = _warp_transition_c2(r, self.R_1, self.R_2)
        return jnp.array([-S_warp * self.v_s, 0.0, 0.0])

    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        """Spatial metric: delta_{ij} + (e^{2Lambda} - 1) n_i n_j."""
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = jnp.sqrt(x_rel**2 + y**2 + z**2 + 1e-60)

        gamma_rr = jnp.exp(2.0 * self._Lambda(r))
        x_vec = jnp.array([x_rel, y, z])
        n_hat = x_vec / r
        gamma = jnp.eye(3) + (gamma_rr - 1.0) * jnp.outer(n_hat, n_hat)
        return jnp.where(r < 1e-10, jnp.eye(3), gamma)

    def shape_function_value(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Warp transition function S_warp(r)."""
        t, x, y, z = coords
        x_rel = x - self.v_s * t
        r = jnp.sqrt(x_rel**2 + y**2 + z**2 + 1e-60)
        return _warp_transition_c2(r, self.R_1, self.R_2)

    def symbolic(self):
        """Symbolic placeholder (potentials are numerical, not analytic)."""
        import sympy as sp
        from ..geometry.metric import SymbolicMetric

        t, x, y, z = sp.symbols("t x y z")
        Phi = sp.Function("Phi")
        Lambda = sp.Function("Lambda")
        r = sp.sqrt(x**2 + y**2 + z**2)

        g = sp.Matrix([
            [-sp.exp(2 * Phi(r)), 0, 0, 0],
            [0, sp.exp(2 * Lambda(r)), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "SShell"


def sshell_from_potentials(
    potentials: SShellPotentials,
    R_1: float,
    R_2: float,
    v_s: float = 0.0,
    smooth_width: float | None = None,
) -> SShellMetric:
    """Construct an SShellMetric from pre-solved potentials.

    Parameters
    ----------
    potentials : SShellPotentials
    R_1 : inner shell radius.
    R_2 : outer shell radius.
    v_s : shift magnitude (default: 0.0).
    smooth_width : transition width (default: 0.05 * (R_2 - R_1)).
    """
    sw = smooth_width if smooth_width is not None else 0.05 * (R_2 - R_1)
    return SShellMetric(
        _r_grid=potentials.r_grid,
        _Phi_grid=potentials.Phi_grid,
        _Lambda_grid=potentials.Lambda_grid,
        _m_grid=potentials.m_grid,
        v_s=v_s,
        R_1=R_1,
        R_2=R_2,
        smooth_width=sw,
        total_mass=potentials.total_mass,
    )


def sshell_from_profiles(
    profiles: SShellSourceProfiles,
    v_s: float = 0.0,
    n_grid: int = 1024,
    smooth_width: float | None = None,
) -> SShellMetric:
    """Construct an SShellMetric from source profiles.

    Solves the constraint equations at construction time.

    Parameters
    ----------
    profiles : SShellSourceProfiles
    v_s : shift magnitude (default: 0.0).
    n_grid : grid resolution for constraint solver.
    smooth_width : transition width.
    """
    potentials = solve_sshell_potentials(
        rho=profiles.density,
        p_r=profiles.radial_pressure,
        R_1=profiles.R_1,
        R_2=profiles.R_2,
        n_grid=n_grid,
    )
    return sshell_from_potentials(
        potentials=potentials,
        R_1=profiles.R_1,
        R_2=profiles.R_2,
        v_s=v_s,
        smooth_width=smooth_width,
    )


def sshell_default(v_s: float = 0.0) -> SShellMetric:
    """Default S-shell: constant density, R_1=10, R_2=20, rho_0=1e-4.

    Parameters
    ----------
    v_s : shift magnitude (default: 0.0 for static,
        0.02 for transport-observable configuration).
    """
    profiles = constant_density_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4)
    return sshell_from_profiles(profiles, v_s=v_s)
