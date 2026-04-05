"""asv benchmark - Hawking-Ellis classifier on 32^3 grid (#9).

Float64 JIT path; measures the warm-code execution of
``classify_hawking_ellis`` batched via ``jax.vmap`` over a 32^3 Alcubierre
stress-energy grid.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU canonical

import jax
import jax.numpy as jnp

import warpax  # noqa: F401
from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions import classify_hawking_ellis
from warpax.geometry import GridSpec, evaluate_curvature_grid


class ClassifierGrid32:
    """asv benchmark - classify_hawking_ellis on Alcubierre 32^3 grid (#9)."""

    warmup_time = 3.0
    number = 2
    repeat = (2, 3, 60.0)
    timeout = 120.0

    def setup(self) -> None:
        metric = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        grid = GridSpec(bounds=[(-6.0, 6.0)] * 3, shape=(32, 32, 32))
        chain = evaluate_curvature_grid(metric, grid, t=0.0, compute_invariants=False)
        # Flatten grid dims: (Nx, Ny, Nz, 4, 4) -> (Nx*Ny*Nz, 4, 4).
        T = chain.stress_energy
        g = chain.metric
        g_inv = chain.metric_inv
        # Compute mixed-index T^a_{\;b} = g^{ac} T_{cb}.
        T_mixed = jnp.einsum("...ac,...cb->...ab", g_inv, T)
        flat_shape = (-1, 4, 4)
        self.T_mixed_flat = T_mixed.reshape(flat_shape)
        self.g_flat = g.reshape(flat_shape)
        self.fn = jax.jit(jax.vmap(classify_hawking_ellis, in_axes=(0, 0)))
        _ = self.fn(self.T_mixed_flat, self.g_flat)  # JIT warmup

    def time_classify_grid(self) -> None:
        result = self.fn(self.T_mixed_flat, self.g_flat)
        result.he_type.block_until_ready()


class ClassifierGrid32Generalised:
    """asv benchmark - ``classify_hawking_ellis`` on Alcubierre 32^3 grid,
    ``solver='generalised'`` path.

    Notes
    -----
    Host-callback overhead: ~5-10x slower than
    :class:`ClassifierGrid32` due to
    ``jax.pure_callback(..., vmap_method='sequential')`` serialising the
    ``scipy.linalg.eig`` calls per grid point. This is documented in

    as a known and accepted cost of the LAPACK QZ binding
    (native Lorentzian pencil handling, no Cholesky-whiten). The standard
    path preserves the original behavior bit-exactly; this sibling class benchmarks the
    opt-in generalised path for regression-tracking the host-callback
    overhead across releases.
    """

    warmup_time = 3.0
    number = 2
    repeat = (2, 3, 60.0)
    timeout = 600.0  # ~5x ClassifierGrid32 budget - host-callback overhead

    def setup(self) -> None:
        metric = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        grid = GridSpec(bounds=[(-6.0, 6.0)] * 3, shape=(32, 32, 32))
        chain = evaluate_curvature_grid(metric, grid, t=0.0, compute_invariants=False)
        # Flatten grid dims: (Nx, Ny, Nz, 4, 4) -> (Nx*Ny*Nz, 4, 4).
        T = chain.stress_energy
        g = chain.metric
        g_inv = chain.metric_inv
        # Compute mixed-index T^a_{\;b} = g^{ac} T_{cb}; keep both T_ab and T_mixed
        # in scope - generalised branch needs T_ab for the pencil solver.
        T_mixed = jnp.einsum("...ac,...cb->...ab", g_inv, T)
        flat_shape = (-1, 4, 4)
        self.T_flat = T.reshape(flat_shape)
        self.T_mixed_flat = T_mixed.reshape(flat_shape)
        self.g_flat = g.reshape(flat_shape)

        # vmap the generalised path; pure_callback is sequential per vmap_method.
        def _classify_gen(T_mixed_i, g_i, T_ab_i):
            return classify_hawking_ellis(
                T_mixed_i, g_i, solver='generalised', T_ab=T_ab_i,
            )
        self.fn = jax.jit(jax.vmap(_classify_gen, in_axes=(0, 0, 0)))
        _ = self.fn(self.T_mixed_flat, self.g_flat, self.T_flat)  # JIT warmup

    def time_classify_grid_generalised(self) -> None:
        result = self.fn(self.T_mixed_flat, self.g_flat, self.T_flat)
        result.he_type.block_until_ready()
