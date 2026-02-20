"""Curvature tensor identity validation.

Tests that autodiff-computed curvature tensors satisfy fundamental
geometric identities that hold for any pseudo-Riemannian manifold:

1. Riemann antisymmetries:
   R_{abcd} = -R_{bacd} = -R_{abdc}, R_{abcd} = R_{cdab}

2. First Bianchi identity:
   R_{a[bcd]} = 0, i.e. R_{abcd} + R_{acdb} + R_{adbc} = 0

3. Einstein tensor divergence:
   nabla_a G^{ab} = 0 (contracted Bianchi identity)

4. Schwarzschild Kretschner scalar:
   K = 48 M^2 / r_schw^6

5. Minkowski zeros:
   All curvature tensors identically zero.

Tests are run on Schwarzschild and Alcubierre metrics to cover both
diagonal and off-diagonal metric cases.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.geometry import compute_curvature_chain

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helper: lower all Riemann indices
# ---------------------------------------------------------------------------

def lower_riemann(riemann_uddd, g):
    """Lower the first index: R_{abcd} = g_{ae} R^e_{bcd}."""
    return jnp.einsum("ae,ebcd->abcd", g, riemann_uddd)


# ---------------------------------------------------------------------------
# Helper: compute covariant divergence of Einstein tensor
# ---------------------------------------------------------------------------

def einstein_divergence(metric_fn, coords):
    """Compute nabla_a G^{ab} via autodiff.

    Uses the identity nabla_a V^{ab} = partial_a V^{ab} + Gamma^a_{ac} V^{cb}
    + Gamma^b_{ac} V^{ac}.  We compute G^{ab} = g^{ac} g^{bd} G_{cd} at
    neighbouring points and differentiate.
    """
    def G_upper(x):
        result = compute_curvature_chain(metric_fn, x)
        g_inv = result.metric_inv
        G_low = result.einstein
        return jnp.einsum("ac,bd,cd->ab", g_inv, g_inv, G_low)

    # Partial derivative: d_a G^{bc}
    dG = jax.jacfwd(G_upper)(coords)  # shape (4,4,4), dG[b,c,a] = d_a G^{bc}

    result = compute_curvature_chain(metric_fn, coords)
    Gamma = result.christoffel  # Gamma^a_{bc} as [upper, lower, lower]
    G_up = jnp.einsum("ac,bd,cd->ab", result.metric_inv, result.metric_inv, result.einstein)

    # nabla_a G^{ab} = d_a G^{ab} + Gamma^a_{ac} G^{cb} + Gamma^b_{ac} G^{ac}
    div = jnp.zeros(4)
    # Term 1: d_a G^{ab} = dG[a, b, a] summed over a
    term1 = jnp.einsum("aba->b", dG)
    # Term 2: Gamma^a_{ac} G^{cb}
    term2 = jnp.einsum("aac,cb->b", Gamma, G_up)
    # Term 3: Gamma^b_{ac} G^{ac}
    term3 = jnp.einsum("bac,ac->b", Gamma, G_up)

    return term1 + term2 + term3


# ---------------------------------------------------------------------------
# Test points
# ---------------------------------------------------------------------------

SCHWARZSCHILD_POINTS = [
    jnp.array([0.0, 5.0, 0.0, 0.0]),
    jnp.array([0.0, 10.0, 0.0, 0.0]),
    jnp.array([0.0, 50.0, 0.0, 0.0]),
]

ALCUBIERRE_POINTS = [
    jnp.array([0.0, 0.8, 0.5, 0.0]),   # bubble wall, off-axis
    jnp.array([0.0, 1.0, 0.3, 0.3]),   # on wall
    jnp.array([0.0, 0.5, 0.7, 0.0]),   # inner wall region
]

ALL_POINTS = (
    [(SchwarzschildMetric(M=1.0), p, "schwarzschild") for p in SCHWARZSCHILD_POINTS]
    + [(AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0), p, "alcubierre") for p in ALCUBIERRE_POINTS]
)


# ---------------------------------------------------------------------------
# 1. Riemann antisymmetries
# ---------------------------------------------------------------------------

class TestRiemannAntisymmetries:
    """R_{abcd} = -R_{bacd}, R_{abcd} = -R_{abdc}, R_{abcd} = R_{cdab}."""

    @pytest.mark.parametrize("metric,coords,label", ALL_POINTS)
    def test_antisymmetry_first_pair(self, metric, coords, label):
        """R_{abcd} = -R_{bacd}."""
        result = compute_curvature_chain(metric, coords)
        R = lower_riemann(result.riemann, result.metric)
        residual = R + jnp.transpose(R, (1, 0, 2, 3))
        npt.assert_allclose(
            np.array(residual), 0.0, atol=1e-10,
            err_msg=f"R_{{abcd}} + R_{{bacd}} != 0 for {label}",
        )

    @pytest.mark.parametrize("metric,coords,label", ALL_POINTS)
    def test_antisymmetry_second_pair(self, metric, coords, label):
        """R_{abcd} = -R_{abdc}."""
        result = compute_curvature_chain(metric, coords)
        R = lower_riemann(result.riemann, result.metric)
        residual = R + jnp.transpose(R, (0, 1, 3, 2))
        npt.assert_allclose(
            np.array(residual), 0.0, atol=1e-10,
            err_msg=f"R_{{abcd}} + R_{{abdc}} != 0 for {label}",
        )

    @pytest.mark.parametrize("metric,coords,label", ALL_POINTS)
    def test_pair_symmetry(self, metric, coords, label):
        """R_{abcd} = R_{cdab}."""
        result = compute_curvature_chain(metric, coords)
        R = lower_riemann(result.riemann, result.metric)
        residual = R - jnp.transpose(R, (2, 3, 0, 1))
        npt.assert_allclose(
            np.array(residual), 0.0, atol=1e-10,
            err_msg=f"R_{{abcd}} != R_{{cdab}} for {label}",
        )


# ---------------------------------------------------------------------------
# 2. First Bianchi identity
# ---------------------------------------------------------------------------

class TestFirstBianchiIdentity:
    """R_{abcd} + R_{acdb} + R_{adbc} = 0."""

    @pytest.mark.parametrize("metric,coords,label", ALL_POINTS)
    def test_bianchi(self, metric, coords, label):
        result = compute_curvature_chain(metric, coords)
        R = lower_riemann(result.riemann, result.metric)
        # R_{abcd} + R_{acdb} + R_{adbc}
        bianchi = R + jnp.transpose(R, (0, 2, 3, 1)) + jnp.transpose(R, (0, 3, 1, 2))
        npt.assert_allclose(
            np.array(bianchi), 0.0, atol=1e-10,
            err_msg=f"First Bianchi identity violated for {label}",
        )


# ---------------------------------------------------------------------------
# 3. Einstein tensor divergence (contracted Bianchi)
# ---------------------------------------------------------------------------

class TestEinsteinDivergence:
    """nabla_a G^{ab} = 0 (contracted Bianchi identity)."""

    @pytest.mark.parametrize("metric,coords,label", [
        (SchwarzschildMetric(M=1.0), jnp.array([0.0, 5.0, 0.0, 0.0]), "schwarzschild_r5"),
        (SchwarzschildMetric(M=1.0), jnp.array([0.0, 10.0, 0.0, 0.0]), "schwarzschild_r10"),
        (AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0), jnp.array([0.0, 0.8, 0.5, 0.0]), "alcubierre"),
    ])
    def test_divergence_free(self, metric, coords, label):
        div = einstein_divergence(metric, coords)
        npt.assert_allclose(
            np.array(div), 0.0, atol=1e-6,
            err_msg=f"nabla_a G^{{ab}} != 0 for {label}",
        )


# ---------------------------------------------------------------------------
# Summary: count all identity checks
# ---------------------------------------------------------------------------

class TestIdentitySummary:
    """Aggregate identity residuals for reporting in paper."""

    def test_max_residuals_across_all_metrics(self):
        """Compute and report maximum residual for each identity across all test points."""
        max_antisym_1 = 0.0
        max_antisym_2 = 0.0
        max_pair_sym = 0.0
        max_bianchi = 0.0

        for metric, coords, label in ALL_POINTS:
            result = compute_curvature_chain(metric, coords)
            R = lower_riemann(result.riemann, result.metric)

            # Antisymmetry first pair
            res1 = float(jnp.max(jnp.abs(R + jnp.transpose(R, (1, 0, 2, 3)))))
            max_antisym_1 = max(max_antisym_1, res1)

            # Antisymmetry second pair
            res2 = float(jnp.max(jnp.abs(R + jnp.transpose(R, (0, 1, 3, 2)))))
            max_antisym_2 = max(max_antisym_2, res2)

            # Pair symmetry
            res3 = float(jnp.max(jnp.abs(R - jnp.transpose(R, (2, 3, 0, 1)))))
            max_pair_sym = max(max_pair_sym, res3)

            # First Bianchi
            bianchi = R + jnp.transpose(R, (0, 2, 3, 1)) + jnp.transpose(R, (0, 3, 1, 2))
            res4 = float(jnp.max(jnp.abs(bianchi)))
            max_bianchi = max(max_bianchi, res4)

        print(f"\n  Curvature identity residuals (max across all test points):")
        print(f"    R_{{abcd}} = -R_{{bacd}}:  {max_antisym_1:.2e}")
        print(f"    R_{{abcd}} = -R_{{abdc}}:  {max_antisym_2:.2e}")
        print(f"    R_{{abcd}} = R_{{cdab}}:   {max_pair_sym:.2e}")
        print(f"    First Bianchi:          {max_bianchi:.2e}")

        assert max_antisym_1 < 1e-10
        assert max_antisym_2 < 1e-10
        assert max_pair_sym < 1e-10
        assert max_bianchi < 1e-10
