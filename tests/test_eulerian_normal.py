"""One Eulerian-normal policy: NaN wherever the t = const slice is not spacelike."""

import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.energy_conditions import compute_eulerian_ec
from warpax.energy_conditions.frame_free import eulerian_momentum_frame
from warpax.energy_conditions.observer import compute_orthonormal_tetrad, eulerian_normal
from warpax.geodesics.initial_conditions import eulerian_frequency
from warpax.geometry.adm_split import adm_split

from ._helpers import ETA

NON_SPACELIKE = jnp.diag(jnp.array([1.0, -1.0, 1.0, 1.0]))


def test_normal_is_unit_timelike_on_valid_slices():
    g = jnp.diag(jnp.array([-4.0, 1.0, 1.0, 1.0]))
    alpha, n = eulerian_normal(jnp.linalg.inv(g))
    assert abs(float(alpha) - 2.0) < 1e-14
    assert abs(float(n @ g @ n) + 1.0) < 1e-14
    assert float(n[0]) > 0


def test_non_spacelike_slice_gives_nan_everywhere():
    """g = diag(1,-1,1,1) is Lorentzian but g^00 = +1, so no unit timelike
    normal to t = const exists; floors and absolute values would fabricate
    finite margins, frequencies and a 1e-15 lapse here."""
    g = NON_SPACELIKE
    gi = jnp.linalg.inv(g)
    alpha, n = eulerian_normal(gi)
    assert jnp.isnan(alpha) and bool(jnp.all(jnp.isnan(n)))
    assert bool(jnp.all(jnp.isnan(compute_orthonormal_tetrad(g)[0])))
    ec = compute_eulerian_ec(jnp.eye(4), g, gi)
    assert all(jnp.isnan(v) for v in ec.values())
    rho, s_par, jmag = eulerian_momentum_frame(jnp.eye(4), g, gi)
    assert jnp.isnan(rho)
    assert jnp.isnan(eulerian_frequency(lambda x: g, jnp.zeros(4), jnp.array([1.0, 0.0, 0.0, 0.0])))
    assert jnp.isnan(adm_split(lambda x: g, jnp.zeros(4)).lapse)


def test_superluminal_alcubierre_keeps_a_timelike_normal():
    """g_00 > 0 at the wall for v_s = 2 does not make g^00 >= 0: the ADM normal
    stays unit timelike at every warp speed."""
    m = AlcubierreMetric(v_s=2.0, R=1.0, sigma=8.0)
    x = jnp.array([0.0, 1.0, 0.0, 0.0])
    g = m(x)
    assert float(g[0, 0]) > 0
    alpha, n = eulerian_normal(jnp.linalg.inv(g))
    assert np.isfinite(float(alpha))
    assert abs(float(n @ g @ n) + 1.0) < 1e-12


def test_batched_normal_matches_pointwise():
    m = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    pts = jnp.array([[0.0, 0.3, 0.1, 0.0], [0.0, 1.0, 0.0, 0.2], [0.0, 2.0, 0.5, 0.5]])
    gi = jnp.stack([jnp.linalg.inv(m(x)) for x in pts])
    alpha_b, n_b = eulerian_normal(gi)
    for i in range(3):
        a, n = eulerian_normal(gi[i])
        assert abs(float(alpha_b[i] - a)) < 1e-15
        assert bool(jnp.allclose(n_b[i], n, atol=1e-15))


def test_interval_fields_reject_non_spacelike_box():
    """The interval chain refuses a box on which g^00 n_0 n_0 is not negative."""
    from mpmath import iv

    from warpax.energy_conditions import _intervalad as ad
    from warpax.energy_conditions._intervalcurv import eulerian_fields_interval

    def metric(t, x, y, z):
        one = ad.constant(1)
        zero = ad.constant(0)
        return [
            [one, zero, zero, zero],
            [zero, -one, zero, zero],
            [zero, zero, one, zero],
            [zero, zero, zero, one],
        ]

    box = [iv.mpf([0, 0.1]) for _ in range(4)]
    with pytest.raises(ValueError, match="spacelike"):
        eulerian_fields_interval(metric, box)


def test_eulerian_nec_is_the_null_sphere_minimum():
    """In an orthonormal frame with T00 = 1 and T12 = T21 = 2 every axial null
    direction n +- e_i gives +1, while s = (1,-1,0)/sqrt(2) gives rho + s^T S s =
    1 - 2 = -1, which is the exact minimum (b = 0, eig(S) = -2, 0, 2)."""
    T = jnp.zeros((4, 4)).at[0, 0].set(1.0).at[1, 2].set(2.0).at[2, 1].set(2.0)
    ec = compute_eulerian_ec(T, ETA, ETA)
    assert abs(float(ec["nec"]) + 1.0) < 1e-8
    assert abs(float(ec["wec"]) - 1.0) < 1e-12


def test_negative_null_dust_counts_as_a_violation():
    """T = -l l with l = (1,1,0,0) is Type II (no timelike eigenvector) and fails
    every condition: rho_Eul = -1 and the null deficit is min_s -(-1 + s_x)^2 = -4.
    It must not be excluded from the all-observer count as a non-Type-I point."""
    from warpax.analysis.invariant_verification import single_frame_miss

    ell = jnp.array([1.0, 1.0, 0.0, 0.0])
    T = -jnp.outer(ell, ell)
    out = single_frame_miss(T[None], ETA[None], ETA[None])
    ec = compute_eulerian_ec(T, ETA, ETA)
    assert abs(float(ec["nec"]) + 4.0) < 1e-8
    assert abs(float(ec["wec"]) + 1.0) < 1e-12
    for c in ("nec", "wec", "sec", "dec"):
        assert out[c]["n_violated"] == 1
        assert out[c]["n_missed"] == 0
        assert out[c]["miss_rate"] == 0.0
        assert out[c]["n_undecided"] == 0


def test_kinematic_scalars_are_nan_on_non_spacelike_slices():
    from warpax.analysis.kinematic_scalars import compute_kinematic_scalars

    def metric_fn(coords):
        return NON_SPACELIKE

    theta, sigma_sq, omega_sq = compute_kinematic_scalars(metric_fn, jnp.zeros(4))
    assert np.isnan(float(theta)) and np.isnan(float(sigma_sq))
    assert float(omega_sq) == 0.0
