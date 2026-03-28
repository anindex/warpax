"""Golden-fixture test for the observer-optimiser multistart pool.

``_make_initial_conditions_3d`` is the entry point that every per-point
BFGS optimisation starts from. If the JAX PRNG internals change in a
way that perturbs the sampled boost vectors, the downstream worst-case
observers shift and the paper's numbers drift with them.

This test pins the exact 16-point starter pool at the canonical
``(n_starts=16, zeta_max=5.0, key=PRNGKey(42))`` configuration so any
PRNG-semantic change will fail CI immediately. The golden values were
captured under ``jax==0.10.0`` on x86_64 / CPU / float64.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from warpax.energy_conditions.optimization import _make_initial_conditions_3d


# Golden starter pool. Row order:
# rows 0-6 -- Eulerian + 6 axis-aligned boosts at zeta_max
# rows 7-15 -- Gaussian-sampled boost vectors scaled by zeta_max
#
# The first 7 rows are deterministic (no PRNG); rows 7-15 are the
# float64 samples returned by jax.random.normal(PRNGKey(42), (9, 3))
# scaled by 5.0. Regenerating after a JAX update: run
# from warpax.energy_conditions.optimization import _make_initial_conditions_3d
# import jax
# print(_make_initial_conditions_3d(16, 5.0, jax.random.PRNGKey(42)))
GOLDEN = np.array(
    [
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [-5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, -5.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, -5.0],
        [-0.923558726409558, -10.84912280198877, 0.9346777589691291],
        [3.0613267852554795, 2.4481247523785092, 1.8450215235414178],
        [1.8543731889309525, 1.1766366207640897, -3.5844941575956857],
        [-3.173121663824603, -3.50010940833805, -7.882790659278354],
        [2.98935432810288, 4.546202300849532, 1.116331111966121],
        [-3.6799338948667746, -10.100550686727946, 1.6485357408377945],
        [-3.8216242644691234, 9.264443397260278, 0.2867568066460418],
        [-5.894341769959686, 1.2134601729839203, 4.332715782619537],
        [-1.1019365643520587, 11.787525561966842, 0.5508630843025522],
    ],
    dtype=np.float64,
)


def test_golden_starter_pool_at_default_config():
    """The canonical starter pool must match the captured values bit-for-bit."""
    ic = np.asarray(_make_initial_conditions_3d(
        n_starts=16, zeta_max=5.0, key=jax.random.PRNGKey(42)
    ))
    assert ic.shape == GOLDEN.shape
    assert ic.dtype == GOLDEN.dtype
    npt.assert_array_equal(ic, GOLDEN)


def test_golden_deterministic_rows_independent_of_key():
    """The first 7 rows come from the deterministic axis-aligned block; the
    PRNG key should not affect them."""
    ic_k0 = np.asarray(_make_initial_conditions_3d(
        n_starts=16, zeta_max=5.0, key=jax.random.PRNGKey(0)
    ))
    ic_k42 = np.asarray(_make_initial_conditions_3d(
        n_starts=16, zeta_max=5.0, key=jax.random.PRNGKey(42)
    ))
    npt.assert_array_equal(ic_k0[:7], ic_k42[:7])


def test_golden_gaussian_rows_do_depend_on_key():
    """Rows 7-15 are PRNG-driven and must differ between PRNG keys --
    this catches a regression where the key is accidentally ignored."""
    ic_k0 = np.asarray(_make_initial_conditions_3d(
        n_starts=16, zeta_max=5.0, key=jax.random.PRNGKey(0)
    ))
    ic_k42 = np.asarray(_make_initial_conditions_3d(
        n_starts=16, zeta_max=5.0, key=jax.random.PRNGKey(42)
    ))
    assert not np.allclose(ic_k0[7:], ic_k42[7:])


def test_golden_reproducible_across_repeated_calls():
    """Calling the builder twice with the same key yields identical output."""
    key = jax.random.PRNGKey(42)
    ic1 = np.asarray(_make_initial_conditions_3d(n_starts=16, zeta_max=5.0, key=key))
    ic2 = np.asarray(_make_initial_conditions_3d(n_starts=16, zeta_max=5.0, key=key))
    npt.assert_array_equal(ic1, ic2)


def test_golden_scales_linearly_with_zeta_max():
    """Doubling ``zeta_max`` doubles the starter pool's magnitudes
    component-wise because both the axis-aligned and Gaussian branches
    scale linearly with ``zeta_max``."""
    ic1 = np.asarray(_make_initial_conditions_3d(
        n_starts=16, zeta_max=5.0, key=jax.random.PRNGKey(42)
    ))
    ic2 = np.asarray(_make_initial_conditions_3d(
        n_starts=16, zeta_max=10.0, key=jax.random.PRNGKey(42)
    ))
    npt.assert_allclose(ic2, 2.0 * ic1, rtol=0, atol=0)
