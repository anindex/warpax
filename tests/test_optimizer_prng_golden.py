"""Golden-fixture test for the observer-optimiser multistart pool.

``_make_initial_conditions_3d`` is the entry point that every per-point
BFGS optimisation starts from. The pinned values catch JAX PRNG-semantic
changes that would shift downstream worst-case observers (and the
paper's numbers) - the oracle is JAX ``random.normal`` under the locked
dependency, not the function under test.

Golden values captured under ``jax==0.10.0`` on x86_64 / CPU / float64.
"""
from __future__ import annotations

import jax
import numpy as np
import numpy.testing as npt

from warpax.energy_conditions.optimization import _make_initial_conditions_3d


# Rows 0-6: deterministic axis-aligned boosts at zeta_max.
# Rows 7-15: jax.random.normal(PRNGKey(42), (9, 3)) * 5.0.
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
