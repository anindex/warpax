"""Shape-function design: spline reproduction, NEC-deficit optimization, certification.

Usage::

    python examples/08_metric_design.py

Stage 1 places the Alcubierre profile ``1 - tanh((r - R) / sigma)^2`` on 24
uniform Hermite-spline knots over [0, 12] and runs ``design_metric`` with
``max_steps=0``, which returns the seed unchanged. Interpolation is exact at
the knots, so the knot error is a reproduction check; the dense-probe error is
large because the 0.52 knot spacing undersamples a wall of width 0.1. The
reproduced values are written to ``tests/fixtures/alcubierre_optimal_parameters.npy``.

Stage 2 optimizes. ``design_metric`` raises the worst-point NEC margin on an
8^3 probe grid over [-2, 2]^3 under a ``decayed`` soft penalty on shape
beyond ``r = 2.5`` (``|f(2.5)| < 0.01``). The seed is a wider wall
(``sigma = 0.3``) on 24 knots over [0, 6], and BFGS runs for a bounded number
of steps. A probe grid can be gamed by aliasing, so the seed and the design
are re-evaluated on a held-out 12^3 grid over [-2.3, 2.3]^3 and the example
asserts the designed margin is no worse there.

Stage 3 certifies the seed and the designed metric with the frame-free
Hawking-Ellis certifier on a wall-clustered grid and reports the Type-I
fraction, the invariant NEC and DEC minima and the Eulerian NEC miss rate.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from warpax import certify
from warpax.design import (
    CONSTRAINT_REGISTRY,
    ConstraintResult,
    ShapeFunction,
    design_metric,
    ec_margin_objective,
)


def alcubierre_values(knots, R: float, sigma: float):
    return 1.0 - jnp.tanh((knots - R) / sigma) ** 2


def decayed_constraint(sf: ShapeFunction, r_far: float = 2.5) -> ConstraintResult:
    """Soft penalty on ``|f(r_far)| >= 0.01``, scaled so it is not traded for NEC margin."""
    margin = 100.0 * (0.01 - jnp.abs(sf(jnp.asarray(r_far))))
    return ConstraintResult(satisfied=margin >= 0.0, margin=margin, name="decayed")


PROBE = dict(probe_shape=(8, 8, 8), probe_bounds=((-2.0, 2.0),) * 3)
HELD_OUT = dict(grid_shape=(12, 12, 12), bounds=((-2.3, 2.3),) * 3)

CONSTRAINT_REGISTRY["decayed"] = decayed_constraint


def reproduce() -> None:
    R, sigma = 1.0, 0.1
    knots = jnp.linspace(0.0, 12.0, 24)
    values = alcubierre_values(knots, R, sigma)
    metric, report = design_metric(
        ShapeFunction.spline(knots, values),
        objective="nec",
        strategy="hard_bound",
        n_starts=16,
        max_steps=0,
        key=jax.random.PRNGKey(42),
    )
    recovered = jax.vmap(metric.shape_fn)(knots)
    rel_err = float(jnp.max(jnp.abs(recovered - values)) / jnp.max(jnp.abs(values)))
    x_dense = jnp.linspace(0.0, 12.0, 1000)
    dense_err = float(
        jnp.max(jnp.abs(jax.vmap(metric.shape_fn)(x_dense) - alcubierre_values(x_dense, R, sigma)))
    )

    print(f"Stage 1: reproduction (24 Hermite knots on [0, 12], R={R}, sigma={sigma})")
    print(f"  Relative error at knots    : {rel_err:.2e} (target < 1e-4)")
    print(f"  Peak-normalized dense error: {dense_err:.3f} (1000 probes)")
    print(f"  Physical                   : {report.physicality.overall}")
    print(f"  Optimizer steps            : {report.n_steps}")
    assert rel_err < 1e-4, f"rel_err={rel_err:.2e} >= 1e-4"

    fixture = (
        pathlib.Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "alcubierre_optimal_parameters.npy"
    )
    fixture.parent.mkdir(parents=True, exist_ok=True)
    np.save(fixture, np.asarray(metric.shape_fn.params["values"]))
    print(f"  Fixture saved: {fixture}")


def design():
    knots = jnp.linspace(0.0, 6.0, 24)
    shape = ShapeFunction.spline(knots, alcubierre_values(knots, 1.0, 0.3))
    seed, seed_report = design_metric(shape, objective="nec", max_steps=0, **PROBE)
    designed, report = design_metric(
        shape,
        objective="nec",
        constraints=["decayed"],
        n_starts=2,
        max_steps=64,
        key=jax.random.PRNGKey(42),
        **PROBE,
    )
    assert report.n_steps > 0 and float(report.final_margin) > float(seed_report.final_margin)
    held_out = {
        label: float(ec_margin_objective(m, "nec", **HELD_OUT))
        for label, m in (("seed", seed), ("designed", designed))
    }
    f_far = float(designed.shape_fn(jnp.asarray(2.5)))
    assert held_out["designed"] >= held_out["seed"], held_out
    assert abs(f_far) < 0.01, f"f(2.5)={f_far:+.4f}"

    print("\nStage 2: NEC margin design (24 Hermite knots on [0, 6], R=1, sigma=0.3)")
    print(f"  Seed worst-point NEC margin     : {float(seed_report.final_margin):+.4e} (8^3 probe)")
    print(f"  Designed worst-point NEC margin : {float(report.final_margin):+.4e} (8^3 probe)")
    print(
        f"  Held-out 12^3 NEC margin        : seed {held_out['seed']:+.4e}, designed {held_out['designed']:+.4e}"
    )
    print(f"  BFGS steps (best start)         : {report.n_steps}, converged={report.converged}")
    print(f"  Designed f(2.5)                 : {f_far:+.4e} (constraint |f| < 0.01)")
    print(f"  Physical                        : {report.physicality.overall}")
    return seed, designed


def certify_both(seed, designed) -> None:
    print("\nStage 3: frame-free certification (wall-clustered 16^3 grid, [-3, 3]^3)")
    for label, metric in (("seed", seed), ("designed", designed)):
        result = certify(metric, shape=(16, 16, 16))
        nec_miss = result.single_frame_miss["nec"]["miss_rate"]
        print(
            f"  {label:8s}: Type I {result.type_fractions['frac_type_i']:.1%} of the wall, "
            f"NEC min {result.invariant_nec_min:+.4e}, DEC min {result.invariant_dec_min:+.4e}, "
            f"Eulerian NEC miss rate {nec_miss:.1%}"
        )
    print("  NaN minima mean no wall point has a rest frame (all Type IV).")


if __name__ == "__main__":
    reproduce()
    certify_both(*design())
