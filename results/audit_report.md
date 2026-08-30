# Read-Only Audit Report

## 1. MATHEMATICAL CORRECTNESS (Weight 1)
- **src/warpax/energy_conditions/enclosure.py:33**
  - **Defect:** `_lo(c)` and `_hi(c)` cast `mpmath.mpf` directly to Python `float`, which rounds to nearest rather than outward, breaking the branch-and-bound interval enclosure and allowing the true global minimum to be improperly discarded.
  - **Reproducer:** `import mpmath; from warpax.energy_conditions.enclosure import _hi; _hi(mpmath.mpf("0.10000000000000001"))` returns exactly `0.1` (rounding down), which would falsely pass a `< 0.1` threshold.
  - **Severity:** Fatal

## 2. QUADRATURE AND AGGREGATION (Weight 2)
- **src/warpax/grids/_axisymmetric.py:48**
  - **Defect:** The radial volume weight uses a pointwise trapezoidal rule `(2.0 * np.pi * R**2 * WR * WMU)` which evaluates to exactly `0` at `R=0`, systematically missing the $O(\Delta r^3)$ origin volume and biasing fractions on the grid.
  - **Reproducer:** `import numpy as np; from warpax.grids._axisymmetric import axisymmetric_grid; grid = axisymmetric_grid(2.0, 40, 40, center=1.0); np.sum(grid.weights[np.sqrt((grid.coords[:,1]-1.0)**2+grid.coords[:,2]**2)<=1.0])` converges to `4.186` instead of `4/3 * pi * 1.0^3 = 4.18879`.
  - **Severity:** Major

- **src/warpax/design/objectives.py:110**
  - **Defect:** `ec_margin_objective` replaces `NaN` margins with `jnp.inf` before calling `jnp.min()` to find the worst point, which silently hides singular points (where the curvature evaluation fails) from the optimizer, making degenerate designs look infinitely good.
  - **Reproducer:** `import jax.numpy as jnp; margins = jnp.array([1.0, 2.0, jnp.nan]); jnp.min(jnp.where(jnp.isfinite(margins), margins, jnp.inf))` returns `1.0`, ignoring the failure.
  - **Severity:** Fatal

- **src/warpax/grids/_clustered.py:141**
  - **Defect:** The function `_cosh_stretch` evaluates `np.sinh(a * x) / np.sinh(a)` without a fallback for `a=0.0`, resulting in a `0/0 = NaN` crash when unclustered (uniform) grids are requested.
  - **Reproducer:** `import numpy as np; x = np.linspace(-1, 1, 10); a = 0.0; np.sinh(a * x) / np.sinh(a)` returns an array of `NaN`s.
  - **Severity:** Cosmetic

## 3. FITTING AND STATISTICS (Weight 3)
- **scripts/run_convergence.py:96**
  - **Defect:** `richardson_extrapolation` reads $r = h_1 / h_2$ and unconditionally assumes $h_2 / h_3 = r$, computing garbage convergence orders and extrapolations if passed non-geometric grid sizes like `--resolutions 30 40 50`.
  - **Reproducer:** Passing `values=[1.0, 0.5, 0.25], grid_sizes=[20, 30, 40]` incorrectly computes $r = 1.5$ and assumes $40/30 = 1.5$, producing an invalid order estimate.
  - **Severity:** Major

- **scripts/run_ssv_bound.py:108**
  - **Defect:** The free log-log exponent fit calculates its $R^2$ in log space (`ss_res_l / ss_tot_l`) but directly compares it against the fixed-exponent model's $R^2$ calculated in linear space, which is a mathematically invalid comparison of goodness-of-fit.
  - **Reproducer:** The script outputs `r_squared_free` computed from `np.log` residuals alongside `r_squared_fixed` from linear residuals.
  - **Severity:** Major

- **scripts/run_curvature_convergence.py:115**
  - **Defect:** The `fit_power_law` loop silently drops points where invariant `v < 0`, hiding bugs if an inherently positive physical invariant (like $K$ or Weyl squared) evaluates to a negative value due to a computational error.
  - **Reproducer:** `v = -1.0; if v is not None and v > 0:` silently skips the invalid negative invariant instead of breaking the log fit.
  - **Severity:** Major

## 4. PERFORMANCE (Weight 4)
- **src/warpax/design/objectives.py:98**
  - **Defect:** `_per_point_margin` is defined *inside* `ec_margin_objective` and decorated with `@eqx.filter_jit`, which creates a new Python function object on every call, busting the JAX compilation cache and triggering a multi-second full-chain recompilation on every objective evaluation.
  - **Reproducer:** Calling `ec_margin_objective(...)` repeatedly in a loop will take seconds per call instead of milliseconds because it JIT-compiles every time.
  - **Severity:** Fatal
