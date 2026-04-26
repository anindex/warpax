# Interpreting Energy Condition Results

A reference for reading warpax output: margin signs, Hawking-Ellis types,
miss-rate definitions, wall-restricted diagnostics, and when to trust a
number.

## Margin sign convention

Every EC margin returned by warpax is **signed**:

- **Margin > 0** -- energy condition satisfied at this observer and point
- **Margin = 0** -- on the violation boundary
- **Margin < 0** -- energy condition violated. The magnitude indicates how
  deep the violation is in stress-energy units.

The robust margin is the minimum over a continuous observer search
(rapidity-capped BFGS via Optimistix). The Eulerian margin uses only the
ADM normal observer. By construction:

    robust_margin <= eulerian_margin

at every point. When the inequality is strict the robust analysis has
found a boosted observer that the Eulerian-only pipeline would miss.

The structured result types are defined in
`warpax.energy_conditions.types`:

- `ECPointResult` -- single-point result: four margins, worst observer,
  worst `(zeta, theta, phi)` parameters, and Hawking-Ellis type.
- `ECGridResult` -- grid-level result: per-point margins, per-condition
  `ECSummary` (`fraction_violated`, `max_violation`, `min_margin`), and
  optional optimizer-convergence diagnostics.
- `WallRestrictedStats` -- post-hoc stats object for the wall-filtered
  subset (see below).

## Hawking-Ellis Type I-IV

At each grid point, warpax classifies the stress-energy tensor `T^a_b` by
the algebraic structure of its eigenvalues:

- **Type I** -- diagonalizable with four real eigenvalues (one timelike,
  three spacelike). The generic case; algebraic EC checks suffice.
- **Type II** -- defective 2x2 null Jordan block, degenerate eigenvalue.
  Corresponds to pure radiation. Requires null-direction optimization for
  NEC.
- **Type III** -- 3x3 null Jordan structure. Very rare in practice.
- **Type IV** -- complex eigenvalue pair (no real timelike eigenvector).
  Requires full continuous observer optimization; cannot be reduced to an
  algebraic check.

Type IV is what Reviewer #2 cared about: warp walls typically produce
Type-IV points where only BFGS-level search catches the worst violation.

See `warpax.energy_conditions.classification.classify_hawking_ellis` for
the classifier. The `he_type` field on `ECPointResult` and the
`he_types` grid on `ECGridResult` encode the type as an integer
(1=Type I, 2=Type II, 3=Type III, 4=Type IV).

## `f_miss` vs `f_miss|viol`

Two different "how much does Eulerian miss?" metrics:

- **Unconditional miss rate**
  `f_miss = # {Eulerian-satisfied and robust-violated} / # {all points}`.
  Low `f_miss` can mean "warp geometry is benign" OR "most of the grid is
  vacuum where both analyses agree". Typically dominated by vacuum
  dilution.

- **Conditional miss rate**
  `f_miss|viol = # {Eulerian-satisfied and robust-violated} / # {robust-violated}`.
  Normalized by the violation set; tells you what fraction of real
  violations the Eulerian analysis would have reported as "satisfied".

Reviewer #5 requested `f_miss|viol` alongside `f_miss`. Both are reported
by warpax: `compare_eulerian_vs_robust` returns `pct_missed` (the
unconditional rate) and `conditional_miss_rate` (the conditional rate);
`compute_wall_restricted_stats` returns `nec_miss_rate`, `wec_miss_rate`,
`sec_miss_rate`, `dec_miss_rate` -- each a wall-conditional miss rate
(the wall-restricted analogue of `f_miss|viol`).

## Wall-restricted vs full-grid

Full-grid statistics average violation fractions across vacuum regions
(where no warp geometry exists) and wall regions (where the warp field
lives). This dilutes the signal. Wall-restricted filtering uses the
shape function to isolate the active region:

```python
from warpax.energy_conditions import (
    shape_function_mask, compute_wall_restricted_stats,
)

wall_mask = shape_function_mask(metric, coords_batch, grid.shape,
                                f_low=0.1, f_high=0.9)
stats = compute_wall_restricted_stats(ec_grid, wall_mask,
                                      eulerian_margins=eul_margins)
```

The default interval `[f_low=0.1, f_high=0.9]` captures the transition
region where the shape function is neither fully interior (`f = 1`) nor
fully exterior (`f = 0`).

### Worked example: Alcubierre, `v_s=0.5`

Measured on a 50^3 grid
(`results/wall_restricted_analysis.json`):

| Statistic | Full grid | Wall-restricted |
|-------------------|-----------|-----------------|
| Grid points | 125000 | 416 |
| Type I fraction | 97.95% | 1.92% |
| Type IV fraction | 2.05% | 98.08% |
| SEC miss rate | 0.32% | 23.08% |

The wall-restricted view makes the concentration effect explicit:
essentially all Type IV points live in the warp wall. The full-grid 2.05%
is a volume-diluted version of the same phenomenon -- dividing
`2560 / 125000` instead of `408 / 416`.

The SEC miss rate moves by two orders of magnitude when restricted to the
wall. That is the point Reviewer #5 wanted to see: the full-grid SEC
miss rate of 0.32% hides a 23.08% miss rate at the physically interesting
region.

## `WallRestrictedStats` fields

Returned by `compute_wall_restricted_stats`. All counts and fractions are
conditional on the supplied wall mask.

- `n_type_i`, `n_type_ii`, `n_type_iii`, `n_type_iv` -- per-Type counts
- `frac_type_i`, `frac_type_ii`, `frac_type_iii`, `frac_type_iv` -- Type
  fractions (each in `[0, 1]`)
- `n_total` -- total points inside the wall mask
- `nec_violated`, `wec_violated`, `sec_violated`, `dec_violated` --
  per-condition violation counts
- `nec_frac_violated`, `wec_frac_violated`, `sec_frac_violated`,
  `dec_frac_violated` -- per-condition violation fractions
- `nec_miss_rate`, `wec_miss_rate`, `sec_miss_rate`, `dec_miss_rate` --
  wall-conditional miss rates. Each is `None` if no violations exist for
  that condition in the wall.

All fields are plain Python `int` / `float` / `None` (not JAX arrays) --
safe for direct printing or JSON serialization.

## When to trust a number

Use the following checks before reporting a warpax result:

1. **Resolution support.** Is the wall resolved? Table 3 in the paper
   reports wall-width / `dx` / cells-per-wall-width per metric. If the
   cells-across-wall count is under 4, the metric is under-resolved and
   reported fractions are lower bounds.
2. **Convergence tier.** Three tiers are flagged in the paper:
   **Richardson** (2+ resolutions with extrapolation), **Stability-only**
   (minimum margin stable under refinement), **Weakest** (single
   resolution, no convergence evidence). See
   `results/metric_metadata.json` `convergence_support` field.
3. **Hawking-Ellis Type distribution.** If more than ~1% Type-II or
   Type-III at interior grid points, investigate; this usually signals a
   numerical artifact at a transition zone rather than a physical
   effect.
4. **Optimizer convergence.** `ECGridResult` exposes
   `nec_opt_converged`, `wec_opt_converged`, `sec_opt_converged`,
   `dec_opt_converged` arrays (1.0 = converged, 0.0 = hit max_steps).
   Non-converged points return the best-found margin and should be
   flagged rather than silently accepted.

## Resolution limits: Lentz and WarpShell

Two metrics in the paper carry special caveats:

- **Lentz** -- wall is analytically ~44x under-resolved at 50^3 (see
  `results/lentz_wall_assessment.json`). Lentz fractions in
  `wall_restricted_analysis.json` are flagged `unresolved_lower_bound`
  and should be read as lower bounds, not point estimates. The full
  assessment is in the accompanying paper Section 4.
- **WarpShell** -- `C^2` quintic Hermite regularization of the
  thin-shell; curvature scales are extreme (max Kretschmann ~1e34).
  Results are physically valid for the regularized implementation but
  should be read as a stress-test of the EC pipeline, not as a claim
  about an idealized thin-shell spacetime. The `intrinsic_caveat`
  field in `results/metric_metadata.json` marks WarpShell with the
  `stress_test` disposition and this is carried through to the
  LaTeX table footnotes automatically.

## See also

- [`quickstart.md`](../tutorials/quickstart.md) -- install-to-first-result path
- [`custom_metric_tutorial.md`](custom_metric_tutorial.md) -- defining your
  own metric
- [`ARCHITECTURE.md`](../explanation/ARCHITECTURE.md) -- autodiff curvature pipeline
- The accompanying paper Sections 3-5 -- methodology paper
