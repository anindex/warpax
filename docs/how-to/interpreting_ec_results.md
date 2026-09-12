# Interpreting energy-condition results

A margin is meaningful only with its method, frame, normalization, and tolerance.
A negative contraction gives a violating observer. A positive result from a
finite search does not establish satisfaction for every observer.

## Choose the right result

| API or diagnostic | Meaning |
|---|---|
| `energy_conditions.slemma.certify_point` | All-observer LMI test for each condition, with a numerical noise floor |
| `warpax.certify`, `frame_free` | Cap-free classification/eigenvalue route with LMI tests at non-Type-I or ill-conditioned points |
| Type-I eigenvalue slack | Rest-frame inequality such as `min_i(rho + p_i)` |
| `verify_point`, `verify_grid` | Type-I slacks plus bounded-rapidity optimizer diagnostics; non-Type-I values come from the search |
| Eulerian contraction | A diagnostic relative to the unit slice normal |
| Rational certificate | Exact sufficient evidence for the supplied tensor entries, when construction succeeds |
| Interval enclosure | Bound including the uncertainty covered by the specified interval calculation |

The sign of an all-observer energy condition is frame independent. Its LMI
margin, a normalized null contraction, and a capped-search minimum generally
have different magnitudes. In particular, an eigenframe Type-I slack cannot be
compared numerically with an Eulerian contraction as a universal severity scale.
A reliable search on the same objective and domain can improve an Eulerian
candidate, but generic optimizer outputs do not carry a guaranteed ordering.

`ECPointResult` contains margins, observer parameters, and the algebraic type.
`ECGridResult` adds per-condition summaries and optional convergence arrays.
`WallRestrictedStats` summarizes a supplied mask.

## Hawking-Ellis classification

| `he_type` | Structure |
|---|---|
| 1 | Real diagonal form with one timelike and three spacelike eigenvectors |
| 2 | Null Jordan block of size 2 |
| 3 | Null Jordan block of size 3 |
| 4 | Complex-conjugate eigenvalue pair; no timelike rest frame |

Type I permits eigenvalue inequalities. LMI tests cover every type, so Type IV
does not require observer optimization to decide the energy conditions.
Near-degenerate floating-point eigenvectors can give unreliable labels.

`solver="auto"` starts with the standard eigensolver and falls back to a
generalized pencil solve when needed and when `warpax[solver]` is installed.
Use `solver="generalized"` to force that route or `solver="standard"` for the
pure-JAX route. Check ambiguous labels with conditioning and precision tests.
The matched Garattini-Zatrimaylov construction has an exact Type-I reduction;
its construction panel uses that reduction in masks and denominators.

## Miss rates and units

For a stated tolerance, mask, and reference verification method:

$$
f_{\rm miss}=\frac{N(\text{Eulerian satisfied, reference violated})}{N(\text{all points})},
\qquad
f_{{\rm miss}|{\rm viol}}=
\frac{N(\text{Eulerian satisfied, reference violated})}{N(\text{reference violated})}.
$$

The unconditional rate is diluted by points outside the active wall. The
conditional rate measures the fraction of detected violations missed by the
Eulerian test. An optimizer-based reference includes only violations it found.

| Result | Range | Formatting |
|---|---|---|
| `ComparisonResult.pct_missed`, `conditional_miss_rate`, `pct_violated_robust` | Percent, `[0,100]` | `:.1f}%` |
| `WallRestrictedStats.*_miss_rate` | Fraction, `[0,1]`, or `None` when there are no violations | `:.1%` after checking for `None` |

State whether statistics count points or use proper-volume weights. A missed
sample does not provide a lower bound on the continuum violation fraction.

## Wall-restricted statistics

Continue with the arrays from the [custom metric tutorial](custom_metric_tutorial.md):

```python
from warpax.energy_conditions import shape_function_mask, compute_wall_restricted_stats

wall_mask = shape_function_mask(metric, coords_batch, grid.shape,
                                f_low=0.1, f_high=0.9)
stats = compute_wall_restricted_stats(
    ec_grid, wall_mask, eulerian_margins=comparison.eulerian_margins,
)
```

The default mask selects `0.1 <= f <= 0.9`. It describes a chosen transition
region, not a coordinate-independent definition of the wall.

For the Alcubierre `v_s=0.5`, `50^3` example in
`results/wall_restricted_analysis.json`:

| Statistic | Full grid | Wall mask |
|---|---|---|
| Grid points | 125000 | 416 |
| Type-I fraction | 84.51% | 0.00% |
| Type-IV fraction | 15.49% | 100.00% |
| Reported SEC miss statistic | 7.19% unconditional | 15.38% conditional on wall violations |

The last row uses different denominators and should not be read as a like-for-like
factor-of-two increase. The type fractions describe this sampled grid.

`WallRestrictedStats` fields are ordinary Python numbers:

- `n_total`; `n_type_i`, `n_type_ii`, `n_type_iii`, `n_type_iv`.
- `frac_type_i`, `frac_type_ii`, `frac_type_iii`, `frac_type_iv`.
- For each of `nec`, `wec`, `sec`, `dec`: `*_violated`, `*_frac_violated`,
  and `*_miss_rate`.

## Before reporting a result

1. Resolve the wall and refine the grid. Low cells-per-wall counts make
   fractions unreliable; stability of a sampled minimum is not a continuum bound.
2. Report the convergence method. Richardson extrapolation needs a suitable
   refinement ladder and fitted order; a small observed spread is a different claim.
3. Inspect type conditioning, tolerances, and points marked inconclusive.
4. Check `*_opt_converged` arrays. Nonconverged searches return a best-found
   value, not a certified optimum.
5. For null integrals, report endpoints, affine normalization, excluded rays,
   step refinement, and null-norm drift. Omitted-tail bounds are needed before
   claiming complete-geodesic ANEC.

Lentz's thin wall is unresolved on the common coarse grid. WarpShell's
regularized transitions can produce very large curvature; those results concern
the chosen regularized metric, not an ideal distributional thin shell.

See the [theory](../explanation/theory.md) and
[reproduction guide](reproduce_observer_robust_paper.md) for the applicable limits.
