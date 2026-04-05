# v0.1.x API defaults

The warpax v0.1.x public API surface pins every kwarg default in the
canonical `tests/fixtures/v1_api_defaults_v1_0.json` fixture. Changes to
any entry require a regeneration + CHANGELOG entry. See
`tests/test_v1_api_surface.py` for the additivity contract.

## Energy condition optimizers

| Function | Kwarg | v0.1.x default |
|----------|-------|--------------|
| `warpax.energy_conditions.optimize_nec` | `atol` | `1e-08` |
| `warpax.energy_conditions.optimize_nec` | `key` | `None` |
| `warpax.energy_conditions.optimize_nec` | `max_steps` | `256` |
| `warpax.energy_conditions.optimize_nec` | `n_starts` | `16` |
| `warpax.energy_conditions.optimize_nec` | `rtol` | `1e-08` |
| `warpax.energy_conditions.optimize_nec` | `strategy` | `'tanh'` |
| `warpax.energy_conditions.optimize_wec` | `atol` | `1e-08` |
| `warpax.energy_conditions.optimize_wec` | `zeta_max` | `5.0` |
| `warpax.energy_conditions.optimize_wec` | `strategy` | `'tanh'` |
| `warpax.energy_conditions.optimize_sec` | `atol` | `1e-08` |
| `warpax.energy_conditions.optimize_sec` | `zeta_max` | `5.0` |
| `warpax.energy_conditions.optimize_sec` | `strategy` | `'tanh'` |
| `warpax.energy_conditions.optimize_dec` | `atol` | `1e-08` |
| `warpax.energy_conditions.optimize_dec` | `max_steps` | `256` |
| `warpax.energy_conditions.optimize_dec` | `mode` | `'three_term_min'` |
| `warpax.energy_conditions.optimize_dec` | `strategy` | `'tanh'` |

## Classification

| Function | Kwarg | v0.1.x default |
|----------|-------|--------------|
| `warpax.energy_conditions.classify_hawking_ellis` | `imag_rtol` | `0.003` |
| `warpax.energy_conditions.classify_hawking_ellis` | `tol` | `1e-10` |
| `warpax.energy_conditions.classify_hawking_ellis_mpmath` | `precision` | `50` |
| `warpax.energy_conditions.verify_classification_at_points` | `precision` | `50` |
| `warpax.energy_conditions.eigenvalues_mpmath` | `precision` | `50` |

## Verification

| Function | Kwarg | v0.1.x default |
|----------|-------|--------------|
| `warpax.energy_conditions.verify_point` | `n_starts` | `16` |
| `warpax.energy_conditions.verify_point` | `zeta_max` | `5.0` |
| `warpax.energy_conditions.verify_grid` | `n_starts` | `16` |
| `warpax.energy_conditions.verify_grid` | `zeta_max` | `5.0` |
| `warpax.energy_conditions.compute_wall_restricted_stats` | `atol` | `1e-10` |
| `warpax.energy_conditions.shape_function_mask` | `f_low` | `0.1` |
| `warpax.energy_conditions.shape_function_mask` | `f_high` | `0.9` |

## Benchmarks

| Class / Function | Kwarg | v0.1.x default |
|------------------|-------|--------------|
| `warpax.benchmarks.AlcubierreMetric.__init__` | `v_s` | `0.5` |
| `warpax.benchmarks.AlcubierreMetric.__init__` | `R` | `1.0` |
| `warpax.benchmarks.AlcubierreMetric.__init__` | `sigma` | `8.0` |
| `warpax.benchmarks.SchwarzschildMetric.__init__` | `M` | `1.0` |

## Metrics

| Class / Function | Kwarg | v0.1.x default |
|------------------|-------|--------------|
| `warpax.metrics.LentzMetric.__init__` | `v_s` | `0.1` |
| `warpax.metrics.LentzMetric.__init__` | `R` | `100.0` |
| `warpax.metrics.NatarioMetric.__init__` | `v_s` | `0.1` |
| `warpax.metrics.RodalMetric.__init__` | `v_s` | `0.1` |
| `warpax.metrics.VanDenBroeckMetric.__init__` | `alpha_vdb` | `0.5` |
| `warpax.metrics.WarpShellMetric.__init__` | `v_s` | `0.02` |
| `warpax.metrics.WarpShellPhysical.__init__` | `v_s` | `0.02` |
| `warpax.metrics.WarpShellStressTest.__init__` | `v_s` | `0.02` |

!!! note "Full table"
    The full pinned-default table lives at
    `tests/fixtures/v1_api_defaults_v1_0.json` (~60 entries). The entries
    above are the ones most often referenced by downstream consumers; see
    the JSON for the complete surface.

## Regeneration

After any v0.x change adds a kwarg (additive only - forbids
modifications / removals):

```bash
pytest tests/test_v1_api_surface.py --regenerate
# Commit the diff together with a CHANGELOG entry.
```
