# EinFields fixture

`minkowski.ckpt/` is a synthetic Orbax checkpoint used by `tests/test_io.py`
to test `load_einfield`. It stores `eta_metric`, a `(4,4)` float64 array
`diag(-1,1,1,1)`. The loader samples it on a regular grid and returns an
`InterpolatedADMMetric`. This exercises checkpoint restoration, not a trained
network's accuracy.

```bash
python -m pip install -e ".[einfields]"
python tests/fixtures/einfields/generate_minkowski_ckpt.py
```

The committed checkpoint is about 28 KB. Tests skip if optional Flax/Orbax
packages are absent or if network topology cannot be rebuilt; inspect skips
when validating this integration.
