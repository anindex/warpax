# EinFields fixture

`minkowski.ckpt/` - hand-synth Orbax checkpoint that makes the fixture's
forward pass output `eta_{ab}` (Minkowski) at any input. Used by
`tests/test_io_einfields.py` to exercise `warpax.io.load_einfield`
(, ).

## Schema

The checkpoint stores a single key, `eta_metric`, a `(4, 4) float64`
array initialized to `diag(-1, 1, 1, 1)`. `load_einfield` samples this
on a regular 4D grid and constructs an `InterpolatedADMMetric`.

This is a deliberately minimal stand-in for a real EinFields network -
enough to exercise the loader's Orbax restore path without pulling in
the full Flax NNX dependency tree for every CI run.

## Regeneration

```bash
pip install 'warpax[einfields]'
python tests/fixtures/einfields/generate_minkowski_ckpt.py
```

The generator writes `minkowski.ckpt/` (an Orbax-populated directory).
The committed `.gitkeep` marker keeps the path under version control
when the extras are not installed (e.g., standard CI runs).

## flax version drift

Per / ARCH-2 mitigation: if Flax's NNX topology API drifts,
loader tests honest-skip via `pytest.importorskip('flax')` +
`pytest.importorskip('orbax.checkpoint')` + explicit `pytest.skip` on
topology-rebuild failure. CI coverage is honest - skip when env is
stale, not silently pass.
