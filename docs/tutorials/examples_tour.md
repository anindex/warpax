# Examples tour

Ten numbered scripts under `examples/`, from a flat-space sanity check to a
design-space phase diagram. Each is self-contained and runs on CPU.
[`examples/README.md`](https://github.com/anindex/warpax/blob/main/examples/README.md)
lists them all with runtimes and the extras each one needs.

```bash
pip install -e ".[dev,viz,design,solver]"
```

Prefix any command with `JAX_PLATFORMS=cpu` for bit-identical CPU runs on a
machine with a GPU.

## First three runs

```bash
python examples/01_minkowski_sanity.py      # ~10 s
python examples/03_alcubierre_analysis.py   # ~10 s, same as the quickstart
python examples/07_custom_warp_metric.py    # ~40 s, custom metric + figure
```

01 verifies the curvature chain returns exact zero on flat space. 03 is the main
result: at an Alcubierre bubble-wall point the Eulerian observer already reads
the WEC as violated, at -1.7e-03, while the worst-case boosted observer puts the
same point at -4.6e+02, five orders deeper. 07 lifts the same machinery onto a
custom `ADMMetric` subclass.

## Next

- [Interpreting EC results](../how-to/interpreting_ec_results.md): margin signs
  and Hawking-Ellis types.
- [Define a custom warp metric](../how-to/custom_metric_tutorial.md): the full
  subclassing recipe.
- [Reproducing the warp-shell admissibility paper](../how-to/reproduce_warpshell_paper.md):
  figure-by-figure mapping back to the published results.
