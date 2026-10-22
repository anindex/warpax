# API reference

Public symbols are rendered from source docstrings with
[mkdocstrings](https://mkdocstrings.github.io/python/). Array annotations retain
their jaxtyping shapes, such as `Float[Array, "4 4"]`.

## warpax.geometry

::: warpax.geometry
    options:
      show_root_heading: false

## warpax.metrics

`SShellMetric` and `TShellMetric` implement prescriptions with boundary-matching
and momentum-equation defects, described in the
[shell guide](../how-to/reproduce_warpshell_paper.md). They do not implement the
Einstein–elastic model of example 11.

::: warpax.metrics
    options:
      show_root_heading: false

## warpax.benchmarks

::: warpax.benchmarks
    options:
      show_root_heading: false

## warpax.energy_conditions

::: warpax.energy_conditions
    options:
      show_root_heading: false

## warpax.geodesics

::: warpax.geodesics
    options:
      show_root_heading: false

## warpax.analysis

::: warpax.analysis
    options:
      show_root_heading: false

## warpax.averaged

::: warpax.averaged
    options:
      show_root_heading: false

## warpax.quantum

::: warpax.quantum
    options:
      show_root_heading: false

## warpax.bondi

::: warpax.bondi
    options:
      show_root_heading: false

## warpax.grids

::: warpax.grids
    options:
      show_root_heading: false

## warpax.junction

::: warpax.junction
    options:
      show_root_heading: false

## warpax.adm

::: warpax.adm
    options:
      show_root_heading: false

## warpax.constraints

::: warpax.constraints
    options:
      show_root_heading: false

## warpax.tov

::: warpax.tov
    options:
      show_root_heading: false

## warpax.transport

::: warpax.transport
    options:
      show_root_heading: false

## warpax.classify

::: warpax.classify
    options:
      show_root_heading: false

## warpax.io

::: warpax.io
    options:
      show_root_heading: false

## warpax.optimization

::: warpax.optimization
    options:
      show_root_heading: false

## warpax.design

::: warpax.design
    options:
      show_root_heading: false

## warpax.visualization

The matplotlib plot functions live in the submodules below and are re-exported
lazily from `warpax.visualization`, so importing the package does not import
matplotlib.

### warpax.visualization.comparison_plots

::: warpax.visualization.comparison_plots
    options:
      show_root_heading: false
      members: [plot_comparison_panel, plot_comparison_table, plot_velocity_sweep]

### warpax.visualization.convergence_plots

::: warpax.visualization.convergence_plots
    options:
      show_root_heading: false
      members: [plot_convergence, plot_convergence_table]

### warpax.visualization.kinematic_plots

::: warpax.visualization.kinematic_plots
    options:
      show_root_heading: false
      members: [plot_kinematic_scalars]

### warpax.visualization.geodesic_plots

::: warpax.visualization.geodesic_plots
    options:
      show_root_heading: false
      members: [plot_blueshift_profile, plot_tidal_evolution]

### warpax.visualization.phase_diagram

::: warpax.visualization.phase_diagram
    options:
      show_root_heading: false
      members: [plot_phase_diagram, plot_phase_summary]

### warpax.visualization.shift_vorticity_plots

::: warpax.visualization.shift_vorticity_plots
    options:
      show_root_heading: false
      members: [plot_shift_vorticity]

## warpax.numerics

::: warpax.numerics
    options:
      show_root_heading: false
