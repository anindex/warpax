"""Visualization tools for energy condition analysis.

Modules
-------
- **comparison_plots**: Eulerian vs robust EC comparison figures.
- **convergence_plots**: Richardson convergence log-log plots and tables.
- **kinematic_plots**: Expansion/shear/vorticity scalar field maps.
- **geodesic_plots**: Tidal eigenvalue evolution and blueshift profile plots.
- **alignment_plots**: Worst-observer alignment angle histograms.
- **phase_diagram**: EC-admissible transport phase diagrams.
- **shift_vorticity_plots**: Shift-vorticity control of the Hawking-Ellis type.

The matplotlib-backed plot functions are imported lazily (PEP 562 ``__getattr__``)
so that importing the sibling ``common``/``manim`` layers -- or running ``--help``
on the CLI scripts -- does not eagerly import matplotlib. matplotlib needs a
writable cache (``MPLCONFIGDIR``) at import time, which is unavailable in some
headless/sandboxed contexts; deferring the import keeps the package usable there.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Public plot function -> defining submodule. Resolved on first access.
_LAZY = {
    "plot_comparison_grid": "comparison_plots",
    "plot_comparison_panel": "comparison_plots",
    "plot_comparison_table": "comparison_plots",
    "plot_velocity_sweep": "comparison_plots",
    "plot_convergence": "convergence_plots",
    "plot_convergence_table": "convergence_plots",
    "plot_blueshift_profile": "geodesic_plots",
    "plot_tidal_evolution": "geodesic_plots",
    "plot_kinematic_comparison": "kinematic_plots",
    "plot_kinematic_scalars": "kinematic_plots",
    "plot_alignment_histogram": "alignment_plots",
    "plot_phase_diagram": "phase_diagram",
    "plot_phase_summary": "phase_diagram",
    "plot_shift_vorticity": "shift_vorticity_plots",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f".{module}", __name__)
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # static-analysis only; never executed at import time
    from .alignment_plots import plot_alignment_histogram
    from .comparison_plots import (
        plot_comparison_grid,
        plot_comparison_panel,
        plot_comparison_table,
        plot_velocity_sweep,
    )
    from .convergence_plots import plot_convergence, plot_convergence_table
    from .geodesic_plots import plot_blueshift_profile, plot_tidal_evolution
    from .kinematic_plots import plot_kinematic_comparison, plot_kinematic_scalars
    from .phase_diagram import plot_phase_diagram, plot_phase_summary
    from .shift_vorticity_plots import plot_shift_vorticity
