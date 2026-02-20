"""Visualization tools for energy condition analysis.

Modules
-------
- **margin_plots**: Energy condition margin heatmaps (diverging colormap).
- **direction_fields**: Worst-observer boost direction quiver plots.
- **report**: Combined multi-panel report generation.
- **comparison_plots**: Eulerian vs robust EC comparison figures.
- **convergence_plots**: Richardson convergence log-log plots and tables.
- **kinematic_plots**: Expansion/shear/vorticity scalar field maps.
- **geodesic_plots**: Tidal eigenvalue evolution and blueshift profile plots.
- **alignment_plots**: Worst-observer alignment angle histograms.
"""
from __future__ import annotations

from .comparison_plots import (
    plot_comparison_grid,
    plot_comparison_panel,
    plot_comparison_table,
    plot_velocity_sweep,
)
from .convergence_plots import plot_convergence, plot_convergence_table
from .geodesic_plots import plot_blueshift_profile, plot_tidal_evolution
from .kinematic_plots import plot_kinematic_comparison, plot_kinematic_scalars
from .alignment_plots import plot_alignment_histogram

__all__ = [
    # comparison_plots
    "plot_comparison_grid",
    "plot_comparison_panel",
    "plot_comparison_table",
    "plot_velocity_sweep",
    # convergence_plots
    "plot_convergence",
    "plot_convergence_table",
    # geodesic_plots
    "plot_blueshift_profile",
    "plot_tidal_evolution",
    # kinematic_plots
    "plot_kinematic_comparison",
    "plot_kinematic_scalars",
    # alignment_plots
    "plot_alignment_histogram",
]
