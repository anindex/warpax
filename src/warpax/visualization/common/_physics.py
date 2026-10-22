"""Time-dependent physics: velocity profiles and frame sequence builder.

All velocity profiles use NumPy (not JAX) since they are evaluated
outside the JAX trace boundary. The frame sequence builder uses
``eqx.tree_at`` to swap ``v_s`` without triggering JIT recompilation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _shape_function_grid(metric, grid_spec, t):
    """Evaluate the shape function f(r_s) on the grid, or None if undefined.

    Lets scenes contour the *real* ``f = 0.5`` bubble wall instead of a
    hard-coded circular fallback. Metrics without ``shape_function_value``
    (no closed-form wall) return ``None``.
    """
    if not hasattr(metric, "shape_function_value"):
        return None
    import jax

    from warpax.geometry.grid import build_coord_batch

    coords_flat = build_coord_batch(grid_spec, t=t)
    f_flat = jax.vmap(metric.shape_function_value)(coords_flat)
    return np.asarray(f_flat).reshape(grid_spec.shape)


def rampdown_profile(
    t: float,
    v_max: float = 0.5,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> float:
    """Velocity turn-off profile: v_s ramps down from v_max to 0.

    Parameters
    ----------
    t : float
        Time value.
    v_max : float
        Maximum velocity at t_start.
    t_start, t_end : float
        Start and end times.

    Returns
    -------
    float
        Velocity at time t, clamped to [0, v_max].
    """
    if t_end == t_start:
        return 0.0
    frac = (t - t_start) / (t_end - t_start)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(np.clip(v_max * (1.0 - frac), 0.0, v_max))


def make_velocity_sweep(
    v_start: float = 0.1,
    v_end: float = 0.99,
    n_frames: int = 60,
    spacing: str = "uniform",
) -> list[float]:
    """Generate a list of velocity values for a sweep animation.

    Parameters
    ----------
    v_start, v_end : float
        Start and end velocities.
    n_frames : int
        Number of frames.
    spacing : str
        ``'uniform'`` for linear spacing, ``'log'`` for logarithmic.

    Returns
    -------
    list[float]
        List of n_frames velocity values.

    Raises
    ------
    ValueError
        If spacing is not ``'uniform'`` or ``'log'``.
    """
    if spacing == "uniform":
        vals = np.linspace(v_start, v_end, n_frames)
    elif spacing == "log":
        vals = np.logspace(np.log10(v_start), np.log10(v_end), n_frames)
    else:
        raise ValueError(f"Unknown spacing: {spacing!r}. Use 'uniform' or 'log'.")
    return [float(v) for v in vals]


def _frame_schedule(
    v_s_fn: Callable[[float], float] | None,
    v_s_values: list[float] | None,
    t_values: list[float] | None,
    progress: bool,
    desc: str,
):
    """Validate the sweep inputs and return an iterable of ``(v_s, t)`` pairs.

    Exactly one of *v_s_fn* and *v_s_values* must be given; *t_values* is
    required with *v_s_fn* and defaults to zeros with *v_s_values*. With
    *progress* the iterable is a tqdm bar when tqdm is installed.
    """
    if v_s_fn is not None and v_s_values is not None:
        raise ValueError("Provide exactly one of v_s_fn or v_s_values, not both.")
    if v_s_fn is None and v_s_values is None:
        raise ValueError("Provide exactly one of v_s_fn or v_s_values.")
    if v_s_fn is not None and t_values is None:
        raise ValueError("t_values is required when v_s_fn is provided.")

    if v_s_values is not None:
        if t_values is None:
            t_values = [0.0] * len(v_s_values)
        elif len(t_values) != len(v_s_values):
            raise ValueError(
                f"v_s_values and t_values length mismatch: {len(v_s_values)} != {len(t_values)}."
            )
        schedule = list(zip(v_s_values, t_values, strict=True))
    else:
        schedule = [(v_s_fn(float(t)), float(t)) for t in t_values]

    if progress:
        try:
            from tqdm.auto import tqdm

            return tqdm(schedule, desc=desc, unit="frame")
        except ImportError:
            pass
    return schedule


def build_frame_sequence(
    metric,
    grid_spec,
    *,
    v_s_fn: Callable[[float], float] | None = None,
    v_s_values: list[float] | None = None,
    t_values: list[float] | None = None,
    compute_invariants: bool = True,
    batch_size: int | None = None,
    progress: bool = True,
) -> list:
    """Build a sequence of FrameData from velocity profiles or explicit values.

    Each frame evaluates a separate constant-speed metric. This is a parameter
    sweep, not an accelerating solution; derivatives of ``v_s_fn`` do not enter
    the curvature. Uses ``eqx.tree_at`` to swap ``v_s``.

    Parameters
    ----------
    metric : MetricSpecification
        Base warp metric (e.g. ``AlcubierreMetric(v_s=0.1)``).
    grid_spec : GridSpec
        Grid specification for curvature evaluation.
    v_s_fn : callable, optional
        Time-dependent velocity function ``v_s(t) -> float``.
        Mutually exclusive with ``v_s_values``.
    v_s_values : list[float], optional
        Explicit list of velocity values for each frame.
        Mutually exclusive with ``v_s_fn``.
    t_values : list[float], optional
        Time values for each frame. Required if ``v_s_fn`` is provided.
        If ``v_s_values`` is provided without ``t_values``, uses ``t=0.0``
        for all frames.
    compute_invariants : bool
        Whether to compute curvature invariants (default True).
    batch_size : int or None
        Chunk size for memory-safe grid evaluation.
    progress : bool
        Show tqdm progress bar (default True).

    Returns
    -------
    list[FrameData]
        One FrameData per frame with curvature invariants as scalar fields.

    Raises
    ------
    ValueError
        If both or neither of ``v_s_fn`` and ``v_s_values`` are provided,
        or if ``v_s_fn`` is provided without ``t_values``.
    """
    import equinox as eqx

    from warpax.geometry.grid import evaluate_curvature_grid
    from warpax.visualization.common._conversion import freeze_curvature

    iterator = _frame_schedule(v_s_fn, v_s_values, t_values, progress, "Building frames")

    frames = []
    metric_name = metric.name()
    for v_s_t, t in iterator:
        metric_t = eqx.tree_at(lambda m: m.v_s, metric, v_s_t)
        result = evaluate_curvature_grid(
            metric_t,
            grid_spec,
            batch_size=batch_size,
            compute_invariants=compute_invariants,
            t=t,
        )
        frame = freeze_curvature(
            result,
            grid_spec,
            metric_name=metric_name,
            v_s=v_s_t,
            t=t,
        )
        frames.append(frame)

    return frames


def build_ec_frame_sequence(
    metric,
    grid_spec,
    *,
    v_s_fn: Callable[[float], float] | None = None,
    v_s_values: list[float] | None = None,
    t_values: list[float] | None = None,
    observer_params=None,
    nec_observer_params=None,
    batch_size: int | None = None,
    progress: bool = True,
    compute_wec: bool = True,
) -> list:
    """Build frame sequence with observer sweep EC margins.

    Like ``build_frame_sequence`` but additionally computes observer-swept
    WEC and NEC margins at each frame, producing FrameData with fields:
    ``wec_margin_sweep``, ``nec_margin_sweep``, and ``energy_density``.
    The inline expansion ``theta`` mirrors
    ``warpax.analysis.kinematic_scalars.expansion`` and must stay identical
    to it (``tests/test_visualization.py`` cross-checks the two).

    Parameters
    ----------
    metric : MetricSpecification
        Base warp metric.
    grid_spec : GridSpec
        Grid specification for curvature evaluation.
    v_s_fn : callable, optional
        Time-dependent velocity function.
    v_s_values : list[float], optional
        Explicit velocity values per frame.
    t_values : list[float], optional
        Time values per frame.
    observer_params : jax.Array or None
        Timelike observer parameters (K, 3) for the WEC sweep. If None,
        uses ``make_rapidity_observers`` (36 observers).
    nec_observer_params : jax.Array or None
        Null direction parameters (K, 3) for the NEC sweep. If None, uses a
        dense angular sampler (``make_angular_observers``, 312 directions).
        The NEC null contraction is rapidity-independent, so sampling only a
        few axis-aligned directions (as the rapidity observers do) produces
        spurious "satisfied" pixels; a dense null sphere is required.
    batch_size : int or None
        Chunk size for grid evaluation.
    progress : bool
        Show tqdm progress bar.
    compute_wec : bool
        Run the timelike WEC sweep (default True). With False the
        ``wec_margin_sweep`` field is omitted, which skips the most expensive
        step for scenes that only need theta, the energy density and the NEC.

    Returns
    -------
    list[FrameData]
        FrameData with curvature invariants + EC sweep margin fields.
    """
    import equinox as eqx
    import jax.numpy as jnp

    from warpax.energy_conditions.sweep import (
        make_angular_observers,
        make_rapidity_observers,
        sweep_nec_margins,
        sweep_wec_margins,
    )
    from warpax.geometry.grid import evaluate_curvature_grid
    from warpax.visualization.common._conversion import (
        DIVERGING_FIELDS,
        _magnitude_clim,
        _symmetric_clim,
        eulerian_energy_density_grid,
    )
    from warpax.visualization.common._frame_data import FrameData

    iterator = _frame_schedule(v_s_fn, v_s_values, t_values, progress, "Building EC frames")

    if compute_wec and observer_params is None:
        observer_params = make_rapidity_observers()
    if nec_observer_params is None:
        nec_observer_params = make_angular_observers(n_theta=13, n_phi=24)

    X, Y, Z = grid_spec.meshgrid
    x_np = np.asarray(X)
    y_np = np.asarray(Y)
    z_np = np.asarray(Z)

    nec_params = nec_observer_params[:, 1:]

    frames = []
    metric_name = metric.name()
    for v_s_t, t in iterator:
        metric_t = eqx.tree_at(lambda m: m.v_s, metric, v_s_t)
        result = evaluate_curvature_grid(
            metric_t,
            grid_spec,
            batch_size=batch_size,
            compute_invariants=True,
            t=t,
        )

        N = int(np.prod(grid_spec.shape))
        T_flat = result.stress_energy.reshape(N, 4, 4)
        g_flat = result.metric.reshape(N, 4, 4)

        nec_margins = sweep_nec_margins(T_flat, g_flat, nec_params)
        worst_nec = np.asarray(jnp.min(nec_margins, axis=-1).reshape(grid_spec.shape))
        energy_density = eulerian_energy_density_grid(result.stress_energy, result.metric_inv)
        alpha = 1.0 / jnp.sqrt(-result.metric_inv[..., 0, 0])
        spatial_inv = jnp.linalg.inv(result.metric[..., 1:, 1:])
        theta = np.asarray(
            alpha * jnp.einsum("...ij,...ij->...", spatial_inv, result.christoffel[..., 0, 1:, 1:])
        )
        T_00_covariant = np.asarray(result.stress_energy[..., 0, 0])
        f_grid = _shape_function_grid(metric_t, grid_spec, t)

        scalar_fields = {
            "ricci_scalar": np.asarray(result.ricci_scalar),
            "kretschmann": np.asarray(result.kretschmann),
            "ricci_squared": np.asarray(result.ricci_squared),
            "weyl_squared": np.asarray(result.weyl_squared),
            "energy_density": energy_density,
            "theta": theta,
            "T_00_covariant": T_00_covariant,
            "nec_margin_sweep": worst_nec,
        }
        if compute_wec:
            wec_margins = sweep_wec_margins(T_flat, g_flat, observer_params)
            scalar_fields["wec_margin_sweep"] = np.asarray(
                jnp.min(wec_margins, axis=-1).reshape(grid_spec.shape)
            )

        colormaps = dict.fromkeys(scalar_fields, "RdBu_r")
        if f_grid is not None:
            scalar_fields["shape_function"] = f_grid
            colormaps["shape_function"] = "viridis"

        clim = {
            name: _symmetric_clim(arr) if name in DIVERGING_FIELDS else _magnitude_clim(arr)
            for name, arr in scalar_fields.items()
        }

        frame = FrameData(
            x=x_np,
            y=y_np,
            z=z_np,
            scalar_fields=scalar_fields,
            metric_name=metric_name,
            v_s=v_s_t,
            grid_shape=grid_spec.shape,
            t=t,
            colormaps=colormaps,
            clim=clim,
        )
        frames.append(frame)

    return frames
