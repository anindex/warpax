"""Time-dependent physics: velocity profiles and frame sequence builder.

Provides velocity profile callables for warp drive animations and a
frame sequence builder that produces lists of FrameData from velocity
sweeps or time-varying v_s(t).

All velocity profiles use NumPy (not JAX) since they are evaluated
outside the JAX trace boundary. The frame sequence builder uses
``eqx.tree_at`` to swap ``v_s`` without triggering JIT recompilation.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Velocity profiles (pure Python/NumPy NOT JAX)
# ---------------------------------------------------------------------------


def linear_ramp(
    t: float,
    v_start: float = 0.1,
    v_end: float = 0.99,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> float:
    """Linear velocity ramp from v_start to v_end over [t_start, t_end].

    Parameters
    ----------
    t : float
        Time value.
    v_start, v_end : float
        Start and end velocities.
    t_start, t_end : float
        Start and end times.

    Returns
    -------
    float
        Velocity at time t, clamped to [min(v_start, v_end), max(v_start, v_end)].
    """
    if t_end == t_start:
        return float(v_end)
    frac = (t - t_start) / (t_end - t_start)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(v_start + frac * (v_end - v_start))


def sigmoid_ramp(
    t: float,
    v_start: float = 0.1,
    v_end: float = 0.99,
    t_mid: float = 0.5,
    steepness: float = 10.0,
) -> float:
    """Sigmoid velocity ramp centered at t_mid.

    Parameters
    ----------
    t : float
        Time value.
    v_start, v_end : float
        Start and end velocities.
    t_mid : float
        Time at which v_s is halfway between v_start and v_end.
    steepness : float
        Controls transition sharpness (larger = sharper).

    Returns
    -------
    float
        Velocity at time t.
    """
    s = float(1.0 / (1.0 + np.exp(-steepness * (t - t_mid))))
    val = v_start + s * (v_end - v_start)
    lo = min(v_start, v_end)
    hi = max(v_start, v_end)
    return float(np.clip(val, lo, hi))


def collapse_profile(
    t: float,
    v_max: float = 0.5,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> float:
    """Collapse profile: v_s ramps DOWN from v_max to 0.

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


def constant_velocity(t: float, v_s: float = 0.5) -> float:
    """Constant velocity profile (trivial, for API consistency).

    Parameters
    ----------
    t : float
        Time value (ignored).
    v_s : float
        Constant velocity.

    Returns
    -------
    float
        Always returns v_s.
    """
    return float(v_s)


# ---------------------------------------------------------------------------
# Velocity sweep convenience
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Frame sequence builder
# ---------------------------------------------------------------------------


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

    Uses ``eqx.tree_at`` to swap ``v_s`` without JIT recompilation.

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

    # Validate inputs
    if v_s_fn is not None and v_s_values is not None:
        raise ValueError("Provide exactly one of v_s_fn or v_s_values, not both.")
    if v_s_fn is None and v_s_values is None:
        raise ValueError("Provide exactly one of v_s_fn or v_s_values.")
    if v_s_fn is not None and t_values is None:
        raise ValueError("t_values is required when v_s_fn is provided.")

    # Build frame schedule: list of (v_s, t) pairs
    if v_s_values is not None:
        if t_values is None:
            t_values = [0.0] * len(v_s_values)
        schedule = list(zip(v_s_values, t_values))
    else:
        schedule = [(v_s_fn(float(t)), float(t)) for t in t_values]

    # Optional progress bar
    iterator = schedule
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(schedule, desc="Building frames", unit="frame")
        except ImportError:
            pass

    frames = []
    metric_name = metric.name()
    for v_s_t, t in iterator:
        # Swap v_s without recompilation
        metric_t = eqx.tree_at(lambda m: m.v_s, metric, v_s_t)
        result = evaluate_curvature_grid(
            metric_t, grid_spec,
            batch_size=batch_size,
            compute_invariants=compute_invariants,
            t=t,
        )
        frame = freeze_curvature(
            result, grid_spec,
            metric_name=metric_name,
            v_s=v_s_t,
            t=t,
        )
        frames.append(frame)

    return frames


# ---------------------------------------------------------------------------
# EC-enriched frame sequence builder
# ---------------------------------------------------------------------------


def build_ec_frame_sequence(
    metric,
    grid_spec,
    *,
    v_s_fn: Callable[[float], float] | None = None,
    v_s_values: list[float] | None = None,
    t_values: list[float] | None = None,
    observer_params=None,
    batch_size: int | None = None,
    progress: bool = True,
) -> list:
    """Build frame sequence with observer sweep EC margins.

    Like ``build_frame_sequence`` but additionally computes observer-swept
    WEC and NEC margins at each frame, producing FrameData with fields:
    ``wec_margin_sweep``, ``nec_margin_sweep``, and ``energy_density``.

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
        Observer parameters (K, 3) for sweep. If None, uses default
        ``make_rapidity_observers()`` (36 observers).
    batch_size : int or None
        Chunk size for grid evaluation.
    progress : bool
        Show tqdm progress bar.

    Returns
    -------
    list[FrameData]
        FrameData with curvature invariants + EC sweep margin fields.
    """
    import equinox as eqx
    import jax.numpy as jnp

    from warpax.energy_conditions.sweep import (
        make_rapidity_observers,
        sweep_nec_margins,
        sweep_wec_margins,
    )
    from warpax.geometry.grid import evaluate_curvature_grid
    from warpax.visualization.common._conversion import _symmetric_clim
    from warpax.visualization.common._frame_data import FrameData

    # Validate inputs
    if v_s_fn is not None and v_s_values is not None:
        raise ValueError("Provide exactly one of v_s_fn or v_s_values, not both.")
    if v_s_fn is None and v_s_values is None:
        raise ValueError("Provide exactly one of v_s_fn or v_s_values.")
    if v_s_fn is not None and t_values is None:
        raise ValueError("t_values is required when v_s_fn is provided.")

    # Default observer parameters
    if observer_params is None:
        observer_params = make_rapidity_observers()

    # Build frame schedule
    if v_s_values is not None:
        if t_values is None:
            t_values = [0.0] * len(v_s_values)
        schedule = list(zip(v_s_values, t_values))
    else:
        schedule = [(v_s_fn(float(t)), float(t)) for t in t_values]

    # Optional progress bar
    iterator = schedule
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(schedule, desc="Building EC frames", unit="frame")
        except ImportError:
            pass

    # Extract grid coordinates as NumPy
    X, Y, Z = grid_spec.meshgrid
    x_np = np.asarray(X)
    y_np = np.asarray(Y)
    z_np = np.asarray(Z)

    # NEC params: (theta, phi) columns
    nec_params = observer_params[:, 1:]

    frames = []
    metric_name = metric.name()
    for v_s_t, t in iterator:
        metric_t = eqx.tree_at(lambda m: m.v_s, metric, v_s_t)
        result = evaluate_curvature_grid(
            metric_t, grid_spec,
            batch_size=batch_size,
            compute_invariants=True,
            t=t,
        )

        # Flatten for sweep
        N = int(np.prod(grid_spec.shape))
        T_flat = result.stress_energy.reshape(N, 4, 4)
        g_flat = result.metric.reshape(N, 4, 4)

        # Compute sweep margins
        wec_margins = sweep_wec_margins(T_flat, g_flat, observer_params)  # (N, K)
        nec_margins = sweep_nec_margins(T_flat, g_flat, nec_params)  # (N, K)

        # Worst-case across observers
        worst_wec = np.asarray(jnp.min(wec_margins, axis=-1).reshape(grid_spec.shape))
        worst_nec = np.asarray(jnp.min(nec_margins, axis=-1).reshape(grid_spec.shape))
        energy_density = np.asarray(result.stress_energy[..., 0, 0])

        # Curvature invariants
        scalar_fields = {
            "ricci_scalar": np.asarray(result.ricci_scalar),
            "kretschner": np.asarray(result.kretschner),
            "ricci_squared": np.asarray(result.ricci_squared),
            "weyl_squared": np.asarray(result.weyl_squared),
            "energy_density": energy_density,
            "wec_margin_sweep": worst_wec,
            "nec_margin_sweep": worst_nec,
        }

        # Colormaps
        colormaps = {
            "ricci_scalar": "RdBu_r",
            "kretschner": "inferno",
            "ricci_squared": "inferno",
            "weyl_squared": "inferno",
            "energy_density": "RdBu_r",
            "wec_margin_sweep": "RdBu_r",
            "nec_margin_sweep": "RdBu_r",
        }

        # Color limits
        clim = {}
        for name, arr in scalar_fields.items():
            if name in {"ricci_scalar", "energy_density", "wec_margin_sweep", "nec_margin_sweep"}:
                clim[name] = _symmetric_clim(arr)
            else:
                vmin = float(np.nanmin(arr))
                vmax = float(np.nanmax(arr))
                if abs(vmax - vmin) < 1e-15:
                    vmax = vmin + 1.0
                clim[name] = (vmin, vmax)

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
