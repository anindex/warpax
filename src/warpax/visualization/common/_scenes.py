"""Scene builders producing lists of FrameData for the animations.

- ``scene_velocity_ramp``: EC violations intensify as v_s sweeps 0.1 to 0.99.
- ``scene_observer_sweep``: Per-rapidity EC margins on fixed Alcubierre geometry.
"""

from __future__ import annotations

import numpy as np


def scene_velocity_ramp(
    grid_spec,
    *,
    n_frames: int = 90,
    v_start: float = 0.1,
    v_end: float = 0.99,
) -> list:
    """Build EC-enriched frame sequence for velocity ramp-up.

    The warp bubble velocity sweeps from *v_start* to *v_end*, producing
    frames with observer-swept WEC and NEC margins that show violations
    intensifying with speed.

    Parameters
    ----------
    grid_spec : GridSpec
        Spatial grid specification.
    n_frames : int
        Number of animation frames (default 90).
    v_start : float
        Starting velocity (default 0.1).
    v_end : float
        Ending velocity (default 0.99).

    Returns
    -------
    list[FrameData]
        Frame sequence with wec_margin_sweep and nec_margin_sweep fields.
    """
    from warpax.benchmarks import AlcubierreMetric

    from ._physics import build_ec_frame_sequence, make_velocity_sweep

    metric = AlcubierreMetric(v_s=v_start)
    v_values = make_velocity_sweep(v_start, v_end, n_frames)

    return build_ec_frame_sequence(
        metric,
        grid_spec,
        v_s_values=v_values,
        progress=True,
    )


def scene_observer_sweep(
    grid_spec,
    *,
    n_frames: int = 60,
    v_s: float = 0.5,
    zeta_max: float = 5.0,
    n_directions: int = 3,
) -> list:
    """Build per-rapidity EC margin frames on fixed Alcubierre geometry.

    Unlike ``build_ec_frame_sequence`` which takes worst-case across ALL
    observers, this scene shows the WEC/NEC margin landscape for a SINGLE
    rapidity at each frame. The geometry is computed once and reused,
    revealing how the violation landscape shifts with observer boost. The
    NEC null contraction is rapidity-independent, so it is computed once
    outside the per-frame loop. Each frame's ``FrameData.t`` carries the
    rapidity for annotation. Only the Eulerian energy density is one-sided
    here; the signed WEC and NEC margins get a symmetric clim, because a
    one-sided ramp pinned at vmax = 0 would paint every satisfied point the
    color of an exactly saturated one.

    Parameters
    ----------
    grid_spec : GridSpec
        Spatial grid specification.
    n_frames : int
        Number of animation frames (default 60).
    v_s : float
        Fixed warp velocity (default 0.5).
    zeta_max : float
        Maximum rapidity, matching the paper's cap (default 5.0).
    n_directions : int
        Number of boost directions per rapidity (default 3: +x, +y, +z).

    Returns
    -------
    list[FrameData]
        Frame sequence where each frame's ``t`` field carries the rapidity
        value for annotation (geometry is identical across frames).
    """
    import jax.numpy as jnp

    from warpax.benchmarks import AlcubierreMetric
    from warpax.energy_conditions.sweep import sweep_nec_margins, sweep_wec_margins
    from warpax.geometry.grid import evaluate_curvature_grid

    from ._conversion import (
        _oneside_neg_clim,
        _symmetric_clim,
        eulerian_energy_density_grid,
    )
    from ._frame_data import FrameData

    metric = AlcubierreMetric(v_s=v_s)

    result = evaluate_curvature_grid(
        metric,
        grid_spec,
        compute_invariants=True,
    )

    N = int(np.prod(grid_spec.shape))
    T_flat = result.stress_energy.reshape(N, 4, 4)
    g_flat = result.metric.reshape(N, 4, 4)
    energy_density = eulerian_energy_density_grid(result.stress_energy, result.metric_inv)
    T_00_covariant = np.asarray(result.stress_energy[..., 0, 0])

    X, Y, Z = grid_spec.meshgrid
    x_np = np.asarray(X)
    y_np = np.asarray(Y)
    z_np = np.asarray(Z)

    all_directions = [
        (float(np.pi / 2), 0.0),
        (float(np.pi / 2), float(np.pi / 2)),
        (0.0, 0.0),
    ]
    directions = all_directions[:n_directions]

    nec_params = jnp.array(directions)
    nec_margins = sweep_nec_margins(T_flat, g_flat, nec_params)
    worst_nec = np.asarray(jnp.min(nec_margins, axis=-1).reshape(grid_spec.shape))

    rapidities = np.linspace(0.0, zeta_max, n_frames)

    iterator = list(rapidities)
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Observer sweep", unit="frame")
    except ImportError:
        pass

    colormaps = {
        "energy_density": "nec_depth",
        "T_00_covariant": "RdBu_r",
        "wec_margin_sweep": "RdBu_r",
        "nec_margin_sweep": "RdBu_r",
    }
    static_clim = {
        "energy_density": _oneside_neg_clim(energy_density),
        "T_00_covariant": _symmetric_clim(T_00_covariant),
        "nec_margin_sweep": _symmetric_clim(worst_nec),
    }

    frames = []
    metric_name = metric.name()

    for zeta in iterator:
        obs_params = jnp.array([[float(zeta), theta, phi] for theta, phi in directions])
        wec_margins = sweep_wec_margins(T_flat, g_flat, obs_params)
        worst_wec = np.asarray(jnp.min(wec_margins, axis=-1).reshape(grid_spec.shape))

        scalar_fields = {
            "energy_density": energy_density,
            "T_00_covariant": T_00_covariant,
            "wec_margin_sweep": worst_wec,
            "nec_margin_sweep": worst_nec,
        }
        clim = {**static_clim, "wec_margin_sweep": _symmetric_clim(worst_wec)}

        frame = FrameData(
            x=x_np,
            y=y_np,
            z=z_np,
            scalar_fields=scalar_fields,
            metric_name=metric_name,
            v_s=v_s,
            grid_shape=grid_spec.shape,
            t=float(zeta),
            colormaps=colormaps,
            clim=clim,
        )
        frames.append(frame)

    return frames
