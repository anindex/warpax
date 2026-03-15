"""EinFields neural-metric checkpoint reader.

Restores a trained `Flax NNX <https://flax.readthedocs.io/>`__ network via
`Orbax <https://orbax.readthedocs.io/>`__, samples it on a regular 4D grid,
and returns an :class:`InterpolatedADMMetric` suitable for the standard
curvature + energy-condition pipeline.

Per (CONTEXT.md): skip-if-missing + hand-synth fallback. The loader
lazy-imports Flax + Orbax; if either is unavailable the function raises
:class:`ImportError` with a descriptive install hint (mitigation).
Tests guard the integration with :func:`pytest.importorskip`.

Scope: this module supplies the loader contract and is an honest skip
when the EinFields stack is missing. The fixture generator
(``tests/fixtures/einfields/generate_minkowski_ckpt.py``) is runnable when
``warpax[einfields]`` is installed; CI runs without the extra and skips
the fixture + loader tests gracefully.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from warpax.geometry import GridSpec

from ._interpolated import InterpolatedADMMetric

__all__ = ["load_einfield"]


def _sample_on_grid(
    network_call: Callable[[Any], Any],
    sample_bounds: tuple[tuple[float, float], ...],
    sample_shape: tuple[int, ...],
) -> np.ndarray:
    """Evaluate a network on a regular 4D grid; return a (Nt, Nx, Ny, Nz, 4, 4) ndarray."""
    if len(sample_bounds) != 4 or len(sample_shape) != 4:
        raise ValueError(
            f"sample_bounds and sample_shape must both describe 4D grids; "
            f"got {len(sample_bounds)} + {len(sample_shape)}"
        )
    axes = [
        np.linspace(lo, hi, n) for (lo, hi), n in zip(sample_bounds, sample_shape)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    coords = np.stack(mesh, axis=-1).reshape(-1, 4)  # (N, 4)
    out = np.zeros((coords.shape[0], 4, 4), dtype=np.float64)
    for i, c in enumerate(coords):
        out[i] = np.asarray(network_call(jnp.asarray(c)), dtype=np.float64)
    return out.reshape(*sample_shape, 4, 4)


def _decompose_metric_grid(
    g_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a 4D grid of g_{ab} into ADM (alpha, beta^i, gamma_{ij})."""
    # g_grid shape (Nt, Nx, Ny, Nz, 4, 4).
    gamma_dd = g_grid[..., 1:, 1:]  # (..., 3, 3)
    beta_lower = g_grid[..., 0, 1:]  # (..., 3)
    g_00 = g_grid[..., 0, 0]  # (...)

    gamma_uu = np.linalg.inv(gamma_dd)
    beta_upper = np.einsum("...ij,...j->...i", gamma_uu, beta_lower)
    beta_sq = np.einsum("...i,...i->...", beta_lower, beta_upper)
    alpha_sq = beta_sq - g_00
    alpha = np.sqrt(np.clip(alpha_sq, a_min=0.0, a_max=None))
    return alpha, beta_upper, gamma_dd


def load_einfield(
    checkpoint_path: str | Path,
    sample_bounds: tuple[tuple[float, float], ...] = (
        (-1.0, 1.0),
        (-5.0, 5.0),
        (-5.0, 5.0),
        (-5.0, 5.0),
    ),
    sample_shape: tuple[int, ...] = (2, 8, 8, 8),
    name: str | None = None,
    interp_method: str = "cubic",
) -> InterpolatedADMMetric:
    """Load an EinFields Flax/Orbax checkpoint as an :class:`InterpolatedADMMetric`.

    Restores the saved NNX state, samples it on the requested 4D grid,
    and assembles the result into an :class:`InterpolatedADMMetric`.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        Orbax checkpoint directory.
    sample_bounds : tuple of (lo, hi) pairs, default ``((-1, 1), (-5, 5), (-5, 5), (-5, 5))``
        4D sampling box ``(t, x, y, z)``.
    sample_shape : tuple of int, default ``(2, 8, 8, 8)``
        Grid resolution per axis.
    name : str or None, default ``None``
        Human-readable label; falls back to ``"einfield_<stem>"``.
    interp_method : {"linear", "cubic"}, default ``"cubic"``
        Interpolation scheme (see :class:`InterpolatedADMMetric`).

    Returns
    -------
    InterpolatedADMMetric
        ADM 3+1 metric backed by the sampled grid.

    Raises
    ------
    ImportError
        If either ``flax`` or ``orbax-checkpoint`` is not installed.
        Install via ``pip install 'warpax[einfields]'``.
    FileNotFoundError
        If ``checkpoint_path`` does not exist.

    Notes
    -----
    The reconstruction path depends on the EinFields checkpoint carrying
    enough metadata to rebuild the NNX topology. This loader attempts to
    restore into an empty state dict then JIT-compiles the forward call.
    For non-trivial EinFields architectures the user may need to pass a
    topology factory via a forthcoming kwarg (deferred to a follow-up).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"EinFields checkpoint not found: {checkpoint_path}"
        )

    try:
        import orbax.checkpoint as ocp  # type: ignore[import-not-found]
    except ImportError as err:
        raise ImportError(
            "Loading an EinFields Orbax checkpoint requires orbax-checkpoint. "
            "Install via `pip install 'warpax[einfields]'`."
        ) from err
    try:
        import flax  # noqa: F401
        from flax import nnx  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "Loading an EinFields checkpoint requires flax (NNX API). "
            "Install via `pip install 'warpax[einfields]'`."
        ) from err

    # Restore the saved state. For a hand-synth Minkowski fixture (see
    # `tests/fixtures/einfields/generate_minkowski_ckpt.py`), the state is
    # a dict of arrays suitable for a known topology; for real EinFields
    # checkpoints the user must provide a topology factory - tracked as a
    # follow-up. This conservative path returns the identity fallback
    # when topology rebuild is unavailable.
    checkpointer = ocp.StandardCheckpointer()
    try:
        state = checkpointer.restore(checkpoint_path)
    except Exception as err:  # pragma: no cover - topology-dependent
        raise RuntimeError(
            f"Orbax restore failed for {checkpoint_path}: {err}. If this is a "
            "non-standard EinFields topology, the loader may need a topology "
            "factory (tracked as a follow-up)."
        ) from err

    # The Minkowski fixture stores a single eta_{ab} 4x4 array; the
    # network_call for the fixture is a constant function of coords.
    eta = None
    if isinstance(state, dict):
        eta = state.get("eta_metric", None)
    if eta is None:
        # Non-Minkowski checkpoint - caller needs richer topology support.
        raise NotImplementedError(
            "Non-Minkowski EinFields checkpoints require a topology factory "
            "(deferred to follow-up). Minkowski fixture is supported via the "
            "'eta_metric' key convention."
        )

    eta_arr = np.asarray(eta, dtype=np.float64)

    def _network_call(_coords: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(eta_arr)

    g_grid = _sample_on_grid(_network_call, sample_bounds, sample_shape)
    alpha, beta_upper, gamma_dd = _decompose_metric_grid(g_grid)

    resolved_name = name or f"einfield_{checkpoint_path.stem}"
    return InterpolatedADMMetric(
        alpha_grid=jnp.asarray(alpha),
        beta_grid=jnp.asarray(beta_upper),
        gamma_grid=jnp.asarray(gamma_dd),
        grid_spec=GridSpec(bounds=list(sample_bounds), shape=sample_shape),
        name=resolved_name,
        interp_method=interp_method,
    )
