"""WarpFactory ``.mat`` reader.

Parses a WarpFactory (``github.com/NerdsWithAttitudes/WarpFactory``) MATLAB
export into an :class:`InterpolatedADMMetric`. Schema-tolerant: v7.3
HDF5-backed ``.mat`` via :mod:`mat73`; older v7 / v6 / v4 via
:func:`scipy.io.loadmat` (mitigation).

Expected schema (after ``metricGet_Alcubierre`` + ``save('...', '-v7.3')``):

- ``metric.tensor`` - 4D array shape ``(4, 4, Nt, Nx, Ny, Nz)`` with the
  full covariant 4-metric at every grid point.
- ``metric.coords`` - struct of 1D coord arrays per axis ``(t, x, y, z)``.
- ``metric.type`` - ``str`` (e.g., ``'Alcubierre'``), used for the
  :attr:`InterpolatedADMMetric.name` accessor. Optional.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from warpax.geometry import GridSpec

from ._interpolated import InterpolatedADMMetric

__all__ = ["load_warpfactory"]


def _detect_mat_version(path: Path) -> str:
    """Return ``'v7.3'`` (HDF5) or ``'v6_v7'`` (scipy-readable) based on header."""
    with open(path, "rb") as f:
        head = f.read(128)
    if head[:8] == b"\x89HDF\r\n\x1a\n":
        return "v7.3"
    if b"MATLAB 5.0 MAT-file" in head:
        return "v6_v7"
    return "unknown"


def _load_v7_3(path: Path) -> dict[str, Any]:
    """Load v7.3 HDF5-backed .mat via mat73 (primary path)."""
    try:
        import mat73
    except ImportError as err:
        raise ImportError(
            "Loading a WarpFactory v7.3 .mat export requires mat73. "
            "Install via `pip install 'warpax[interop]'`."
        ) from err
    return mat73.loadmat(str(path))


def _load_v6_v7(path: Path) -> dict[str, Any]:
    """Load v6/v7 .mat via scipy.io.loadmat (secondary path)."""
    try:
        from scipy.io import loadmat
    except ImportError as err:
        raise ImportError(
            "Loading a WarpFactory v6/v7 .mat export requires scipy. "
            "Install via `pip install 'warpax[viz]'` or add scipy manually."
        ) from err
    return loadmat(str(path), struct_as_record=False, squeeze_me=True)


def _extract_adm_grids(
    mat_data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[float, float], ...], tuple[int, ...]]:
    """Pull (alpha, beta, gamma, bounds, shape) out of a WarpFactory mat dict.

    Decomposes the stored ``g_{ab}`` tensor into ADM 3+1 quantities:

    .. math::

        g_{00} = -\\alpha^2 + \\beta_i \\beta^i, \\quad
        g_{0i} = \\beta_i, \\quad
        g_{ij} = \\gamma_{ij}.

    So ``gamma_{ij}(t, x) = g_{ij}(t, x)``, ``beta^i = gamma^{ij} beta_j``
    with ``beta_j = g_{0j}``, and
    ``alpha = sqrt(beta^i beta_i - g_{00})`` (taking the positive branch).
    """
    metric = mat_data.get("metric")
    if metric is None:
        raise ValueError(
            "WarpFactory .mat file missing top-level 'metric' key; "
            "file may be a non-WarpFactory export."
        )

    # metric dict vs scipy mat_struct: accessor varies
    def _get(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    tensor = _get(metric, "tensor")
    coords = _get(metric, "coords")
    if tensor is None or coords is None:
        raise ValueError(
            "WarpFactory .mat file missing 'metric.tensor' or 'metric.coords'; "
            "file does not match WarpFactory v7.3 schema."
        )

    g_full = np.asarray(tensor, dtype=np.float64)  # (4, 4, Nt, Nx, Ny, Nz)
    if g_full.ndim != 6 or g_full.shape[:2] != (4, 4):
        raise ValueError(
            f"WarpFactory 'metric.tensor' must have shape (4, 4, Nt, Nx, Ny, Nz); "
            f"got {g_full.shape}."
        )

    # ADM decomposition from covariant g_{ab}.
    # Shift: beta_i = g_{0i}. beta^i = gamma^{ij} beta_j. alpha^2 = beta^i beta_i - g_{00}.
    gamma_dd = np.moveaxis(g_full[1:, 1:], (0, 1), (-2, -1))  # (Nt, Nx, Ny, Nz, 3, 3)
    beta_lower = np.moveaxis(g_full[0, 1:], 0, -1)  # (Nt, Nx, Ny, Nz, 3)
    g_00 = g_full[0, 0]  # (Nt, Nx, Ny, Nz)

    # Invert gamma pointwise for beta^i computation.
    gamma_uu = np.linalg.inv(gamma_dd)
    beta_upper = np.einsum("...ij,...j->...i", gamma_uu, beta_lower)
    beta_sq = np.einsum("...i,...i->...", beta_lower, beta_upper)
    alpha_sq = beta_sq - g_00
    alpha = np.sqrt(np.clip(alpha_sq, a_min=0.0, a_max=None))

    # Coordinate arrays.
    t = np.asarray(_get(coords, "t"), dtype=np.float64).ravel()
    x = np.asarray(_get(coords, "x"), dtype=np.float64).ravel()
    y = np.asarray(_get(coords, "y"), dtype=np.float64).ravel()
    z = np.asarray(_get(coords, "z"), dtype=np.float64).ravel()
    bounds = (
        (float(t.min()), float(t.max())),
        (float(x.min()), float(x.max())),
        (float(y.min()), float(y.max())),
        (float(z.min()), float(z.max())),
    )
    shape = (t.size, x.size, y.size, z.size)

    # Final shape assertion.
    if alpha.shape != shape:
        raise ValueError(
            f"Extracted alpha shape {alpha.shape} does not match implied "
            f"4D grid shape {shape}."
        )

    return alpha, beta_upper, gamma_dd, bounds, shape


def _derive_name(mat_data: dict[str, Any], path: Path) -> str:
    """Best-effort human-readable name from metric.type or the filename."""
    metric = mat_data.get("metric")
    if metric is not None:
        mtype = (
            metric.get("type")
            if isinstance(metric, dict)
            else getattr(metric, "type", None)
        )
        if isinstance(mtype, (bytes, str)):
            name = mtype.decode() if isinstance(mtype, bytes) else mtype
            return f"warpfactory_{name.lower()}"
    return f"warpfactory_{path.stem}"


def load_warpfactory(
    path: str | Path, interp_method: str = "cubic"
) -> InterpolatedADMMetric:
    """Load a WarpFactory ``.mat`` export into an :class:`InterpolatedADMMetric`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.mat`` file (any MATLAB version). Schema-version is
        autodetected from the file header.
    interp_method : {"linear", "cubic"}, default "cubic"
        Interpolation scheme passed through to
        :class:`InterpolatedADMMetric`.

    Returns
    -------
    InterpolatedADMMetric
        ADM 3+1 metric with 4D grids for ``alpha``, ``beta^i``, and
        ``gamma_{ij}``, plus a 4D :class:`GridSpec` (``(Nt, Nx, Ny, Nz)``).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file does not match the expected WarpFactory v7.3 schema
        (missing ``metric.tensor`` / ``metric.coords``, wrong tensor
        shape, etc.).
    ImportError
        If the required backend (``mat73`` for v7.3, ``scipy`` for v6/v7)
        is not installed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"WarpFactory .mat not found: {path}")

    version = _detect_mat_version(path)
    if version == "v7.3":
        mat_data = _load_v7_3(path)
    elif version == "v6_v7":
        mat_data = _load_v6_v7(path)
    else:
        # Last-resort: try scipy first, then mat73.
        try:
            mat_data = _load_v6_v7(path)
        except Exception:
            mat_data = _load_v7_3(path)

    alpha_np, beta_np, gamma_np, bounds, shape = _extract_adm_grids(mat_data)
    name = _derive_name(mat_data, path)

    return InterpolatedADMMetric(
        alpha_grid=jnp.asarray(alpha_np),
        beta_grid=jnp.asarray(beta_np),
        gamma_grid=jnp.asarray(gamma_np),
        grid_spec=GridSpec(bounds=list(bounds), shape=shape),
        name=name,
        interp_method=interp_method,
    )
