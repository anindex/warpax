"""Cactus / Einstein Toolkit HDF5 slice reader.

Reads an ET-compatible HDF5 file containing ADM quantities at a specified
iteration/timelevel group, assembles the 4-metric, and returns an
:class:`InterpolatedADMMetric` for the standard warpax pipeline.

Schema (matches our hand-synth generator at
``tests/fixtures/cactus/generate_minkowski_slice.py``):

.. code-block:: text

    /ITERATION={i}/TIMELEVEL={t}/
        alp -> (nz, ny, nx) float64 lapse
        betax -> (nz, ny, nx) float64 shift (lower index)
        betay -> (nz, ny, nx) float64
        betaz -> (nz, ny, nx) float64
        gxx -> (nz, ny, nx) float64 spatial metric (symmetric)
        gxy -> (nz, ny, nx) float64
        gxz -> (nz, ny, nx) float64
        gyy -> (nz, ny, nx) float64
        gyz -> (nz, ny, nx) float64
        gzz -> (nz, ny, nx) float64

    Attributes on the TIMELEVEL group:
        time -> float coordinate time
        x0, y0, z0 -> float lower-bound of grid on each axis
        dx, dy, dz -> float grid spacing on each axis

Orientation convention (pin): arrays are C-order with shape
``(nz, ny, nx)``; this loader transposes to ``(nx, ny, nz)`` for the
warpax ``(t, x, y, z)`` ordering. Nt=1 (single timelevel).

Scope: single-iteration, single-timelevel ONLY. Multi-iteration +
component groups (AMR) deferred to / .
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from warpax.geometry import GridSpec

from ._interpolated import InterpolatedADMMetric

__all__ = ["load_cactus_slice"]


def _read_slice_arrays(
    f, iteration: int, timelevel: int
) -> dict[str, np.ndarray]:
    """Read the ET-canonical keys from the specified group."""
    group_path = f"ITERATION={iteration}/TIMELEVEL={timelevel}"
    if group_path not in f:
        raise ValueError(
            f"HDF5 group '{group_path}' not found. Available top-level groups: "
            f"{list(f.keys())}."
        )
    grp = f[group_path]
    keys = ("alp", "betax", "betay", "betaz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz")
    missing = [k for k in keys if k not in grp]
    if missing:
        raise ValueError(
            f"HDF5 group '{group_path}' missing ET-canonical datasets: {missing}"
        )
    data = {k: np.asarray(grp[k][()], dtype=np.float64) for k in keys}

    attrs = {}
    for attr_key in ("time", "x0", "y0", "z0", "dx", "dy", "dz"):
        if attr_key in grp.attrs:
            attrs[attr_key] = float(grp.attrs[attr_key])
    return {"arrays": data, "attrs": attrs}


def _assemble_adm_from_zyx(
    data: dict[str, np.ndarray], attrs: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[float, float], ...], tuple[int, int, int, int]]:
    """Transpose ET ``(nz, ny, nx)`` layout to warpax ``(nx, ny, nz)`` + wrap Nt=1."""
    # Each ET array is (nz, ny, nx); transpose to (nx, ny, nz).
    perm = (2, 1, 0)
    alp_zyx = data["alp"]
    nz, ny, nx = alp_zyx.shape
    # Single timelevel wrapped into the warpax (Nt, Nx, Ny, Nz) shape.
    alpha = alp_zyx.transpose(perm)[None, :, :, :]  # (1, nx, ny, nz)

    # Shift: stored lower-index beta_i; warpax stores upper-index beta^i.
    # For Minkowski or any case with gamma_{ij} = delta_{ij}, lower == upper.
    # For general cases we invert the spatial metric pointwise.
    beta_lower = np.stack(
        [
            data["betax"].transpose(perm),
            data["betay"].transpose(perm),
            data["betaz"].transpose(perm),
        ],
        axis=-1,
    )  # (nx, ny, nz, 3)

    # Assemble gamma_{ij}, symmetric.
    def _g(key: str) -> np.ndarray:
        return data[key].transpose(perm)

    gxx = _g("gxx")
    gxy = _g("gxy")
    gxz = _g("gxz")
    gyy = _g("gyy")
    gyz = _g("gyz")
    gzz = _g("gzz")
    gamma = np.stack(
        [
            np.stack([gxx, gxy, gxz], axis=-1),
            np.stack([gxy, gyy, gyz], axis=-1),
            np.stack([gxz, gyz, gzz], axis=-1),
        ],
        axis=-2,
    )  # (nx, ny, nz, 3, 3)

    # Convert lower-index beta_i -> upper-index beta^i via gamma^{ij}.
    gamma_inv = np.linalg.inv(gamma)
    beta_upper_3d = np.einsum("...ij,...j->...i", gamma_inv, beta_lower)

    # Wrap to (Nt=1, Nx, Ny, Nz, ...).
    beta = beta_upper_3d[None, ...]
    gamma4d = gamma[None, ...]

    # Bounds: use x0..z0 + dx..dz attrs if present, else default to (-1, 1).
    x0 = attrs.get("x0", -1.0)
    y0 = attrs.get("y0", -1.0)
    z0 = attrs.get("z0", -1.0)
    dx = attrs.get("dx", 2.0 / max(nx - 1, 1))
    dy = attrs.get("dy", 2.0 / max(ny - 1, 1))
    dz = attrs.get("dz", 2.0 / max(nz - 1, 1))
    t_val = attrs.get("time", 0.0)

    # Single-timelevel: bound t with a non-degenerate [t, t+1] interval so
    # the interpolator's (high - low) division stays finite. Nt=1 means
    # map_coordinates index=0 always selects the single slice.
    bounds = (
        (t_val, t_val + 1.0),
        (x0, x0 + dx * (nx - 1)),
        (y0, y0 + dy * (ny - 1)),
        (z0, z0 + dz * (nz - 1)),
    )
    shape = (1, nx, ny, nz)
    return alpha, beta, gamma4d, bounds, shape


def load_cactus_slice(
    path: str | Path,
    iteration: int = 0,
    timelevel: int = 0,
    interp_method: str = "cubic",
) -> InterpolatedADMMetric:
    """Load a Cactus / ET HDF5 slice into an :class:`InterpolatedADMMetric`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ET-compatible HDF5 file.
    iteration : int, default 0
        Iteration index (matches the ``ITERATION={i}`` group prefix).
    timelevel : int, default 0
        Timelevel index (matches the ``TIMELEVEL={t}`` group prefix).
    interp_method : {"linear", "cubic"}, default "cubic"
        Interpolation scheme passed to :class:`InterpolatedADMMetric`.

    Returns
    -------
    InterpolatedADMMetric
        ADM 3+1 metric backed by the loaded slice
        (``Nt=1`` single-timelevel wrapper).

    Raises
    ------
    ImportError
        If ``h5py`` is not installed
        (``pip install 'warpax[interop]'``).
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the HDF5 file does not carry the expected ET-canonical
        iteration/timelevel group or dataset keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cactus HDF5 not found: {path}")
    try:
        import h5py
    except ImportError as err:
        raise ImportError(
            "Loading a Cactus/ET HDF5 slice requires h5py. "
            "Install via `pip install 'warpax[interop]'`."
        ) from err

    with h5py.File(path, "r") as f:
        payload = _read_slice_arrays(f, iteration, timelevel)

    alpha_np, beta_np, gamma_np, bounds, shape = _assemble_adm_from_zyx(
        payload["arrays"], payload["attrs"]
    )

    return InterpolatedADMMetric(
        alpha_grid=jnp.asarray(alpha_np),
        beta_grid=jnp.asarray(beta_np),
        gamma_grid=jnp.asarray(gamma_np),
        grid_spec=GridSpec(bounds=list(bounds), shape=shape),
        name=f"cactus_{path.stem}_it{iteration}_tl{timelevel}",
        interp_method=interp_method,
    )
