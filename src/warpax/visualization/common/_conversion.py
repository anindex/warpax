"""JAX-to-NumPy conversion: freeze curvature and EC results into FrameData.

Two public functions:
- ``freeze_curvature``: GridCurvatureResult -> FrameData (curvature invariants)
- ``freeze_ec``: ECGridResult -> FrameData (EC margins + classification)

All conversions use ``np.asarray()`` to force device-to-host transfer and
produce genuine ``np.ndarray`` objects.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._frame_data import FrameData

if TYPE_CHECKING:
    from warpax.energy_conditions.types import ECGridResult
    from warpax.geometry.grid import GridCurvatureResult
    from warpax.geometry.types import GridSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _symmetric_clim(arr: np.ndarray) -> tuple[float, float]:
    """Symmetric color limits centered at 0 for diverging colormaps.

    Returns ``(-max_abs, max_abs)`` where ``max_abs = max(|min|, |max|)``.
    Falls back to ``(-1.0, 1.0)`` if data is near-zero.
    """
    max_abs = max(abs(float(np.nanmin(arr))), abs(float(np.nanmax(arr))))
    if max_abs < 1e-15:
        max_abs = 1.0
    return (-max_abs, max_abs)


def _magnitude_clim(arr: np.ndarray) -> tuple[float, float]:
    """Color limits for magnitude (sequential) colormaps."""
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1.0
    return (vmin, vmax)


def _extract_coordinates(grid_spec: "GridSpec") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract spatial coordinates from GridSpec as NumPy arrays."""
    X, Y, Z = grid_spec.meshgrid  # each (Nx, Ny, Nz), JAX arrays
    return np.asarray(X), np.asarray(Y), np.asarray(Z)


# ---------------------------------------------------------------------------
# Curvature freeze
# ---------------------------------------------------------------------------

# Colormaps: diverging for signed quantities, sequential for magnitudes
_CURVATURE_COLORMAPS: dict[str, str] = {
    "ricci_scalar": "RdBu_r",
    "kretschner": "inferno",
    "ricci_squared": "inferno",
    "weyl_squared": "inferno",
    "energy_density": "RdBu_r",
}

# Fields that get symmetric (diverging) color limits
_DIVERGING_CURVATURE_FIELDS = {"ricci_scalar", "energy_density"}


def freeze_curvature(
    grid_result: "GridCurvatureResult",
    grid_spec: "GridSpec",
    *,
    metric_name: str,
    v_s: float,
    t: float = 0.0,
) -> FrameData:
    """Freeze a GridCurvatureResult into a FrameData snapshot.

    Converts all JAX arrays to NumPy and computes rendering hints.

    Parameters
    ----------
    grid_result : GridCurvatureResult
        Output from ``evaluate_curvature_grid``.
    grid_spec : GridSpec
        Grid specification used for the computation.
    metric_name : str
        Warp metric identifier (e.g. ``"alcubierre"``).
    v_s : float
        Warp bubble velocity parameter.
    t : float, optional
        Time coordinate of the evaluation (default 0.0).

    Returns
    -------
    FrameData
        Frozen snapshot with curvature invariants as scalar fields.
    """
    x, y, z = _extract_coordinates(grid_spec)

    # Convert curvature scalar fields to NumPy
    scalar_fields: dict[str, np.ndarray] = {
        "ricci_scalar": np.asarray(grid_result.ricci_scalar),
        "kretschner": np.asarray(grid_result.kretschner),
        "ricci_squared": np.asarray(grid_result.ricci_squared),
        "weyl_squared": np.asarray(grid_result.weyl_squared),
        "energy_density": np.asarray(grid_result.stress_energy[..., 0, 0]),
    }

    # Compute rendering hints
    colormaps = dict(_CURVATURE_COLORMAPS)
    clim: dict[str, tuple[float, float]] = {}
    for name, arr in scalar_fields.items():
        if name in _DIVERGING_CURVATURE_FIELDS:
            clim[name] = _symmetric_clim(arr)
        else:
            clim[name] = _magnitude_clim(arr)

    return FrameData(
        x=x,
        y=y,
        z=z,
        scalar_fields=scalar_fields,
        metric_name=metric_name,
        v_s=v_s,
        grid_shape=grid_spec.shape,
        t=t,
        colormaps=colormaps,
        clim=clim,
    )


# ---------------------------------------------------------------------------
# EC freeze
# ---------------------------------------------------------------------------

_EC_COLORMAPS: dict[str, str] = {
    "nec_margin": "RdBu_r",
    "wec_margin": "RdBu_r",
    "sec_margin": "RdBu_r",
    "dec_margin": "RdBu_r",
    "he_type": "tab10",
    "rho": "RdBu_r",
}


def freeze_ec(
    ec_result: "ECGridResult",
    grid_spec: "GridSpec",
    *,
    metric_name: str,
    v_s: float,
    t: float = 0.0,
    curvature_result: "GridCurvatureResult | None" = None,
) -> FrameData:
    """Freeze an ECGridResult into a FrameData snapshot.

    Converts EC margins, classification, and optionally curvature invariants
    to NumPy arrays with rendering hints.

    Parameters
    ----------
    ec_result : ECGridResult
        Output from ``verify_grid``.
    grid_spec : GridSpec
        Grid specification used for the computation.
    metric_name : str
        Warp metric identifier.
    v_s : float
        Warp bubble velocity parameter.
    t : float, optional
        Time coordinate (default 0.0).
    curvature_result : GridCurvatureResult, optional
        If provided, curvature invariants are also included in the snapshot.

    Returns
    -------
    FrameData
        Frozen snapshot with EC margins and classification as scalar fields.
    """
    x, y, z = _extract_coordinates(grid_spec)

    # Core EC scalar fields
    scalar_fields: dict[str, np.ndarray] = {
        "nec_margin": np.asarray(ec_result.nec_margins),
        "wec_margin": np.asarray(ec_result.wec_margins),
        "sec_margin": np.asarray(ec_result.sec_margins),
        "dec_margin": np.asarray(ec_result.dec_margins),
        "he_type": np.asarray(ec_result.he_types),
        "rho": np.asarray(ec_result.rho),
    }

    # Optionally include curvature invariants
    if curvature_result is not None:
        scalar_fields["ricci_scalar"] = np.asarray(curvature_result.ricci_scalar)
        scalar_fields["kretschner"] = np.asarray(curvature_result.kretschner)
        scalar_fields["ricci_squared"] = np.asarray(curvature_result.ricci_squared)
        scalar_fields["weyl_squared"] = np.asarray(curvature_result.weyl_squared)
        scalar_fields["energy_density"] = np.asarray(
            curvature_result.stress_energy[..., 0, 0]
        )

    # Build colormaps dict
    colormaps: dict[str, str] = {}
    for name in scalar_fields:
        if name in _EC_COLORMAPS:
            colormaps[name] = _EC_COLORMAPS[name]
        elif name in _CURVATURE_COLORMAPS:
            colormaps[name] = _CURVATURE_COLORMAPS[name]
        else:
            colormaps[name] = "viridis"

    # Build clim dict
    _diverging_ec = {"nec_margin", "wec_margin", "sec_margin", "dec_margin", "rho"}
    clim: dict[str, tuple[float, float]] = {}
    for name, arr in scalar_fields.items():
        if name == "he_type":
            clim[name] = (1.0, 4.0)
        elif name in _diverging_ec or name in _DIVERGING_CURVATURE_FIELDS:
            clim[name] = _symmetric_clim(arr)
        else:
            clim[name] = _magnitude_clim(arr)

    return FrameData(
        x=x,
        y=y,
        z=z,
        scalar_fields=scalar_fields,
        metric_name=metric_name,
        v_s=v_s,
        grid_shape=grid_spec.shape,
        t=t,
        colormaps=colormaps,
        clim=clim,
    )
