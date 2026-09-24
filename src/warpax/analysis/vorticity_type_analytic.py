r"""Empirical slope diagnostics for a restricted rotational shift family.

For unit lapse and a fixed Euclidean spatial metric, Eulerian momentum
involves second spatial derivatives of the shift. Shift curl alone does
not determine Hawking--Ellis type. A localized rotation varies momentum,
shear and vorticity together; its fitted imaginary-eigenvalue slope is a
family-specific diagnostic, not a general Type-IV criterion.

The momentum-aligned 2x2 stress block has a complex pair when
``4*j**2 > (rho + S_parallel)**2``. Applying this reduction to a full tensor
requires the transverse couplings to vanish. The scripts compare the
restricted prediction with the full eigenspectrum.
"""

from __future__ import annotations

import numpy as np

# Below this |Im lambda| (relative to the eigenvalue scale) the spectrum is
# treated as real, matching the classifier's degeneracy tolerance.
_IMAG_FLOOR = 1e-10


def imaginary_part_estimate(omega: float, kappa: float) -> float:
    r"""Fitted imaginary eigenvalue estimate ``f = kappa * omega``.

    Parameters
    ----------
    omega : float
        Shift vorticity magnitude ``sqrt(omega^2)`` (``omega^2`` from
        :func:`..analysis.shift_kinematics.compute_shift_kinematics`).
    kappa : float
        Wall-geometry coefficient (fit per family via :func:`fit_kappa`).
    """
    return float(kappa) * float(abs(omega))


def excess_over_pure_rotation(
    imag_measured: float,
    omega: float,
    kappa: float,
    omega_floor: float = 1e-12,
) -> float | None:
    r"""Ratio ``Im_measured / (kappa * omega)`` at a point.

    Quantifies how far a full metric sits above the pure-rotation prediction
    (a ratio of 1 means the controlled-limit slope is exact there). Returns
    ``None`` when the shift is effectively irrotational, where the
    pure-rotation prediction is vacuous.
    """
    if omega <= omega_floor:
        return None
    return float(imag_measured) / (float(kappa) * float(omega))


def typeIV_threshold(kappa: float, imag_floor: float = _IMAG_FLOOR) -> float:
    r"""Threshold of the fitted scalar estimate: ``omega* = imag_floor / kappa``.

    This compares ``kappa*omega`` with the supplied absolute floor; it is not
    an algebraic-type test for a general stress tensor.
    """
    if kappa <= 0.0:
        return float("inf")
    return float(imag_floor) / float(kappa)


def fit_kappa(omega: np.ndarray, imag: np.ndarray) -> dict:
    r"""Fit ``f = kappa * omega`` through the origin; return ``kappa`` and ``R^2``.

    Parameters
    ----------
    omega : array
        Vorticity magnitudes ``sqrt(omega^2)``.
    imag : array
        Corresponding measured ``max|Im lambda|`` of ``T^a_b``.
    """
    omega = np.asarray(omega, dtype=float)
    imag = np.asarray(imag, dtype=float)
    denom = float(np.sum(omega * omega))
    kappa = float(np.sum(omega * imag) / denom) if denom > 0 else 0.0
    pred = kappa * omega
    ss_res = float(np.sum((imag - pred) ** 2))
    # Uncentered total sum of squares: the model is forced through the origin
    # (no intercept), so the correct baseline is 0, not the mean. Using the
    # mean-centered form here understates R^2 (and can even go negative).
    ss_tot = float(np.sum(imag**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"kappa": kappa, "r_squared": r2, "n": int(omega.size)}
