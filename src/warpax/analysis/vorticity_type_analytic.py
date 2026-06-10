r"""Analytic mechanism: shift vorticity sources the Hawking-Ellis Type-IV pair.

Rodal (arXiv:2512.18008) showed *empirically* that an irrotational warp shift
yields a globally Type-I stress-energy and that adding vorticity destroys it.
Santiago-Schuster-Visser supply the irrotational-implies-Type-I lemma for the
unit-lapse, flat-slice (``alpha = 1``, ``gamma_ij = delta_ij``) drive family.
This module establishes the *mechanism* of the converse: the imaginary part
``f`` of the Type-IV eigenvalue pair ``{-rho +/- i f, p_1, p_2}`` (Martin-Moruno
& Visser, PRD 103 124003) is, at leading order in the wall gradient, *linear in
the shift vorticity*,

.. math::

    f \;=\; \kappa \, \omega \,, \qquad \omega = \sqrt{\omega^2}\,,

with a wall-geometry coefficient ``kappa``. A controlled pure-rotation shift
(zero expansion, zero shear) makes this exact: the symmetric part of the shift
gradient sources the (real) Type-I spectrum, while the antisymmetric part
(vorticity) sources the momentum-density asymmetry of ``T^a_b`` that admits no
rest frame -- i.e. the imaginary eigenvalue pair. ``f -> 0`` as ``omega -> 0``,
recovering Type I.

This is a sufficient-direction / controlled-limit mechanism, not a full converse:
for general (non-flat-slice) shells the link remains numerical. The numeric
validation against Natario/Alcubierre/VdB (vorticity, Type IV) versus Rodal
(irrotational, Type I) is in ``scripts/derive_vorticity_type.py``. The
cross-metric excess of the measured ``f`` over ``kappa * omega`` -- largest for
the high-shear, zero-expansion Natario wall -- quantifies the wall-geometry
dependence of ``kappa`` and points to a subleading shear coupling.
"""
from __future__ import annotations

import numpy as np

# Below this |Im lambda| (relative to the eigenvalue scale) the spectrum is
# treated as real, matching the classifier's degeneracy tolerance.
_IMAG_FLOOR = 1e-10


def imaginary_part_estimate(omega: float, kappa: float) -> float:
    r"""Leading-order Type-IV imaginary eigenvalue part ``f = kappa * omega``.

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
    r"""Vorticity above which the wall is Type IV: ``omega* = imag_floor / kappa``.

    Below ``omega*`` the imaginary part is within the classifier's real-spectrum
    tolerance and the point reads as Type I; above it, the complex pair appears.
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
    ss_tot = float(np.sum(imag ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"kappa": kappa, "r_squared": r2, "n": int(omega.size)}
