r"""Independent all-observer verification of warp-drive positive-energy claims (Contribution 2).

The warp-drive literature increasingly reports "positive-energy" or
"global Hawking-Ellis Type I" constructions (Lentz 2021; Bobrick-Martire 2021;
Fell-Heisenberg 2021; Rodal arXiv:2512.18008, 2025). These claims are typically
established in the EULERIAN (ADM normal) frame and at a single velocity. Santiago,
Schuster & Visser (PRD 105, 064038) showed this is insufficient: the weak/null
energy conditions require ALL timelike/null observers, not the comoving Eulerian
one. This module operationalises that principle. From the eigenstructure of
``T^a_b`` alone (frame-independent; see :mod:`..energy_conditions.frame_free`) it
reports:

  - the *single-frame miss*: the fraction of all-observer EC violations that the
    Eulerian frame fails to see (Eulerian margin >= 0 yet the spacetime violates
    the condition for some observer / is Type-IV with no rest frame);
  - the integrated exotic-matter content E_- (the community figure of merit, cf.
    Pfenning-Ford), reported both invariantly (Type-I proper energy density) and
    in the Eulerian frame for comparison;
  - peak proper-energy-deficit reduction factors between metrics, to independently
    check published claims (e.g. Rodal's ~38x vs Alcubierre).

Tone of the verification (by design): a single-frame, single-velocity claim may be
correct *as stated*; the all-observer reality across velocities can still differ.
The miss fraction is a property of the single-frame DIAGNOSTIC, not a refutation
of any author's algebra.

The Eulerian comparison uses :func:`..energy_conditions.compute_eulerian_ec`,
whose ADM normal is only timelike for ``v_s < 1``; the verification is therefore run at
subluminal velocities (the regime in which positive-energy claims are stated).
The invariant quantities themselves remain valid at all velocities (Contribution 1).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..energy_conditions.frame_free import certify_grid_frame_free
from ..energy_conditions.verifier import _eulerian_ec_point

_CONDS = ("nec", "wec", "sec", "dec")


def _flat(x, trailing):
    return jnp.reshape(x, (-1, *trailing)) if trailing else jnp.reshape(x, (-1,))


def single_frame_miss(
    T_field: Float[Array, "... 4 4"],
    g_field: Float[Array, "... 4 4"],
    g_inv_field: Float[Array, "... 4 4"],
    *,
    mask: Float[Array, "..."] | None = None,
    volume_weights: Float[Array, "..."] | None = None,
    atol: float = 1e-10,
) -> dict:
    r"""Fraction of all-observer EC violations missed by the Eulerian frame.

    A point counts as all-observer violating a condition if either
    (i) it is Type I with the invariant eigenvalue margin ``< -atol``, or
    (ii) it is Type IV: a complex-eigenvalue stress-energy has no rest frame and
    violates the NEC -- hence also WEC/SEC/DEC -- for some observer (Hawking-Ellis;
    Martin-Moruno-Visser 2017). It is "missed" when the Eulerian margin is
    ``>= 0`` (the single frame reports no problem). Type II/III points (negligible
    in these metrics) are excluded from the count, conservatively.

    Returns per-condition ``miss_rate`` (volume-weighted), ``n_violated``,
    ``n_missed``, plus the Type-IV volume fraction within the selection.
    """
    ff = certify_grid_frame_free(T_field, g_field, g_inv_field)
    flat_T = _flat(T_field, (4, 4))
    flat_g = _flat(g_field, (4, 4))
    flat_gi = _flat(g_inv_field, (4, 4))
    eul = jax.vmap(_eulerian_ec_point)(flat_T, flat_g, flat_gi)

    he = np.asarray(ff.he_types).ravel()
    is_typeI = he == 1.0
    is_typeIV = he == 4.0
    vac = np.asarray(ff.is_vacuum).ravel() > 0.5

    n = he.size
    sel = np.ones(n, bool) if mask is None else np.asarray(mask).ravel().astype(bool)
    w = (np.ones(n) if volume_weights is None
         else np.asarray(volume_weights).ravel())
    w = w * sel

    inv = {
        "nec": np.asarray(ff.nec_margins).ravel(),
        "wec": np.asarray(ff.wec_margins).ravel(),
        "sec": np.asarray(ff.sec_margins).ravel(),
        "dec": np.asarray(ff.dec_margins).ravel(),
    }
    out: dict = {}
    for c in _CONDS:
        eul_m = np.asarray(eul[c]).ravel()
        typeI_viol = is_typeI & (inv[c] < -atol)
        # Type IV (excluding near-vacuum noise) violates every condition: a
        # complex-eigenvalue T^a_b has no rest frame and fails the NEC.
        typeIV_viol = is_typeIV & (~vac)
        all_obs_viol = typeI_viol | typeIV_viol
        missed = all_obs_viol & (eul_m >= 0.0)
        wv = float(np.sum(w * all_obs_viol))
        wm = float(np.sum(w * missed))
        out[c] = {
            "miss_rate": (wm / wv) if wv > 0 else None,
            "n_violated": int(np.sum(sel & all_obs_viol)),
            "n_missed": int(np.sum(sel & missed)),
        }
    wsel = float(np.sum(w))
    out["frac_type_iv"] = (
        float(np.sum(w * (he == 4.0)) / wsel) if wsel > 0 else 0.0
    )
    out["n_selected"] = int(np.sum(sel))
    return out


def integrated_exotic_content(
    T_field: Float[Array, "... 4 4"],
    g_field: Float[Array, "... 4 4"],
    g_inv_field: Float[Array, "... 4 4"],
    volume_weights: Float[Array, "..."],
    *,
    mask: Float[Array, "..."] | None = None,
) -> dict:
    r"""Integrated exotic-matter content E_- (figure of merit; cf. Pfenning-Ford).

    Reports the proper-volume integral of negative energy density:
      - invariant: over Type-I points, using the Lorentz-invariant proper energy
        density ``rho`` (the timelike eigenvalue). Non-Type-I points have no
        invariant energy density and are excluded but counted by volume.
      - Eulerian: over all points, using ``rho_Eul = T_{ab} n^a n^b`` (valid for
        v_s < 1), for comparison with single-frame claims.

    Returns ``E_minus_inv``, ``E_plus_inv``, ``E_minus_eul``, ``E_plus_eul``,
    ``balance_inv`` (E_+/|E_-|), ``typeIV_volume_frac``.
    """
    ff = certify_grid_frame_free(T_field, g_field, g_inv_field)
    flat_T = _flat(T_field, (4, 4))
    flat_g = _flat(g_field, (4, 4))
    flat_gi = _flat(g_inv_field, (4, 4))
    eul = jax.vmap(_eulerian_ec_point)(flat_T, flat_g, flat_gi)
    rho_eul = np.asarray(eul["wec"]).ravel()  # T_ab n^a n^b

    he = np.asarray(ff.he_types).ravel()
    rho_inv = np.asarray(ff.rho).ravel()  # NaN for non-Type-I
    w = np.asarray(volume_weights).ravel()
    if mask is not None:
        w = w * np.asarray(mask).ravel().astype(float)

    is_typeI = (he == 1.0) & np.isfinite(rho_inv)
    wI = w * is_typeI
    E_minus_inv = float(np.sum(wI * np.minimum(rho_inv_safe(rho_inv), 0.0)))
    E_plus_inv = float(np.sum(wI * np.maximum(rho_inv_safe(rho_inv), 0.0)))
    E_minus_eul = float(np.sum(w * np.minimum(rho_eul, 0.0)))
    E_plus_eul = float(np.sum(w * np.maximum(rho_eul, 0.0)))
    wsel = float(np.sum(w))
    return {
        "E_minus_inv": E_minus_inv,
        "E_plus_inv": E_plus_inv,
        "E_minus_eul": E_minus_eul,
        "E_plus_eul": E_plus_eul,
        "balance_inv": (E_plus_inv / abs(E_minus_inv)) if E_minus_inv != 0 else None,
        "typeIV_volume_frac": (
            float(np.sum(w * (he == 4.0)) / wsel) if wsel > 0 else 0.0
        ),
    }


def rho_inv_safe(rho_inv: np.ndarray) -> np.ndarray:
    """Replace NaN invariant energy density (non-Type-I) with 0 for integration."""
    return np.where(np.isfinite(rho_inv), rho_inv, 0.0)


def peak_proper_energy_deficit(
    T_field: Float[Array, "... 4 4"],
    g_field: Float[Array, "... 4 4"],
    g_inv_field: Float[Array, "... 4 4"],
    *,
    mask: Float[Array, "..."] | None = None,
) -> dict:
    """Peak proper-energy deficit = ``-min(rho)`` (invariant Type-I and Eulerian).

    The invariant value is over Type-I points only (the Eulerian value over all
    points). Used by :func:`reduction_factors` to check published reduction
    claims.
    """
    ff = certify_grid_frame_free(T_field, g_field, g_inv_field)
    flat_T = _flat(T_field, (4, 4))
    flat_g = _flat(g_field, (4, 4))
    flat_gi = _flat(g_inv_field, (4, 4))
    eul = jax.vmap(_eulerian_ec_point)(flat_T, flat_g, flat_gi)
    rho_eul = np.asarray(eul["wec"]).ravel()

    he = np.asarray(ff.he_types).ravel()
    rho_inv = np.asarray(ff.rho).ravel()
    sel = (np.ones(he.size, bool) if mask is None
           else np.asarray(mask).ravel().astype(bool))
    typeI = sel & (he == 1.0) & np.isfinite(rho_inv)
    inv_vals = rho_inv[typeI]
    eul_vals = rho_eul[sel]
    return {
        "peak_deficit_inv": float(-np.min(inv_vals)) if inv_vals.size else float("nan"),
        "peak_deficit_eul": float(-np.min(eul_vals)) if eul_vals.size else float("nan"),
    }


def reduction_factors(peaks: dict, baseline: str = "Alcubierre") -> dict:
    """Peak-deficit reduction factors relative to ``baseline`` (both frames).

    ``peaks`` maps metric name -> dict with ``peak_deficit_inv`` /
    ``peak_deficit_eul`` (output of :func:`peak_proper_energy_deficit`).
    Factor > 1 means a smaller deficit than the baseline.
    """
    base = peaks[baseline]
    out = {}
    for name, p in peaks.items():
        out[name] = {
            "vs_" + baseline + "_inv": (
                base["peak_deficit_inv"] / p["peak_deficit_inv"]
                if p["peak_deficit_inv"] not in (0.0,) and np.isfinite(p["peak_deficit_inv"])
                else None
            ),
            "vs_" + baseline + "_eul": (
                base["peak_deficit_eul"] / p["peak_deficit_eul"]
                if p["peak_deficit_eul"] not in (0.0,) and np.isfinite(p["peak_deficit_eul"])
                else None
            ),
        }
    return out
