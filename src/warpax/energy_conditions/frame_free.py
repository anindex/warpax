"""Frame-independent, all-velocity energy-condition certification.

The Hawking-Ellis classification of the mixed stress-energy tensor ``T^a_b``
and, for Type-I matter, the eigenvalue inequalities on ``(rho, p_i)`` constitute
an *observer-independent* energy-condition test (Hawking & Ellis 1973;
Santiago, Schuster & Visser 2021; Martin-Moruno & Visser 2017). Neither
construction references the Eulerian normal ``n^a`` or a timelike tetrad, so --
unlike the rapidity-capped optimizer in :mod:`.optimization`, whose tetrad uses
``alpha = 1/sqrt(-g^{00})`` -- this certification is well-defined at ALL warp
velocities, including v_s >= 1 where ``g_00`` changes sign and the
coordinate-stationary (Eulerian) congruence ceases to exist.

For Type-I points the returned NEC/WEC/SEC margins are exact and cap-free; the
DEC margin is the eigenvalue bound ``rho - |p_i|``, necessary and sufficient at
Type I (see :func:`.eigenvalue_checks.check_dec_typeI_eigenvalue_bound`). For non-Type-I
points (II/III/IV) no invariant rest frame exists, so the eigenvalue margins are
NaN by construction; callers should report these as "intrinsically
observer-dependent" rather than silently substituting a frame-dependent
optimizer value.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .classification import classify_with_solver
from .eigenvalue_checks import check_all
from .types import FrameFreeGridResult
from .verifier import _classify_grid_batch


def certify_point_frame_free(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    g_inv: Float[Array, "4 4"] | None = None,
    *,
    solver: str = "auto",
    tol: float = 1e-10,
) -> dict:
    """Frame-independent EC certification at a single spacetime point.

    Parameters
    ----------
    T_ab : Float[Array, "4 4"]
        Covariant stress-energy tensor ``T_{ab}``.
    g_ab : Float[Array, "4 4"]
        Covariant metric ``g_{ab}``.
    g_inv : Float[Array, "4 4"] or None
        Inverse metric ``g^{ab}``; computed from ``g_ab`` when ``None``.
    solver : {"auto", "standard", "generalized"}
        Eigenvalue backend (see :func:`.classification.classify_with_solver`).
    tol : float
        Classification tolerance.

    Returns
    -------
    dict
        ``he_type`` (1-4), ``rho``, ``pressures`` (NaN if non-Type-I),
        ``nec``/``wec``/``sec``/``dec`` margins (NaN if non-Type-I),
        ``eigenvalues``, ``eigenvalues_imag``, ``is_vacuum``.
    """
    if g_inv is None:
        g_inv = jnp.linalg.inv(g_ab)
    T_mixed = jnp.einsum("ac,cb->ab", g_inv, T_ab)
    cls = classify_with_solver(T_mixed, g_ab, T_ab, solver=solver, tol=tol)
    nec, wec, sec, dec = check_all(cls.rho, cls.pressures)
    return {
        "he_type": cls.he_type,
        "rho": cls.rho,
        "pressures": cls.pressures,
        "nec": nec,
        "wec": wec,
        "sec": sec,
        "dec": dec,
        "eigenvalues": cls.eigenvalues,
        "eigenvalues_imag": cls.eigenvalues_imag,
        "is_vacuum": cls.is_vacuum,
    }


def certify_grid_frame_free(
    T_field: Float[Array, "... 4 4"],
    g_field: Float[Array, "... 4 4"],
    g_inv_field: Float[Array, "... 4 4"] | None = None,
    *,
    solver: str = "auto",
    tol: float = 1e-10,
) -> FrameFreeGridResult:
    """Frame-independent EC certification across an evaluation grid.

    Reuses :func:`._classify_grid_batch` (standard ``jnp.linalg.eig`` with an
    automatic generalized-pencil fallback on near-degenerate points) and
    :func:`.eigenvalue_checks.check_all`. No optimizer, no Eulerian normal, no
    timelike tetrad: valid at all velocities.

    Parameters
    ----------
    T_field : Float[Array, "... 4 4"]
        Covariant stress-energy on a grid; leading dims are the grid shape.
    g_field, g_inv_field : Float[Array, "... 4 4"]
        Covariant and (optional) inverse metric on the same grid.
    solver : {"auto", "standard", "generalized"}
        Eigenvalue backend.
    tol : float
        Classification tolerance.

    Returns
    -------
    FrameFreeGridResult
    """
    grid_shape = T_field.shape[:-2]
    flat_T = jnp.reshape(T_field, (-1, 4, 4))
    flat_g = jnp.reshape(g_field, (-1, 4, 4))
    if g_inv_field is None:
        flat_ginv = jax.vmap(jnp.linalg.inv)(flat_g)
    else:
        flat_ginv = jnp.reshape(g_inv_field, (-1, 4, 4))
    flat_Tmixed = jnp.einsum("nac,ncb->nab", flat_ginv, flat_T)

    cls = _classify_grid_batch(flat_Tmixed, flat_g, flat_T, solver=solver)
    nec, wec, sec, dec = jax.vmap(check_all)(cls.rho, cls.pressures)

    he = np.asarray(cls.he_type)
    n_vacuum = int(np.sum(np.asarray(cls.is_vacuum) > 0.5))
    # nanmax: a NaN-sanitized eigenvalue must not poison the grid-wide
    # imaginary-part diagnostic (it is a summary, not a certified margin).
    max_imag = float(np.nanmax(np.abs(np.asarray(cls.eigenvalues_imag))))

    def _rs(x, trailing=()):  # reshape flat -> grid
        return jnp.reshape(x, (*grid_shape, *trailing))

    return FrameFreeGridResult(
        he_types=_rs(cls.he_type),
        eigenvalues=_rs(cls.eigenvalues, (4,)),
        eigenvalues_imag=_rs(cls.eigenvalues_imag, (4,)),
        rho=_rs(cls.rho),
        pressures=_rs(cls.pressures, (3,)),
        nec_margins=_rs(nec),
        wec_margins=_rs(wec),
        sec_margins=_rs(sec),
        dec_margins=_rs(dec),
        is_vacuum=_rs(cls.is_vacuum),
        n_type_i=int(np.sum(he == 1.0)),
        n_type_ii=int(np.sum(he == 2.0)),
        n_type_iii=int(np.sum(he == 3.0)),
        n_type_iv=int(np.sum(he == 4.0)),
        n_vacuum=n_vacuum,
        n_total=int(he.size),
        max_imag_eigenvalue=max_imag,
    )


def type_fractions(
    result: FrameFreeGridResult,
    mask: Float[Array, "..."] | None = None,
    volume_weights: Float[Array, "..."] | None = None,
) -> dict[str, float]:
    """Volume-weighted Hawking-Ellis type fractions (optionally wall-restricted).

    Parameters
    ----------
    result : FrameFreeGridResult
    mask : Float[Array, "..."] or None
        Boolean/0-1 selection (e.g. the wall mask). ``None`` selects all points.
    volume_weights : Float[Array, "..."] or None
        Proper-volume weights (e.g. ``GridSpec.volume_weights_array``) so that
        clustered grids are not biased toward the densely-sampled wall. ``None``
        gives uniform (point-count) weighting.

    Returns
    -------
    dict
        ``frac_type_i/ii/iii/iv`` and ``n_selected``.
    """
    he = np.asarray(result.he_types).ravel()
    sel = (
        np.ones_like(he, dtype=float)
        if mask is None
        else np.asarray(mask).ravel().astype(float)
    )
    w = (
        sel
        if volume_weights is None
        else sel * np.asarray(volume_weights).ravel()
    )
    wt = float(np.sum(w))
    if wt <= 0.0:
        return {f"frac_type_{k}": 0.0 for k in ("i", "ii", "iii", "iv")} | {
            "n_selected": 0
        }
    out = {
        f"frac_type_{k}": float(np.sum(w * (he == t)) / wt)
        for k, t in (("i", 1.0), ("ii", 2.0), ("iii", 3.0), ("iv", 4.0))
    }
    out["n_selected"] = int(np.sum(sel > 0.5))
    return out


def typeI_min_margins(
    result: FrameFreeGridResult,
    mask: Float[Array, "..."] | None = None,
) -> dict[str, float]:
    """Minimum invariant eigenvalue margins over Type-I points (optionally masked).

    These are the cap-free, frame-independent "peak deficit" severities: the most
    negative value of each eigenvalue inequality slack across Type-I points.
    Returns NaN for a condition when no Type-I points are selected.
    """
    he = np.asarray(result.he_types).ravel()
    sel = (
        np.ones_like(he, dtype=bool)
        if mask is None
        else np.asarray(mask).ravel().astype(bool)
    )
    typeI = sel & (he == 1.0)
    out: dict[str, float] = {}
    for key, field in (
        ("nec", result.nec_margins),
        ("wec", result.wec_margins),
        ("sec", result.sec_margins),
        ("dec", result.dec_margins),
    ):
        vals = np.asarray(field).ravel()[typeI]
        vals = vals[np.isfinite(vals)]
        out[f"{key}_min"] = float(np.min(vals)) if vals.size else float("nan")
    out["n_type_i_selected"] = int(np.sum(typeI))
    return out
