"""Energy-condition margins without an observer rapidity cap.

Well-conditioned Type-I points use the necessary and sufficient eigenvalue
inequalities on ``(rho, p_i)``. Other points, including Type-I points with an
ill-conditioned eigenbasis, use the numerical linear matrix inequalities in
:mod:`.slemma`. Both reductions are exact in mathematics; computed margins are
subject to floating-point error and the stated tolerances.

The LMI uses an Eulerian orthonormal tetrad. It requires a Lorentzian metric
with spacelike coordinate slices, not a timelike coordinate-time direction.
The slice normal remains timelike at any warp speed under these hypotheses.

Type-I eigenvalue slacks and Eulerian LMI margins have different normalizations.
Their signs test the same conditions, but their magnitudes cannot be compared
as a single observer-independent severity. The momentum-direction contraction
is a separate, sufficient NEC violation diagnostic when momentum is nonzero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .classification import classify_with_solver
from .eigenvalue_checks import check_all
from .slemma import certify_point as certify_point_lmi
from .slemma import noise_floor
from .types import FrameFreeGridResult
from .verifier import _classify_grid_batch


def eulerian_null_witness(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    g_inv: Float[Array, "4 4"],
) -> Float[Array, ""]:
    """Evaluate the NEC contraction along the Eulerian momentum direction.

    For nonzero momentum ``j``, the two null vectors ``k = n +/- jhat`` have
    ``-g(k, n) = 1``. The smaller contraction is

        witness = rho + S(jhat, jhat) - 2 |j|.

    A negative value witnesses NEC failure, subject to numerical error. A
    nonnegative value does not establish NEC, at any Hawking-Ellis type.
    The discriminant ``(rho + S_par)^2 - 4 |j|^2 < 0`` is sufficient for a
    negative witness; it is not a general characterization of Type III or IV.
    Changing the Eulerian normal can change the direction being tested.

    At exactly zero momentum the implementation returns ``rho`` because
    ``jhat`` is zero. That fallback is not a null contraction; use
    :func:`.slemma.null_deficit` for the full normalized null minimum.
    """
    rho, S_par, jmag = eulerian_momentum_frame(T_ab, g_ab, g_inv)
    return rho + S_par - 2.0 * jmag


def eulerian_momentum_frame(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    g_inv: Float[Array, "4 4"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Return ``(rho_n, S_par, |j|)`` in the Eulerian frame.

    For nonzero momentum, ``S_par = S(jhat, jhat)``. The momentum-plane
    discriminant is ``(rho_n + S_par)^2 - 4 |j|^2``. At zero momentum the
    implementation sets ``jhat`` and ``S_par`` to zero. The metric must be
    Lorentzian with spacelike coordinate slices.
    """
    # Eulerian normal n^a: n_a = (-1, 0, 0, 0) up to lapse; n^a = g^{ab} n_b,
    # renormalized to n.n = -1 so the construction is lapse-agnostic.
    n_low = jnp.array([-1.0, 0.0, 0.0, 0.0])
    n_up = g_inv @ n_low
    n_up = n_up / jnp.sqrt(jnp.abs(n_low @ n_up))
    n_low2 = g_ab @ n_up
    proj = jnp.eye(4) + jnp.outer(n_up, n_low2)  # h^a_b = delta + n^a n_b
    T_mixed = g_inv @ T_ab
    rho = n_up @ (T_ab @ n_up)
    j_up = -(proj @ (T_mixed @ n_up))  # spatial momentum density j^a
    j2 = j_up @ (g_ab @ j_up)
    jmag = jnp.sqrt(jnp.clip(j2, min=0.0))
    # Normalize by the largest component first to preserve small nonzero momentum.
    j_scale = jnp.max(jnp.abs(j_up))
    j_unit = j_up / jnp.where(j_scale > 0.0, j_scale, 1.0)
    j2_unit = j_unit @ (g_ab @ j_unit)
    jhat = j_unit / jnp.sqrt(jnp.where(j2_unit > 0.0, j2_unit, 1.0))
    S_par = jhat @ (g_ab @ ((proj @ (T_mixed @ proj)) @ jhat))
    return rho, S_par, jmag


# Large boosts can make a Type-I eigenbasis nearly singular. Route such points
# to the LMI; this cutoff is a conditioning policy, not a rigorous error bound.
_EVEC_COND_MAX = 1e5


def ill_conditioned_eigenbasis(evecs, cond_max: float = _EVEC_COND_MAX):
    """True where the eigenvector matrix is too ill-conditioned to read."""
    sv = jnp.linalg.svd(evecs, compute_uv=False)
    return (sv[..., 0] / jnp.maximum(sv[..., -1], 1e-300)) > cond_max


def _exact_margins(he_type, nec_I, wec_I, sec_I, dec_I, witness, lmi, ill_conditioned=None):
    """Select numerical margins by algebraic type and eigenbasis conditioning.

    Well-conditioned Type-I points use eigenvalue-inequality slacks. All
    other points use :func:`.slemma.certify_point`; the NEC LMI margin is
    doubled to give the null deficit at Eulerian normalization.

    The Type-I NEC slack ``min_i(rho + p_i)`` uses eigenframe normalization.
    It agrees with the Eulerian null deficit when the two timelike frames
    coincide, but generally has a different magnitude. WEC, SEC, and DEC
    slacks likewise differ from their LMI counterparts. Compare signs with
    appropriate numerical tolerances, not magnitudes across the two routes.
    The ``witness`` argument does not determine any returned margin.
    """
    is_I = he_type == 1
    if ill_conditioned is not None:
        is_I = is_I & ~ill_conditioned
    # The LMI deficit uses Eulerian normalization; nec_I uses the eigenframe.
    nonI_nec = 2.0 * lmi["nec"]
    nec = jnp.where(is_I, nec_I, nonI_nec)
    wec = jnp.where(is_I, wec_I, lmi["wec"])
    sec = jnp.where(is_I, sec_I, lmi["sec"])
    dec = jnp.where(is_I, dec_I, lmi["dec"])
    return nec, wec, sec, dec


def certify_point_frame_free(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    g_inv: Float[Array, "4 4"] | None = None,
    *,
    solver: str = "auto",
    tol: float = 1e-10,
) -> dict:
    """Compute cap-free energy-condition margins at one spacetime point.

    The metric must have Lorentzian signature and spacelike coordinate slices.
    Marginal values require numerical error assessment; this function does not
    produce exact rational certificates.

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
        Classification tolerance, not an energy-condition error bound.

    Returns
    -------
    dict
        ``he_type`` (1-4), ``rho``, ``pressures``, ``eigenvalues``,
        ``eigenvalues_imag``, ``is_vacuum``, and ``nec``/``wec``/``sec``/``dec``
        margins. Margins use the LMI for non-Type-I points or ill-conditioned
        eigenbases. Rest-frame quantities are meaningful only at Type I.
    """
    if g_inv is None:
        g_inv = jnp.linalg.inv(g_ab)
    T_mixed = jnp.einsum("ac,cb->ab", g_inv, T_ab)
    cls = classify_with_solver(T_mixed, g_ab, T_ab, solver=solver, tol=tol)
    nec_I, wec_I, sec_I, dec_I = check_all(cls.rho, cls.pressures)
    witness = eulerian_null_witness(T_ab, g_ab, g_inv)
    nec, wec, sec, dec = _exact_margins(
        cls.he_type,
        nec_I,
        wec_I,
        sec_I,
        dec_I,
        witness,
        certify_point_lmi(T_ab, g_ab),
        ill_conditioned_eigenbasis(cls.eigenvectors),
    )
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
    lmi_where: Float[Array, "..."] | None = None,
) -> FrameFreeGridResult:
    """Compute cap-free energy-condition margins across an evaluation grid.

    Uses eigenvalue inequalities at well-conditioned Type-I points and the
    Eulerian tetrad LMI elsewhere. The metric must have Lorentzian signature
    and spacelike coordinate slices. See :func:`_exact_margins` for the
    normalization of each route.

    Parameters
    ----------
    T_field : Float[Array, "... 4 4"]
        Covariant stress-energy on a grid; leading dimensions are the grid shape.
    g_field, g_inv_field : Float[Array, "... 4 4"]
        Covariant and optional inverse metric on the same grid.
    solver : {"auto", "standard", "generalized"}
        Eigenvalue backend.
    tol : float
        Classification tolerance, not an energy-condition error bound.
    lmi_where : Float[Array, "..."] | None
        Boolean mask restricting LMI evaluation. Non-Type-I points and
        ill-conditioned Type-I points outside the mask receive NaN margins.
        Well-conditioned Type-I margins are computed everywhere. ``None``
        evaluates every required LMI.

    Returns
    -------
    FrameFreeGridResult
        Grid margins, type counts, and diagnostics. ``nec_noise_floor`` uses
        the scale of the returned NEC margins, twice the underlying LMI
        floor. ``lmi_substituted`` marks ill-conditioned eigenbases.
    """
    grid_shape = T_field.shape[:-2]
    flat_T = jnp.reshape(T_field, (-1, 4, 4))
    flat_g = jnp.reshape(g_field, (-1, 4, 4))
    if g_inv_field is None:
        flat_ginv = jax.vmap(jnp.linalg.inv)(flat_g)
    else:
        flat_ginv = jnp.reshape(g_inv_field, (-1, 4, 4))
    flat_Tmixed = jnp.einsum("nac,ncb->nab", flat_ginv, flat_T)

    cls = _classify_grid_batch(flat_Tmixed, flat_g, flat_T, solver=solver, tol=tol)
    nec_I, wec_I, sec_I, dec_I = jax.vmap(check_all)(cls.rho, cls.pressures)
    witness = jax.vmap(eulerian_null_witness)(flat_T, flat_g, flat_ginv)

    he = np.asarray(cls.he_type)
    # Evaluate the costlier LMI only where eigenvalue margins are unavailable.
    ill = np.asarray(jax.vmap(ill_conditioned_eigenbasis)(cls.eigenvectors))
    wanted = (he != 1) | ill
    if lmi_where is not None:
        wanted &= np.asarray(jnp.reshape(lmi_where, (-1,))).astype(bool)
    nonI = np.flatnonzero(wanted)
    unused = jnp.zeros_like(witness)
    if nonI.size:
        idx = jnp.asarray(nonI)
        sub = jax.vmap(certify_point_lmi)(flat_T[idx], flat_g[idx])
        lmi = {k: unused.at[idx].set(v) for k, v in sub.items()}
    else:
        lmi = {"nec": unused, "wec": unused, "sec": unused, "dec": unused}
    if lmi_where is not None:
        # Excluded points needing the LMI have no computed margin.
        nan = jnp.full_like(unused, jnp.nan)
        skipped = jnp.asarray(((he != 1) | ill) & ~wanted)
        lmi = {k: jnp.where(skipped, nan, v) for k, v in lmi.items()}
    nec, wec, sec, dec = jax.vmap(_exact_margins)(
        cls.he_type,
        nec_I,
        wec_I,
        sec_I,
        dec_I,
        witness,
        lmi,
        jnp.asarray(ill),
    )

    n_vacuum = int(np.sum(np.asarray(cls.is_vacuum) > 0.5))
    # nanmax: a NaN-sanitized eigenvalue must not poison the grid-wide
    # imaginary-part diagnostic (it is a summary, not a certified margin).
    max_imag = float(np.nanmax(np.abs(np.asarray(cls.eigenvalues_imag))))

    def _rs(x, trailing=()):  # reshape flat -> grid
        return jnp.reshape(x, (*grid_shape, *trailing))

    # Preserve the LMI tolerance for comparisons on both sides of zero.
    # The routed NEC slot is twice the LMI margin, so its floor has the same scale.
    nec_floor = 2 * jax.vmap(lambda T, g: noise_floor(T, g, condition="nec"))(flat_T, flat_g)

    return FrameFreeGridResult(
        he_types=_rs(cls.he_type),
        eigenvalues=_rs(cls.eigenvalues, (4,)),
        eigenvalues_imag=_rs(cls.eigenvalues_imag, (4,)),
        rho=_rs(cls.rho),
        pressures=_rs(cls.pressures, (3,)),
        nec_noise_floor=_rs(nec_floor),
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
        lmi_substituted=_rs(jnp.asarray(ill, dtype=nec.dtype)),
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
    sel = np.ones_like(he, dtype=float) if mask is None else np.asarray(mask).ravel().astype(float)
    w = sel if volume_weights is None else sel * np.asarray(volume_weights).ravel()
    wt = float(np.sum(w))
    if wt <= 0.0:
        return {f"frac_type_{k}": 0.0 for k in ("i", "ii", "iii", "iv")} | {"n_selected": 0}
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
    """Minimum eigenvalue-inequality slacks over selected Type-I points.

    These slacks are invariant under frame changes in exact arithmetic, but
    are not capped observer minima. Type-I points with ill-conditioned
    eigenbases carry LMI margins with a different normalization and are
    excluded. Returns NaN when a condition has no finite selected margin.
    """
    he = np.asarray(result.he_types).ravel()
    sel = np.ones_like(he, dtype=bool) if mask is None else np.asarray(mask).ravel().astype(bool)
    typeI = sel & (he == 1.0)
    if result.lmi_substituted is not None:
        typeI &= np.asarray(result.lmi_substituted).ravel() < 0.5
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
