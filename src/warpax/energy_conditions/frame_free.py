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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from .classification import _EVEC_COND_MAX as _EVEC_COND_MAX
from .classification import classify_with_solver, ill_conditioned_eigenbasis
from .eigenvalue_checks import check_all
from .observer import eulerian_normal
from .slemma import _NOISE_REL, noise_floor
from .slemma import certify_point as certify_point_lmi
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

    ``h^a_b = delta^a_b + n^a n_b`` projects onto the slice and
    ``j^a = -h^a_b T^b_c n^c`` is the spatial momentum density; ``jhat`` is
    normalized by the largest component first to preserve small nonzero momentum.
    """
    _, n_up = eulerian_normal(g_inv)
    n_low2 = g_ab @ n_up
    proj = jnp.eye(4) + jnp.outer(n_up, n_low2)
    T_mixed = g_inv @ T_ab
    rho = n_up @ (T_ab @ n_up)
    j_up = -(proj @ (T_mixed @ n_up))
    j2 = j_up @ (g_ab @ j_up)
    jmag = jnp.sqrt(jnp.clip(j2, min=0.0))
    j_scale = jnp.max(jnp.abs(j_up))
    j_unit = j_up / jnp.where(j_scale > 0.0, j_scale, 1.0)
    j2_unit = j_unit @ (g_ab @ j_unit)
    jhat = j_unit / jnp.sqrt(jnp.where(j2_unit > 0.0, j2_unit, 1.0))
    S_par = jhat @ (g_ab @ ((proj @ (T_mixed @ proj)) @ jhat))
    return rho, S_par, jmag


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


@jax.jit
def _mixed_batch(flat_T, flat_g, flat_ginv):
    """Inverse metric (when absent) and mixed stress-energy over a flat batch."""
    if flat_ginv is None:
        flat_ginv = jax.vmap(jnp.linalg.inv)(flat_g)
    return flat_ginv, jnp.einsum("nac,ncb->nab", flat_ginv, flat_T)


@jax.jit
def _margins_batch(he_type, eigenvalues, rho, pressures, evecs, flat_T, flat_g, flat_ginv):
    """Type-I slacks, Eulerian witness, conditioning flag and routed NEC floor per point.

    The floor follows the route of the returned NEC margin: well-conditioned
    Type-I points get ``_NOISE_REL * max|lambda|`` on the eigenframe slack,
    every other point twice the Eulerian LMI floor of :func:`.slemma.null_deficit`.
    """
    slacks = jax.vmap(check_all)(rho, pressures)
    witness = jax.vmap(eulerian_null_witness)(flat_T, flat_g, flat_ginv)
    ill = jax.vmap(ill_conditioned_eigenbasis)(evecs)
    lmi_floor = jax.vmap(lambda T, g: noise_floor(T, g, condition="nec"))(flat_T, flat_g)
    eig_floor = _NOISE_REL * jnp.max(jnp.abs(eigenvalues), axis=-1)
    nec_floor = jnp.where((he_type == 1) & ~ill, eig_floor, 2 * lmi_floor)
    return slacks, witness, ill, nec_floor


@eqx.filter_jit
def _vmap_batch(point_fn, *args):
    """Jitted ``vmap(point_fn)``; ``point_fn`` is looked up by the caller at call time."""
    return jax.vmap(point_fn)(*args)


def certify_grid_frame_free(
    T_field: Float[Array, "... 4 4"],
    g_field: Float[Array, "... 4 4"],
    g_inv_field: Float[Array, "... 4 4"] | None = None,
    *,
    solver: str = "auto",
    tol: float = 1e-10,
    lmi_where: Bool[Array, "..."] | Bool[np.ndarray, "..."] | None = None,
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
    lmi_where : Bool[Array, "..."] | Bool[np.ndarray, "..."] | None
        Boolean mask restricting LMI evaluation. Non-Type-I points and
        ill-conditioned Type-I points outside the mask receive NaN margins.
        Well-conditioned Type-I margins are computed everywhere. ``None``
        evaluates every required LMI.

    Returns
    -------
    FrameFreeGridResult
        Grid margins, type counts, and diagnostics. ``nec_noise_floor`` is
        matched to the route of each NEC margin: the eigenframe scale
        ``1e-12 * max|lambda|`` at well-conditioned Type-I points and twice
        the Eulerian LMI floor elsewhere, so comparisons keep the tolerance
        of the returned margin on both sides of zero.
        ``lmi_substituted`` marks ill-conditioned eigenbases. The LMI runs
        only where eigenvalue margins are unavailable; points excluded by
        ``lmi_where`` that need it receive NaN. ``max_imag_eigenvalue`` is a
        ``nanmax`` summary, not a certified margin.
    """
    grid_shape = T_field.shape[:-2]
    flat_T = jnp.reshape(T_field, (-1, 4, 4))
    flat_g = jnp.reshape(g_field, (-1, 4, 4))
    flat_ginv = None if g_inv_field is None else jnp.reshape(g_inv_field, (-1, 4, 4))
    flat_ginv, flat_Tmixed = _mixed_batch(flat_T, flat_g, flat_ginv)

    cls = _classify_grid_batch(flat_Tmixed, flat_g, flat_T, solver=solver, tol=tol)
    (nec_I, wec_I, sec_I, dec_I), witness, ill_dev, nec_floor = _margins_batch(
        cls.he_type,
        cls.eigenvalues,
        cls.rho,
        cls.pressures,
        cls.eigenvectors,
        flat_T,
        flat_g,
        flat_ginv,
    )

    he = np.asarray(cls.he_type)
    ill = np.asarray(ill_dev)
    wanted = (he != 1) | ill
    if lmi_where is not None:
        wanted &= np.asarray(jnp.reshape(lmi_where, (-1,))).astype(bool)
    nonI = np.flatnonzero(wanted)
    unused = jnp.zeros_like(witness)
    if nonI.size:
        idx = jnp.asarray(nonI)
        sub = _vmap_batch(certify_point_lmi, flat_T[idx], flat_g[idx])
        lmi = {k: unused.at[idx].set(v) for k, v in sub.items()}
    else:
        lmi = {"nec": unused, "wec": unused, "sec": unused, "dec": unused}
    if lmi_where is not None:
        nan = jnp.full_like(unused, jnp.nan)
        skipped = jnp.asarray(((he != 1) | ill) & ~wanted)
        lmi = {k: jnp.where(skipped, nan, v) for k, v in lmi.items()}
    nec, wec, sec, dec = _vmap_batch(
        _exact_margins, cls.he_type, nec_I, wec_I, sec_I, dec_I, witness, lmi, ill_dev
    )

    n_vacuum = int(np.sum(np.asarray(cls.is_vacuum) > 0.5))
    max_imag = float(np.nanmax(np.abs(np.asarray(cls.eigenvalues_imag))))

    def _rs(x, trailing=()):
        return jnp.reshape(x, (*grid_shape, *trailing))

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
    mask: Bool[Array, "..."] | Bool[np.ndarray, "..."] | None = None,
    volume_weights: Float[Array, "..."] | Float[np.ndarray, "..."] | None = None,
) -> dict[str, float]:
    """Volume-weighted Hawking-Ellis type fractions (optionally wall-restricted).

    Parameters
    ----------
    result : FrameFreeGridResult
    mask : Bool[Array, "..."] | Bool[np.ndarray, "..."] | None
        Boolean selection (e.g. the wall mask). ``None`` selects all points.
    volume_weights : Float[Array, "..."] | Float[np.ndarray, "..."] | None
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
    mask: Bool[Array, "..."] | Bool[np.ndarray, "..."] | None = None,
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
