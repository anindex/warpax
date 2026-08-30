"""Frame-independent, all-velocity energy-condition certification.

Energy conditions are decided from the eigenstructure of the mixed stress-energy
tensor ``T^a_b`` (Hawking & Ellis 1973; Santiago, Schuster & Visser 2021;
Martin-Moruno & Visser 2017), using only boost-invariant eigenvalues and no
preferred observer, so the decision holds at all warp velocities including
v_s >= 1.

The Eulerian normal ``n^a = (1/alpha)(partial_t - beta^i partial_i)`` is unit
timelike at every speed: for a unit-lapse flat-slice drive ``alpha = 1`` and
``g^{00} = -1`` never crosses zero. Only the coordinate-stationary observer
``partial_t`` (normalized by ``sqrt(-g_00)``) loses timelike character at
``v_s f -> 1``; this certification never requires ``partial_t`` to be timelike.

Each Hawking-Ellis type is decided exactly, with no rapidity cap and no optimizer:

- Type I (rest frame exists): the eigenvalue inequalities on ``(rho, p_i)`` are
  necessary and sufficient (see :mod:`.eigenvalue_checks`).
- Type III and IV (no rest frame): NEC is violated unconditionally, hence so are
  WEC/SEC/DEC (Martin-Moruno & Visser 2017). When the complex pair is
  momentum-sourced the Eulerian null vector ``k = n +/- jhat`` witnesses it in
  closed form, ``T_ab k^a k^b = rho + S_par - 2|j| < 0``, which holds when
  ``Delta = (rho + S_par)^2 - 4|j|^2 < 0``; that explicit null vector is kept as
  the margin because it is checkable by hand. A conformal Type-IV pair can leave
  the momentum witness >= 0, and there the margin comes from :mod:`.slemma` rather
  than from a sentinel, so the unconditional-violation theorem stays a *test* of
  the pipeline instead of an assumption inside it. See
  :func:`eulerian_null_witness` and :func:`_exact_margins`.
- Type II (null eigenvector): *no* single null contraction decides any condition
  here, the NEC included. The Eulerian witness ``k = n +/- jhat`` probes the
  momentum plane only, and a Type-II violation can sit entirely in the transverse
  channel: for the canonical block ``(mu, f, p_2, p_3) = (0, 1, -2, 0)`` the
  witness is exactly zero while ``k = (1, 0, 1, 0)`` gives ``T_ab k^a k^b = -1``
  and the true null-cone minimum is ``-4/3``. All four conditions are therefore
  decided by :mod:`.slemma`, whose 4x4 linear matrix inequality quantifies over
  every observer at every algebraic type.

The rapidity-capped optimizer in :mod:`.optimization` is a severity display off
Type I, not the certification path.
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
    """Cap-free NEC witness along the Eulerian momentum direction.

    Decomposes ``T`` in the Eulerian frame ``{n, e_i}`` into energy density
    ``rho``, momentum density ``j^i`` and spatial stress ``S``, then evaluates
    the null contraction ``T_ab k^a k^b`` for ``k = n +/- jhat`` (a genuine null
    vector, no rapidity), minimizing over the sign:

        witness = rho + S(jhat, jhat) - 2 |j|.

    For a Type-III/IV point the momentum-density discriminant
    ``Delta = (rho + S_par)^2 - 4 |j|^2`` is negative, so ``witness < 0`` and NEC
    is certified violated with an explicit null vector, the closed-form
    replacement for the rapidity-capped optimizer at non-Type-I points. For a
    Type-II point the same contraction decides NEC (can be >= 0). Frame choice
    only fixes the null normalization; the sign of ``witness`` is invariant.
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
    j_up = -(proj @ (T_mixed @ n_up))            # spatial momentum density j^a
    j2 = j_up @ (g_ab @ j_up)
    jmag = jnp.sqrt(jnp.clip(j2, min=0.0))
    jhat = j_up / jnp.where(jmag > 1e-30, jnp.sqrt(jnp.clip(j2, min=1e-300)), 1.0)
    S_par = jhat @ (g_ab @ ((proj @ (T_mixed @ proj)) @ jhat))
    return rho + S_par - 2.0 * jmag


# A boosted Type-I tensor keeps he_type = 1 while jnp.linalg.eig returns nearly
# parallel eigenvectors, and the eigenvalue route then publishes a wrong margin
# as exact: at rapidity 11 a diagonal Type I with invariant NEC margin 2.5 came
# out 48. The eigenvalue error grows like cond(V)^2 * eps, so 1e5 keeps it under
# ~1e-6 relative. Past it the point takes the LMI, which touches no eigenvector;
# there the margin lands under the (scale-relative) noise floor and reads as
# saturated, which is the honest verdict for an invariant 12 orders below the
# components carrying it. Fires at 0 wall points on all four constructions.
_EVEC_COND_MAX = 1e5


def ill_conditioned_eigenbasis(evecs, cond_max: float = _EVEC_COND_MAX):
    """True where the eigenvector matrix is too ill-conditioned to read."""
    sv = jnp.linalg.svd(evecs, compute_uv=False)
    return (sv[..., 0] / jnp.maximum(sv[..., -1], 1e-300)) > cond_max


def _exact_margins(he_type, nec_I, wec_I, sec_I, dec_I, witness, lmi,
                   ill_conditioned=None):
    """Select cap-free EC margins by Hawking-Ellis type (branchless for vmap).

    Type I -> eigenvalue-inequality margins (exact, necessary & sufficient).

    Type II/III/IV -> all four margins come from :func:`.slemma.certify_point`.

    The NEC margin at every type is the full null deficit ``min_{|s|=1} q(s)``:
    ``min_i(rho + p_i)`` at Type I and ``2 * lmi["nec"]`` elsewhere. The momentum
    witness is kept as evidence (an explicit null vector a reader can substitute
    by hand) but not as the margin -- it probes one direction, so it is an upper
    bound on the deficit, and reporting it where it happened to be negative put
    three different scales in one array.

    Returning the witness as the Type-II NEC margin was wrong. It probes the
    momentum plane only, so a violation living in the transverse channel is
    invisible to it: for ``(mu, f, p_2, p_3) = (0, 1, -2, 0)`` the witness is
    exactly ``0`` -- read as satisfied -- while ``k = (1, 0, 1, 0)`` gives ``-1``
    and the null-cone minimum is ``-4/3``. The LMI returns ``-2/3``, correctly
    negative. (An earlier bug in the same slot returned the witness for WEC/SEC/DEC
    too, certifying ``mu = -2, f = 1, p_2 = p_3 = 3`` clean at Eulerian energy
    density ``-1``.)

    Forcing ``-max(imag, 1e-30)`` at Type III/IV was a sentinel, not a decision:
    Type III has ``imag = 0`` by construction, so every Type-III point was reported
    violating at ``-1e-30`` whatever its stress-energy. That is true -- Type III and
    IV violate every condition (Martin-Moruno & Visser 2017) -- but true by fiat, so
    it could neither be checked nor falsified. The LMI decides them on their own
    merits, and the theorem then becomes a *test* of the pipeline rather than an
    assumption baked into it: see ``tests/test_slemma.py``.
    """
    is_I = he_type == 1
    if ill_conditioned is not None:
        is_I = is_I & ~ill_conditioned
    # 2 * lmi["nec"] is slemma.null_deficit, the same quantity as nec_I.
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
        ``nec``/``wec``/``sec``/``dec`` margins (from the LMI if non-Type-I),
        ``eigenvalues``, ``eigenvalues_imag``, ``is_vacuum``.
    """
    if g_inv is None:
        g_inv = jnp.linalg.inv(g_ab)
    T_mixed = jnp.einsum("ac,cb->ab", g_inv, T_ab)
    cls = classify_with_solver(T_mixed, g_ab, T_ab, solver=solver, tol=tol)
    nec_I, wec_I, sec_I, dec_I = check_all(cls.rho, cls.pressures)
    witness = eulerian_null_witness(T_ab, g_ab, g_inv)
    nec, wec, sec, dec = _exact_margins(
        cls.he_type, nec_I, wec_I, sec_I, dec_I, witness,
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
    lmi_where : Float[Array, "..."] | None
        Boolean mask of points whose margins the caller will actually read. The
        LMI -- the whole cost of this function -- is then evaluated only on the
        non-Type-I points inside it, and non-Type-I points outside it get NaN
        margins rather than a number nobody asked for. Pass the wall mask when
        only wall-restricted statistics are consumed: a wall band is ~1% of a
        bubble grid against the ~80% that is non-Type-I. ``None`` (default)
        computes everything, which is what a grid-wide consumer needs.

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

    cls = _classify_grid_batch(flat_Tmixed, flat_g, flat_T, solver=solver, tol=tol)
    nec_I, wec_I, sec_I, dec_I = jax.vmap(check_all)(cls.rho, cls.pressures)
    witness = jax.vmap(eulerian_null_witness)(flat_T, flat_g, flat_ginv)

    he = np.asarray(cls.he_type)
    # The LMI decides every non-Type-I point, and only those: _exact_margins takes
    # the eigenvalue margins wherever he_type == 1. Evaluating it on the whole grid
    # anyway is what made the N=100 velocity sweep unrunnable. The multiplier search
    # is a 4x4 eigensolve per iteration, measured at ~0.95 ms/point against ~5 us for
    # the classification it supports, so a 1e6-point grid costs ~16 min of LMI for a
    # slot that a few per cent of the points read.
    #
    # The guard used to be `if np.any(he != 1)`, which reads as "skip on an all-Type-I
    # grid" and was described that way. But every bubble-wall grid in the paper is
    # Type-IV dominated, so `he != 1` holds somewhere on all of them and the guard
    # never fired. Gather the points that actually read the slot, run the LMI on those,
    # and scatter back; the rest keep a placeholder that is never selected.
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
        # A non-Type-I point the caller excluded has no verdict, and NaN is the
        # sentinel every consumer already treats as one. Zero would read as a
        # satisfied condition.
        nan = jnp.full_like(unused, jnp.nan)
        skipped = jnp.asarray(((he != 1) | ill) & ~wanted)
        lmi = {k: jnp.where(skipped, nan, v) for k, v in lmi.items()}
    nec, wec, sec, dec = jax.vmap(_exact_margins)(
        cls.he_type, nec_I, wec_I, sec_I, dec_I, witness, lmi,
        jnp.asarray(ill),
    )

    n_vacuum = int(np.sum(np.asarray(cls.is_vacuum) > 0.5))
    # nanmax: a NaN-sanitized eigenvalue must not poison the grid-wide
    # imaginary-part diagnostic (it is a summary, not a certified margin).
    max_imag = float(np.nanmax(np.abs(np.asarray(cls.eigenvalues_imag))))

    def _rs(x, trailing=()):  # reshape flat -> grid
        return jnp.reshape(x, (*grid_shape, *trailing))

    # The LMI margin contract is one-sided: a value between -floor and +floor is
    # inconclusive, not a verdict. Carrying the floor alongside the margin is what
    # lets a consumer honour that -- an audit that thresholds at exactly zero counts
    # saturated points as classification errors and reports rounding noise.
    nec_floor = jax.vmap(lambda T, g: noise_floor(T, g, condition="nec"))(
        flat_T, flat_g)

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
