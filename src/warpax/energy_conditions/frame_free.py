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
  WEC/SEC/DEC. When the complex pair is momentum-sourced the Eulerian null vector
  ``k = n +/- jhat`` witnesses it in closed form,
  ``T_ab k^a k^b = rho + S_par - 2|j| < 0``, which holds when
  ``Delta = (rho + S_par)^2 - 4|j|^2 < 0``. A conformal Type-IV pair can leave the
  momentum witness >= 0; the point is still certified Type IV, so ``-|Im lambda|``
  is used as the margin. See :func:`eulerian_null_witness` and
  :func:`_exact_margins`.
- Type II (null eigenvector): the same null contraction decides NEC and can be
  positive, so its margin is reported directly.

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


def _exact_margins(he_type, nec_I, wec_I, sec_I, dec_I, witness, imag):
    """Select cap-free EC margins by Hawking-Ellis type (branchless for vmap).

    Type I -> eigenvalue-inequality margins (exact, necessary & sufficient).
    Type II -> the Eulerian null witness (decides NEC; can be >= 0).
    Type III/IV (no causal eigenvector, NEC violated) -> the momentum witness when
    it is negative; when a conformal/transverse pair leaves witness >= 0 the point
    is still certified violating, so the margin is forced negative: -|Im lambda|
    for Type IV, a tiny negative sentinel for the degenerate Type III (imag = 0).
    No NaN, no cap, no optimizer.
    """
    is_I = he_type == 1
    is_III = he_type == 3
    is_IV = he_type == 4
    # Off Type I the Type-I eigenvalue margins are undefined (no rest frame). The
    # momentum witness carries the certification when it is negative; a
    # conformal/transverse Type-III/IV pair (witness >= 0) is still violating, so
    # -max(|Im lambda|, tiny) forces a certified-negative margin.
    force = (is_III | is_IV) & (witness >= 0.0)
    nonI = jnp.where(force, -jnp.maximum(imag, 1e-30), witness)
    nec = jnp.where(is_I, nec_I, nonI)
    wec = jnp.where(is_I, wec_I, nonI)
    sec = jnp.where(is_I, sec_I, nonI)
    dec = jnp.where(is_I, dec_I, nonI)
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
        ``nec``/``wec``/``sec``/``dec`` margins (certified witness if non-Type-I),
        ``eigenvalues``, ``eigenvalues_imag``, ``is_vacuum``.
    """
    if g_inv is None:
        g_inv = jnp.linalg.inv(g_ab)
    T_mixed = jnp.einsum("ac,cb->ab", g_inv, T_ab)
    cls = classify_with_solver(T_mixed, g_ab, T_ab, solver=solver, tol=tol)
    nec_I, wec_I, sec_I, dec_I = check_all(cls.rho, cls.pressures)
    witness = eulerian_null_witness(T_ab, g_ab, g_inv)
    imag = jnp.max(jnp.abs(cls.eigenvalues_imag))
    nec, wec, sec, dec = _exact_margins(
        cls.he_type, nec_I, wec_I, sec_I, dec_I, witness, imag
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
    nec_I, wec_I, sec_I, dec_I = jax.vmap(check_all)(cls.rho, cls.pressures)
    witness = jax.vmap(eulerian_null_witness)(flat_T, flat_g, flat_ginv)
    imag = jnp.max(jnp.abs(cls.eigenvalues_imag), axis=-1)
    nec, wec, sec, dec = jax.vmap(_exact_margins)(
        cls.he_type, nec_I, wec_I, sec_I, dec_I, witness, imag
    )

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
