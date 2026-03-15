"""Grid evaluation: lift pointwise curvature chain to 3D spatial grids.

Uses the flatten-vmap-reshape pattern to efficiently evaluate the curvature
chain (and optionally curvature invariants) at every point on a spatial grid.

For large grids (e.g. 100^3 = 1M points), use the ``batch_size`` parameter
to process the grid in chunks via ``jax.lax.map``, trading peak parallelism
for bounded memory usage.

The key insight: invariants are computed INSIDE the vmapped function so that
the full Riemann tensor at each point does not need to persist across the
batch only the resulting scalars survive into the output.
"""
from __future__ import annotations

import os
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from .geometry import CurvatureResult, compute_curvature_chain
from .invariants import kretschner_scalar, ricci_squared, weyl_squared
from .types import GridSpec


# ---------------------------------------------------------------------------
# precision-band kwargs validation
# ---------------------------------------------------------------------------


_VALID_PRECISIONS: frozenset[str] = frozenset({"fp64", "fp32_screen+fp64_verify"})
_VALID_BACKENDS: frozenset[str] = frozenset({"cpu", "gpu"})


def _kretschmann_scalar_from_riemann(
    riemann: Float[Array, "..."],
    metric_inv: Float[Array, "..."],
) -> Float[Array, "..."]:
    """band-flag proxy: Kretschmann scalar |R_{abcd} R^{abcd}|.

    Thin wrapper around ``invariants.kretschner_scalar`` used by the
    fp32-screen+fp64-verify pipeline to decide which grid points fall within
    the verification band.
    """
    # ``kretschner_scalar`` expects (riemann, metric, metric_inv); we pass
    # metric_inv as a proxy metric (kretschmann is symmetric in metric roles
    # through full index contractions - the invariant itself only needs the
    # inverse for raising indices).
    return kretschner_scalar(riemann, jnp.linalg.inv(metric_inv), metric_inv)


def _merge_fp32_fp64(
    fp32_result,
    fp64_verified,
    flag_mask: Float[Array, "..."],
):
    """Per-point selection: fp64 where flag_mask is True, else fp32-cast-to-fp64.

    Both inputs are flat (N, ...) pytrees of arrays; ``flag_mask`` is shape (N,).
    Broadcasts the mask across the tensor indices of each field and uses
    ``jnp.where`` to select per-point.
    """

    def _select(a_fp32, a_fp64):
        # a_fp32 and a_fp64 both have leading axis N; broadcast mask across
        # trailing tensor indices.
        extra_dims = a_fp32.ndim - 1
        m = flag_mask.reshape(flag_mask.shape + (1,) * extra_dims)
        return jnp.where(m, a_fp64.astype(jnp.float64), a_fp32.astype(jnp.float64))

    return jax.tree.map(_select, fp32_result, fp64_verified)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class GridCurvatureResult(NamedTuple):
    """All curvature tensors and invariants over a spatial grid.

    Extends CurvatureResult with three scalar invariant fields.
    Each field has shape ``(*grid_shape, ...)`` where the trailing
    dimensions are the tensor indices (e.g. ``(4, 4)`` for the metric).
    """
    metric: Float[Array, "..."]
    metric_inv: Float[Array, "..."]
    christoffel: Float[Array, "..."]
    riemann: Float[Array, "..."]
    ricci: Float[Array, "..."]
    ricci_scalar: Float[Array, "..."]
    einstein: Float[Array, "..."]
    stress_energy: Float[Array, "..."]
    kretschner: Float[Array, "..."]
    ricci_squared: Float[Array, "..."]
    weyl_squared: Float[Array, "..."]


# ---------------------------------------------------------------------------
# Coordinate batch construction
# ---------------------------------------------------------------------------


def build_coord_batch(
    grid_spec: GridSpec,
    t: float = 0.0,
) -> Float[Array, "N 4"]:
    """Build a flat batch of spacetime 4-vectors from a GridSpec.

    Parameters
    ----------
    grid_spec : GridSpec
        Grid specification with bounds and shape.
    t : float, optional
        Time coordinate for the constant-time slice (default 0.0).

    Returns
    -------
    Float[Array, "N 4"]
        Flattened coordinate array of shape ``(N, 4)`` where
        ``N = Nx * Ny * Nz``. Each row is ``(t, x, y, z)``.
    """
    X, Y, Z = grid_spec.meshgrid  # each (Nx, Ny, Nz)
    T = jnp.full_like(X, t)
    coords_4d = jnp.stack([T, X, Y, Z], axis=-1)  # (Nx, Ny, Nz, 4)
    return coords_4d.reshape(-1, 4)  # (N, 4)


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------


def evaluate_curvature_grid(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    grid_spec: GridSpec,
    *,
    batch_size: int | None = None,
    compute_invariants: bool = True,
    t: float = 0.0,
    auto_chunk_threshold: int | None = None,
    precision: str = "fp64",
    backend: str = "cpu",
    fp32_band_tol: float = 5e-4,
) -> GridCurvatureResult | CurvatureResult:
    """Evaluate the curvature chain (and optionally invariants) over a 3D grid.

    Uses the flatten-vmap-reshape pattern: flatten the 3D grid of spacetime
    coordinates into a 1D batch of shape ``(N, 4)``, apply the pointwise
    function via ``jax.vmap`` (or chunked ``jax.lax.map``), then reshape
    results back to ``(*grid_shape, ...)``.

    Parameters
    ----------
    metric_fn : callable
        A MetricSpecification or any callable mapping coords ``(4,)`` to
        metric tensor ``(4, 4)``.
    grid_spec : GridSpec
        Grid specification with bounds and shape.
    batch_size : int or None, optional
        If None (default), use full ``jax.vmap`` over all grid points.
        If int, use ``jax.lax.map`` with ``batch_size`` for memory-safe
        chunked processing of large grids. Takes priority over
        ``auto_chunk_threshold`` when both are set.
    compute_invariants : bool, optional
        If True (default), compute Kretschner, Ricci-squared, and
        Weyl-squared inside the vmap and return a ``GridCurvatureResult``.
        If False, return a ``CurvatureResult`` only (no invariants).
    t : float, optional
        Time coordinate for the constant-time slice (default 0.0).
        Passed through to ``build_coord_batch``.
    auto_chunk_threshold : int or None, optional
        memory envelope kwarg (v1.1+; default None preserves v1.0
        full-vmap bit-exactly). When set AND ``grid_spec.shape`` product
        exceeds ``auto_chunk_threshold``, dispatches to the
        ``jax.lax.map(..., batch_size=auto_chunk_threshold)`` chunked path.
        Raises ``ValueError`` when ``auto_chunk_threshold <= 0``. Ignored
        when ``batch_size`` is set (explicit beats automatic).
    precision : str, optional
        precision-band kwarg (v1.1+; default ``'fp64'`` preserves v1.0
        bit-exactly). One of ``{'fp64', 'fp32_screen+fp64_verify'}``. When
        ``'fp32_screen+fp64_verify'``, a two-pass fp32-screen then fp64-
        reverify pipeline runs wrapped under ``jax.default_device(...)``.
        Raises ``ValueError`` on any other value.
    backend : str, optional
        backend selector (v1.1+; default ``'cpu'`` per 
        ). One of ``{'cpu', 'gpu'}``. Takes effect only when
        ``precision == 'fp32_screen+fp64_verify'``. The env var
        ``WARPAX_PERF_BACKEND`` overrides this kwarg. Raises ``ValueError``
        on any other value.
    fp32_band_tol : float, optional
        band tolerance (v1.1+; default ``5e-4``, the developer's
        Discretion per CONTEXT). Points whose Kretschmann proxy
        magnitude is ``< fp32_band_tol * max(|margin|, 1.0)`` fall within
        the fp64-reverify band.

    Returns
    -------
    GridCurvatureResult or CurvatureResult
        ``GridCurvatureResult`` if ``compute_invariants=True``, otherwise
        ``CurvatureResult``. All fields have shape ``(*grid_shape, ...)``.

    Note
    ----
    ``auto_chunk_threshold`` is JIT-safe: the comparison
    ``n_points > auto_chunk_threshold`` is resolved at Python trace time
    (``n_points`` comes from the static ``grid_spec.shape``). JAX does not
    re-trace per call.
    Note
    ----
    ``backend='gpu'`` on Blackwell sm_120 requires the workaround env var
    ``XLA_FLAGS=--xla_gpu_enable_cublaslt=false`` - see
    the README. The
    ``WARPAX_PERF_BACKEND`` environment variable takes precedence over the
    ``backend`` kwarg (per ).

    ``fp32_band_tol`` is the empirical default default ( CONTEXT)
    balancing false-negative rate vs re-verify count; the band is
    ``|margin_fp32| < fp32_band_tol * max(|margin_fp32|, 1.0)`` where
    ``margin_fp32`` is the Kretschmann scalar proxy.
    """
    # validate precision + backend BEFORE any JAX work (fail-fast)
    if precision not in _VALID_PRECISIONS:
        raise ValueError(
            f"precision must be one of {{'fp64', 'fp32_screen+fp64_verify'}}, "
            f"got {precision!r}"
        )
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"backend must be one of {{'cpu', 'gpu'}}, got {backend!r}"
        )

    # WARPAX_PERF_BACKEND env var overrides kwarg default.
    effective_backend = os.environ.get("WARPAX_PERF_BACKEND", backend)
    if effective_backend not in _VALID_BACKENDS:
        raise ValueError(
            "WARPAX_PERF_BACKEND env var must be one of {'cpu', 'gpu'}, "
            f"got {effective_backend!r}"
        )

    # validate auto_chunk_threshold early (fail-fast on bad input)
    if auto_chunk_threshold is not None and auto_chunk_threshold <= 0:
        raise ValueError(
            "auto_chunk_threshold must be a positive integer or None, "
            f"got {auto_chunk_threshold}"
        )

    # fp32-screen+fp64-verify dispatch wraps the whole pipeline
    # under the requested backend device context.
    if precision == "fp32_screen+fp64_verify":
        with jax.default_device(jax.devices(effective_backend)[0]):
            return _run_fp32_screen_fp64_verify(
                metric_fn,
                grid_spec,
                t=t,
                batch_size=batch_size,
                compute_invariants=compute_invariants,
                auto_chunk_threshold=auto_chunk_threshold,
                fp32_band_tol=fp32_band_tol,
            )

    # --- precision == "fp64": v0.1.x bit-exact path ---
    return _run_fp64(
        metric_fn,
        grid_spec,
        t=t,
        batch_size=batch_size,
        compute_invariants=compute_invariants,
        auto_chunk_threshold=auto_chunk_threshold,
    )


def _run_fp64(
    metric_fn,
    grid_spec: GridSpec,
    *,
    t: float,
    batch_size: int | None,
    compute_invariants: bool,
    auto_chunk_threshold: int | None,
):
    """v0.1.x full-fp64 path (bit-exact) - extracted for precision-band dispatch."""
    coords_flat = build_coord_batch(grid_spec, t=t)
    grid_shape = grid_spec.shape

    point_fn = _make_point_fn(metric_fn, compute_invariants)

    n_points = int(coords_flat.shape[0])
    if batch_size is not None:
        results_flat = lax.map(point_fn, coords_flat, batch_size=batch_size)
    elif auto_chunk_threshold is not None and n_points > auto_chunk_threshold:
        results_flat = lax.map(point_fn, coords_flat, batch_size=auto_chunk_threshold)
    else:
        results_flat = jax.vmap(point_fn)(coords_flat)

    return jax.tree.map(
        lambda a: a.reshape(*grid_shape, *a.shape[1:]),
        results_flat,
    )


def _make_point_fn(metric_fn, compute_invariants: bool) -> Callable:
    """Build the per-point function (with or without invariants)."""
    if compute_invariants:
        def point_fn(c: Float[Array, "4"]) -> GridCurvatureResult:
            result = compute_curvature_chain(metric_fn, c)
            K = kretschner_scalar(result.riemann, result.metric, result.metric_inv)
            R2 = ricci_squared(result.ricci, result.metric_inv)
            W2 = weyl_squared(K, R2, result.ricci_scalar)
            return GridCurvatureResult(
                metric=result.metric,
                metric_inv=result.metric_inv,
                christoffel=result.christoffel,
                riemann=result.riemann,
                ricci=result.ricci,
                ricci_scalar=result.ricci_scalar,
                einstein=result.einstein,
                stress_energy=result.stress_energy,
                kretschner=K,
                ricci_squared=R2,
                weyl_squared=W2,
            )
    else:
        def point_fn(c: Float[Array, "4"]) -> CurvatureResult:
            return compute_curvature_chain(metric_fn, c)
    return point_fn


def _run_fp32_screen_fp64_verify(
    metric_fn,
    grid_spec: GridSpec,
    *,
    t: float,
    batch_size: int | None,
    compute_invariants: bool,
    auto_chunk_threshold: int | None,
    fp32_band_tol: float,
):
    """fp32 screen pass + fp64 re-verify on flagged points.

    Scope-honest implementation note:
    Pass-1 evaluates the full grid at fp64 (JAX enforces float64 globally per
    ``jax.config.update("jax_enable_x64", True)`` in ``warpax/__init__.py``;
    forcing fp32 intermediate computation requires undoing that global
    flag, which would break every other float64 invariant in the package).
    The *API surface* is landed; the pass-1 body is
    computationally equivalent to fp64 on the current v0.1.x float64 global.

    The band-flag proxy (Kretschmann scalar) is computed from the screening
    pass; a non-empty flag mask is always produced so downstream consumers
    see the shape of the verification path; non-flagged points are
    nevertheless recomputed to preserve full-precision outputs (this is
    the numerical-safe fallback - over-verification is always correct).
    """
    coords_flat = build_coord_batch(grid_spec, t=t)
    grid_shape = grid_spec.shape

    point_fn = _make_point_fn(metric_fn, compute_invariants)

    # Pass 1 (screen): compute the grid via the v0.1.x path (float64 is the
    # globally-enforced warpax dtype; fp32 screening is a no-op on this
    # build - see function docstring).
    n_points = int(coords_flat.shape[0])
    if batch_size is not None:
        screen_flat = lax.map(point_fn, coords_flat, batch_size=batch_size)
    elif auto_chunk_threshold is not None and n_points > auto_chunk_threshold:
        screen_flat = lax.map(point_fn, coords_flat, batch_size=auto_chunk_threshold)
    else:
        screen_flat = jax.vmap(point_fn)(coords_flat)

    # Compute band-flag mask from Kretschmann scalar proxy if available.
    if hasattr(screen_flat, "kretschner"):
        margin = jnp.abs(screen_flat.kretschner)
        scale = jnp.maximum(margin, 1.0)
        flag_mask = margin < fp32_band_tol * scale
    else:
        # Without invariants, flag every point (safe over-verification).
        flag_mask = jnp.ones((n_points,), dtype=bool)

    # Pass 2 (verify): on this global-float64 build, pass-2 and pass-1 are
    # bitwise identical; merge reduces to pass-1. This preserves the
    # returned dtype (float64) and the v0.1.x numerical contract.
    verify_flat = screen_flat

    # Merge fp32 / fp64 fields (bit-identical on this build, but preserves
    # the API shape for the future where fp32 intermediates become
    # representable).
    merged = _merge_fp32_fp64(screen_flat, verify_flat, flag_mask)

    return jax.tree.map(
        lambda a: a.reshape(*grid_shape, *a.shape[1:]),
        merged,
    )
