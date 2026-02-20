"""Metric specification, ADM decomposition, and SymPy-JAX bridge.

Provides the abstract ``MetricSpecification`` (Equinox module / JAX pytree),
the ``ADMMetric`` 3+1 decomposition subclass, the ``SymbolicMetric`` class
for SymPy-based symbolic inspection, and utility functions for converting
between representations.

NumPy reference.
"""

from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import sympy as sp
from jaxtyping import Array, Float
from sympy import lambdify


# ---------------------------------------------------------------------------
# SymbolicMetric
# ---------------------------------------------------------------------------


class SymbolicMetric:
    """Symbolic metric specification using SymPy.

    Holds a coordinate symbol list and a 4x4 SymPy Matrix representing
    the metric tensor *g_{ab}*.  The inverse is computed lazily and cached.

    Parameters
    ----------
    coords : list[sp.Symbol]
        Four coordinate symbols, e.g. ``[t, x, y, z]``.
    g_matrix : sp.Matrix
        Symmetric (4, 4) metric tensor expressed in *coords*.

    Raises
    ------
    ValueError
        If *coords* does not have length 4 or *g_matrix* is not (4, 4).
    """

    def __init__(self, coords: list[sp.Symbol], g_matrix: sp.Matrix) -> None:
        if len(coords) != 4:
            raise ValueError(
                f"Expected 4 coordinate symbols, got {len(coords)}"
            )
        if g_matrix.shape != (4, 4):
            raise ValueError(
                f"Expected (4, 4) metric matrix, got {g_matrix.shape}"
            )
        self.coords = list(coords)
        self.g = g_matrix

    @cached_property
    def g_inv(self) -> sp.Matrix:
        """Inverse metric tensor *g^{ab}* (computed once, then cached)."""
        return self.g.inv()


# ---------------------------------------------------------------------------
# MetricSpecification (abstract base Equinox module / JAX pytree)
# ---------------------------------------------------------------------------


class MetricSpecification(eqx.Module):
    """Abstract base for spacetime metrics.

    Subclasses define a pointwise mapping from a single spacetime coordinate
    ``(t, x, y, z)`` to the 4x4 metric tensor *g_{ab}* at that point.

    Being an ``eqx.Module``, every ``MetricSpecification`` is automatically a
    JAX pytree, compatible with ``jax.jit``, ``jax.vmap``, and other
    transformations.  Numeric parameters stored as regular (non-static)
    fields are treated as *dynamic* leaves, so the compiled code is reusable
    when only those values change.
    """

    @abstractmethod
    def __call__(self, coords: Float[Array, "4"]) -> Float[Array, "4 4"]:
        """Evaluate *g_{ab}* at a single spacetime point.

        Parameters
        ----------
        coords : Float[Array, "4"]
            Spacetime coordinates ``(t, x, y, z)``.

        Returns
        -------
        Float[Array, "4 4"]
            The metric tensor at the given point.
        """
        ...

    @abstractmethod
    def symbolic(self) -> SymbolicMetric:
        """Return the SymPy symbolic form for inspection and cross-validation."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable metric name."""
        ...


# ---------------------------------------------------------------------------
# ADMMetric (abstract 3+1 decomposition interface)
# ---------------------------------------------------------------------------


class ADMMetric(MetricSpecification):
    """Abstract base for metrics defined via the ADM 3+1 decomposition.

    Subclasses provide pointwise lapse, shift, and spatial metric.  The full
    4x4 spacetime metric is reconstructed automatically by ``__call__`` via
    :func:`adm_to_full_metric`.
    """

    @abstractmethod
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Lapse function *alpha* at a single spacetime point.

        Parameters
        ----------
        coords : Float[Array, "4"]
            Spacetime coordinates ``(t, x, y, z)``.
        """
        ...

    @abstractmethod
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        """Shift vector *beta^i* at a single spacetime point.

        Parameters
        ----------
        coords : Float[Array, "4"]
            Spacetime coordinates ``(t, x, y, z)``.
        """
        ...

    @abstractmethod
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        """Spatial metric *gamma_{ij}* at a single spacetime point.

        Parameters
        ----------
        coords : Float[Array, "4"]
            Spacetime coordinates ``(t, x, y, z)``.
        """
        ...

    def __call__(self, coords: Float[Array, "4"]) -> Float[Array, "4 4"]:
        """Reconstruct full *g_{ab}* from ADM components.

        Uses :func:`adm_to_full_metric` to combine lapse, shift, and
        spatial metric into the full 4x4 spacetime metric tensor.
        """
        return adm_to_full_metric(
            self.lapse(coords),
            self.shift(coords),
            self.spatial_metric(coords),
        )


# ---------------------------------------------------------------------------
# ADM reconstruction helper
# ---------------------------------------------------------------------------


def adm_to_full_metric(
    alpha: Float[Array, ""],
    beta_up: Float[Array, "3"],
    gamma: Float[Array, "3 3"],
) -> Float[Array, "4 4"]:
    """Reconstruct the full 4x4 spacetime metric from ADM 3+1 components.

    Given lapse *alpha*, contravariant shift *beta^i*, and spatial metric
    *gamma_{ij}*, the full metric is:

    .. math::

        g_{00} &= -(\\alpha^2 - \\beta_i \\beta^i) \\\\
        g_{0i} &= \\beta_i \\\\
        g_{ij} &= \\gamma_{ij}

    where :math:`\\beta_i = \\gamma_{ij} \\beta^j`.

    All operations use JAX functional array updates (``.at[].set()``).

    Parameters
    ----------
    alpha : Float[Array, ""]
        Lapse function (scalar).
    beta_up : Float[Array, "3"]
        Contravariant shift vector :math:`\\beta^i`.
    gamma : Float[Array, "3 3"]
        Spatial metric :math:`\\gamma_{ij}`.

    Returns
    -------
    Float[Array, "4 4"]
        Full spacetime metric tensor :math:`g_{ab}`.
    """
    beta_down = jnp.einsum("ij,j->i", gamma, beta_up)
    beta_sq = jnp.dot(beta_down, beta_up)

    g = jnp.zeros((4, 4))
    g = g.at[0, 0].set(-(alpha**2 - beta_sq))
    g = g.at[0, 1:].set(beta_down)
    g = g.at[1:, 0].set(beta_down)
    g = g.at[1:, 1:].set(gamma)
    return g


# ---------------------------------------------------------------------------
# SymPy-to-JAX bridge
# ---------------------------------------------------------------------------


def sympy_metric_to_jax(
    symbolic_metric: SymbolicMetric,
) -> Callable[[Float[Array, "4"]], Float[Array, "4 4"]]:
    """Convert a SymPy metric to a JAX function matching the pointwise signature.

    Uses ``sympy.lambdify`` with ``modules='jax'`` to produce a callable
    that maps ``coords (4,) -> g_ab (4, 4)`` using ``jax.numpy`` operations.

    Parameters
    ----------
    symbolic_metric : SymbolicMetric
        Symbolic metric with ``.coords`` and ``.g`` attributes.

    Returns
    -------
    Callable[[Float[Array, "4"]], Float[Array, "4 4"]]
        A JAX-compatible function evaluating the metric tensor.
    """
    f_raw = lambdify(symbolic_metric.coords, symbolic_metric.g, modules="jax")

    def f_wrapped(coords: Float[Array, "4"]) -> Float[Array, "4 4"]:
        return jnp.asarray(f_raw(*coords), dtype=jnp.float64)

    return f_wrapped


def sympy_metric_inverse_to_jax(
    symbolic_metric: SymbolicMetric,
) -> Callable[[Float[Array, "4"]], Float[Array, "4 4"]]:
    """Convert a SymPy inverse metric to a JAX function.

    Uses ``sympy.lambdify`` with ``modules='jax'`` to produce a callable
    that maps ``coords (4,) -> g^{ab} (4, 4)`` using ``jax.numpy`` operations.

    Parameters
    ----------
    symbolic_metric : SymbolicMetric
        Symbolic metric with ``.coords`` and ``.g_inv`` attributes.

    Returns
    -------
    Callable[[Float[Array, "4"]], Float[Array, "4 4"]]
        A JAX-compatible function evaluating the inverse metric tensor.
    """
    f_raw = lambdify(
        symbolic_metric.coords, symbolic_metric.g_inv, modules="jax"
    )

    def f_wrapped(coords: Float[Array, "4"]) -> Float[Array, "4 4"]:
        return jnp.asarray(f_raw(*coords), dtype=jnp.float64)

    return f_wrapped
