"""- ShapeFunction: differentiable shape-function basis library.

A shape function ``f : R -> R`` (typically ``f(r)`` for a radial coordinate)
is the scalar profile that modulates the warp-bubble metric. v0.1.x warpax
exposes five hand-picked shape families (Alcubierre tanh, Rodal Gaussian
wall, Natario dual-lobe, Van den Broeck compound, Lentz walls); this module
introduces a *differentiable parameter family* so that the optimizer
can search the shape space via ``jax.grad`` over the full
curvature chain.

This module provides three basis families, each exposed as a
``classmethod`` constructor on :class:`ShapeFunction`:

- :meth:`ShapeFunction.spline` - cubic B-spline via ``interpax.interp1d``.
  Default 24 knots; matches the Alcubierre reproduction target. Requires the ``[design]`` extra
  (``pip install "warpax[design]"``).
- :meth:`ShapeFunction.bernstein` - pure-JAX Bernstein polynomial basis
  on ``r / r_max`` (rescaled to ``[0, 1]``); no extra dependencies.
- :meth:`ShapeFunction.gmm` - Gaussian-mixture-model
  ``sum_k amps_k * exp(-((r - means_k) / widths_k)^2)``.

Every evaluation path is JAX-traceable (no Python branching on traced
values), so ``jax.grad`` / ``jax.jit`` / ``jax.vmap`` all work
out-of-the-box. Per-basis differentiability contract:
``|jax.jacfwd(sf)(r) - finite_difference(sf)(r)|_max < 1e-7`` on
100 random probe points.

References
----------
- §6 (Constrained BFGS Strategy) - authoritative spec;
  rationale + Bernstein/GMM fallback motivation.
"""
from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class ShapeFunction(eqx.Module):
    """Differentiable shape function for warp-drive metric design.

    A single :class:`ShapeFunction` instance carries:

    - ``basis`` (static): one of ``"spline"``, ``"bernstein"``, ``"gmm"``.
    - ``params`` (dynamic pytree leaves): basis-specific array dict.
    - ``order`` (static): spline polynomial order (``3`` = cubic); ignored
      for non-spline bases.

    The instance is callable on a scalar or array of radial coordinates
    ``r``; evaluation dispatches on ``basis`` (static field, so no JIT
    retrace penalty per basis family).

    All three basis families are ``jax.grad``-compatible. Arithmetic
    flows through standard JAX primitives (interpax for spline,
    polynomial sums for Bernstein, sums of Gaussians for GMM).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from warpax.design import ShapeFunction
    >>> sf = ShapeFunction.spline(
    ... knots=jnp.linspace(0.0, 1.0, 24),
    ... values=jnp.sin(jnp.linspace(0.0, 1.0, 24) * 3.0),
    ... )
    >>> float(sf(jnp.asarray(0.5))) # doctest: +SKIP
    -0.143...
    """

    basis: str = eqx.field(static=True)
    params: dict[str, Any]
    order: int = eqx.field(static=True, default=3)

    # ------------------------------------------------------------------
    # Classmethod constructors (basis-specific)
    # ------------------------------------------------------------------

    @classmethod
    def spline(
        cls,
        knots: Float[Array, "K"],
        values: Float[Array, "K"],
        order: int = 3,
    ) -> "ShapeFunction":
        """Cubic B-spline via ``interpax.interp1d``.

        Parameters
        ----------
        knots
            Monotone increasing sample points in the radial domain.
            Default 24 knots (matches Alcubierre
            target of ``< 1e-4`` relative error).
        values
            Shape-function amplitudes at ``knots``. Same length as
            ``knots``.
        order
            Spline polynomial order. ``3`` = cubic (the only value
            currently supported by the underlying ``interpax.interp1d``
            method name ``'cubic'``). Additional methods (``'akima'``,
            ``'pchip'``) would be a future FUT extension.

        Returns
        -------
        ShapeFunction
            With ``basis='spline'`` and ``params={'knots': knots,
            'values': values}``.

        Notes
        -----
        Differentiability: ``interpax.interp1d`` is a pure-JAX routine
        that computes spline coefficients on the fly, so ``jax.grad`` /
        ``jax.jacfwd`` flow through both ``r`` and ``values`` (the
        knot locations are static).
        """
        knots = jnp.asarray(knots)
        values = jnp.asarray(values)
        if knots.shape != values.shape:
            raise ValueError(
                f"spline: knots.shape={knots.shape} must equal "
                f"values.shape={values.shape}"
            )
        return cls(
            basis="spline",
            params={"knots": knots, "values": values},
            order=order,
        )

    @classmethod
    def bernstein(
        cls,
        coeffs: Float[Array, "N"],
        r_max: float = 1.0,
    ) -> "ShapeFunction":
        """Pure-JAX Bernstein polynomial basis.

        ``f(r) = sum_{k=0}^{n} coeffs_k * B_{n,k}(t)``
        where ``t = r / r_max`` (clipped to ``[0, 1]``) and
        ``B_{n,k}(t) = C(n, k) * t^k * (1 - t)^(n - k)``.

        Parameters
        ----------
        coeffs
            Bernstein control coefficients. The polynomial degree is
            ``n = len(coeffs) - 1``.
        r_max
            Upper bound of the radial domain; Bernstein basis evaluates
            in the normalized coordinate ``t = r / r_max``.

        Returns
        -------
        ShapeFunction
            With ``basis='bernstein'``.

        Notes
        -----
        Differentiability: Bernstein evaluation is pure polynomial
        arithmetic, so ``jax.grad`` flows through both ``r`` and
        ``coeffs``. No dependency on ``interpax`` - the Bernstein
        basis is the stdlib-only fallback when the ``[design]`` extra
        is not installed.
        """
        coeffs = jnp.asarray(coeffs)
        if coeffs.ndim != 1 or coeffs.shape[0] < 2:
            raise ValueError(
                f"bernstein: coeffs must be 1-D with at least 2 entries, "
                f"got shape={coeffs.shape}"
            )
        return cls(
            basis="bernstein",
            params={"coeffs": coeffs, "r_max": jnp.asarray(r_max)},
        )

    @classmethod
    def gmm(
        cls,
        means: Float[Array, "M"],
        widths: Float[Array, "M"],
        amps: Float[Array, "M"],
    ) -> "ShapeFunction":
        """Gaussian-mixture-model basis.

        ``f(r) = sum_k amps_k * exp(-((r - means_k) / widths_k)^2)``

        Parameters
        ----------
        means
            Gaussian component centers.
        widths
            Gaussian component scale parameters (must be > 0; not
            validated at construction - optimizer enforces via
            sigmoid reparameterization).
        amps
            Gaussian component amplitudes.

        Returns
        -------
        ShapeFunction
            With ``basis='gmm'``.

        Notes
        -----
        Differentiability: analytic Gaussian arithmetic; all three
        parameter arrays flow through ``jax.grad``.
        """
        means = jnp.asarray(means)
        widths = jnp.asarray(widths)
        amps = jnp.asarray(amps)
        if means.shape != widths.shape or means.shape != amps.shape:
            raise ValueError(
                f"gmm: means/widths/amps must have equal shape, got "
                f"means={means.shape} widths={widths.shape} amps={amps.shape}"
            )
        return cls(
            basis="gmm",
            params={"means": means, "widths": widths, "amps": amps},
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, r: Float[Array, ""]) -> Float[Array, ""]:
        """Evaluate the shape function at radial coordinate ``r``.

        Parameters
        ----------
        r
            Scalar (or 0-D array) radial coordinate. For vectorized
            evaluation use ``jax.vmap(sf)(r_array)``.

        Returns
        -------
        Float[Array, ""]
            Scalar amplitude.

        Notes
        -----
        Dispatch on ``self.basis`` uses a Python ``if`` chain - safe
        since ``basis`` is a static field (no JIT retrace penalty).
        """
        r = jnp.asarray(r)
        if self.basis == "spline":
            return _eval_spline(r, self.params)
        elif self.basis == "bernstein":
            return _eval_bernstein(r, self.params)
        elif self.basis == "gmm":
            return _eval_gmm(r, self.params)
        else:
            raise ValueError(
                f"ShapeFunction: unknown basis {self.basis!r}; "
                f"expected one of {'spline', 'bernstein', 'gmm'}."
            )


# ---------------------------------------------------------------------------
# Internal evaluation helpers
# ---------------------------------------------------------------------------


def _eval_spline(r, params):
    """Cubic B-spline evaluation via ``interpax.interp1d``.

    Lazily imports interpax so the ``[design]`` extra is optional for
    users of Bernstein / GMM bases.
    """
    try:
        import interpax
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ShapeFunction.spline requires the '[design]' extra: "
            "`pip install \"warpax[design]\"`."
        ) from exc
    knots = params["knots"]
    values = params["values"]
    return interpax.interp1d(r, knots, values, method="cubic")


def _eval_bernstein(r, params):
    """Bernstein polynomial evaluation via closed-form recursion.

    ``f(t) = sum_k c_k * B_{n,k}(t)`` with ``t = r / r_max`` clipped
    to ``[0, 1]``.
    """
    coeffs = params["coeffs"]
    r_max = params["r_max"]
    n = coeffs.shape[0] - 1
    t = jnp.clip(r / r_max, 0.0, 1.0)

    # Binomial coefficients C(n, k) for k = 0 .. n
    k_idx = jnp.arange(n + 1)
    # Use log-gamma to avoid overflow for high-degree polynomials;
    # then exponentiate. jax.scipy.special.gammaln is JIT-safe.
    from jax.scipy.special import gammaln

    log_binom = gammaln(n + 1.0) - gammaln(k_idx + 1.0) - gammaln(n - k_idx + 1.0)
    binom = jnp.exp(log_binom)
    # Safe zero-handling: 0^0 = 1; t^k * (1-t)^(n-k) vanishes at endpoints
    # except for the edge basis functions. jnp.power handles 0**0 as 1.
    basis_vals = binom * (t ** k_idx) * ((1.0 - t) ** (n - k_idx))
    return jnp.sum(coeffs * basis_vals)


def _eval_gmm(r, params):
    """Gaussian-mixture-model evaluation."""
    means = params["means"]
    widths = params["widths"]
    amps = params["amps"]
    arg = (r - means) / widths
    return jnp.sum(amps * jnp.exp(-(arg ** 2)))
