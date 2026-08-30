r"""First-order jets over mpmath intervals, for the centered (mean-value) bound.

Why this exists
---------------
:mod:`._intervalcurv` evaluates ``g -> Gamma -> Riemann -> Ricci -> G_ab -> T_ab``
in interval arithmetic, which is rigorous but loose: the same derivative of the
shift enters many terms that largely cancel, and interval arithmetic cannot see the
cancellation. Measured on the Alcubierre wall, the enclosure of ``rho_n`` over a box
is about 148 times wider than the true range of ``rho_n`` on that box, and the ratio
is constant in the box width, pure dependency inflation, not a resolution effect.

The excess of the certified lower bound is ``C*h`` with ``h`` the box half-width and
``C`` of order 100, so a branch-and-bound at tolerance ``tol`` must retain roughly
``C^2/tol`` boxes. That is why the 120000-box budget could not close.

The standard cure is the centered, or mean-value, form. For ``f`` continuously
differentiable on a box ``X`` with centre ``c``,

    f(X)  subset of  f(c) + sum_m [df/dx_m](X) * (X_m - c_m),

whose width is ``O(h)`` with the *true* gradient magnitude as the constant, and whose
excess over the true range is ``O(h^2)`` rather than ``O(h)``. The point value
``f(c)`` is computed on a degenerate box, where there is no dependency at all.

Getting the derivative enclosures
---------------------------------
The centered form needs rigorous enclosures of ``d rho/dx`` over the box, i.e. one
more derivative of the metric than :class:`~._intervalad.Dual2` carries. Rather than
write a ``Dual3``, we nest: ``Dual2`` is generic in ``+ - * /``, so instantiating it
over *this* ring, interval value plus interval gradient, makes every quantity it
produces carry one extra derivative for free. Nesting forward-mode AD is exact
(mixed partials commute), so ``gd[i][j].h[k][l].d[m]`` encloses
``d^3 g_ij / dx_k dx_l dx_m``, and running the curvature chain over ``Jet`` returns
``rho_n`` with ``.d[m]`` enclosing ``d rho_n / dx_m``.

Every entry is an ``mpmath.iv`` interval, so outward rounding is inherited and the
result is as rigorous as the un-nested chain.
"""

from __future__ import annotations

import mpmath
from mpmath import iv

__all__ = ["Jet", "constant", "exp", "grad", "is_jet", "log", "seed", "sqrt", "value"]

_N = 4  # spacetime dimension


def _zeros():
    return [iv.mpf([0, 0]) for _ in range(_N)]


# Shared zero gradient. iv.mpf is immutable and no jet mutates d in place, so
# allocating four fresh intervals per Jet was 20% of the branch-and-bound.
# A tuple, so an attempted in-place write raises instead of corrupting.
_ZERO_D = tuple(iv.mpf([0, 0]) for _ in range(_N))


class Jet:
    """Interval value and interval gradient of a function of the box coordinates.

    ``v`` encloses ``f`` over the box; ``d[m]`` encloses ``df/dx_m`` over the box.
    """

    __slots__ = ("d", "v")

    def __init__(self, v, d=None):
        self.v = v
        self.d = _ZERO_D if d is None else d

    # Endpoint access, so pivot selection and positivity tests in _intervalcurv
    # work on a Jet exactly as they do on a bare interval.
    @property
    def a(self):
        return self.v.a

    @property
    def b(self):
        return self.v.b

    def __add__(self, o):
        o = _lift(o)
        return Jet(self.v + o.v, [self.d[m] + o.d[m] for m in range(_N)])

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, [-self.d[m] for m in range(_N)])

    def __sub__(self, o):
        return self + (-_lift(o))

    def __rsub__(self, o):
        return _lift(o) + (-self)

    def __mul__(self, o):
        o = _lift(o)
        return Jet(
            self.v * o.v,
            [self.d[m] * o.v + self.v * o.d[m] for m in range(_N)],
        )

    __rmul__ = __mul__

    # Division is written out rather than routed through ``self * (1/o)``. Interval
    # division rounds once; multiplying by a reciprocal rounds twice, so the shortcut
    # would widen every quotient in the curvature chain by a few ulps and the jet ring
    # would stop reproducing the plain one bit for bit.
    def __truediv__(self, o):
        o = _lift(o)
        v = self.v / o.v
        o2 = o.v * o.v
        return Jet(v, [(self.d[m] * o.v - self.v * o.d[m]) / o2 for m in range(_N)])

    def __rtruediv__(self, o):
        return _lift(o).__truediv__(self)

    def __pow__(self, n: int):
        if not isinstance(n, int):
            raise TypeError("only integer powers are supported")
        if n == 0:
            return constant(1)
        if n < 0:
            return _reciprocal(self ** (-n))
        out = self
        for _ in range(n - 1):
            out = out * self
        return out

    def __abs__(self):
        # |f| is not differentiable through zero. Every use in the curvature chain
        # is on a quantity bounded away from zero (the squared norm of the Eulerian
        # normal); refusing the ambiguous case keeps the result rigorous.
        lo, hi = float(mpmath.mpf(self.v.a)), float(mpmath.mpf(self.v.b))
        if lo > 0:
            return self
        if hi < 0:
            return -self
        raise ValueError("abs() of a jet whose value interval contains zero")


def is_jet(x) -> bool:
    return isinstance(x, Jet)


# mpmath's interval type raises NotImplementedError from its own ``convert`` rather
# than returning NotImplemented, so Python never falls back to ``Jet.__radd__`` /
# ``Jet.__rmul__`` when a bare interval sits on the left. Addition and multiplication
# commute, so these helpers simply put the jet first; they are the only reason
# ``Dual2`` can be instantiated over either ring without duplicating its arithmetic.

def add(x, y):
    if isinstance(y, Jet) and not isinstance(x, Jet):
        return y + x
    return x + y


def mul(x, y):
    if isinstance(y, Jet) and not isinstance(x, Jet):
        return y * x
    return x * y


def sub(x, y):
    if isinstance(y, Jet) and not isinstance(x, Jet):
        return (-y) + x
    return x - y


def div(x, y):
    if isinstance(y, Jet) and not isinstance(x, Jet):
        return constant(x).__truediv__(y)
    return x / y


def _lift(o):
    return o if isinstance(o, Jet) else constant(o)


def constant(c) -> Jet:
    """A constant: value ``c``, zero gradient."""
    c = c if hasattr(c, "a") else iv.mpf([c, c])
    return Jet(c)


def seed(box_interval, index: int) -> Jet:
    """The ``index``-th coordinate itself: value the box, gradient ``e_index``."""
    d = _zeros()
    d[index] = iv.mpf([1, 1])
    return Jet(box_interval, d)


def _chain(j: Jet, f0, f1) -> Jet:
    """Propagate a scalar function: ``(f o j)' = f'(j) * j'``."""
    return Jet(f0, [f1 * j.d[m] for m in range(_N)])


def _reciprocal(j: Jet) -> Jet:
    inv = 1 / j.v
    return _chain(j, inv, -(inv**2))


def sqrt(j):
    """``sqrt`` on a jet or a bare interval."""
    if not isinstance(j, Jet):
        return iv.sqrt(j)
    lo = j.v.a if j.v.a > 0 else mpmath.mpf(0)
    s = iv.sqrt(iv.mpf([lo, j.v.b]))
    return _chain(j, s, 1 / (2 * s))


def exp(j):
    if not isinstance(j, Jet):
        return iv.exp(j)
    e = iv.exp(j.v)
    return _chain(j, e, e)


def log(j):
    if not isinstance(j, Jet):
        return iv.log(j)
    return _chain(j, iv.log(j.v), 1 / j.v)


def value(x):
    """The interval value of a jet, or the interval itself."""
    return x.v if isinstance(x, Jet) else x


def grad(x):
    """The interval gradient of a jet, or a zero gradient for a bare interval."""
    return x.d if isinstance(x, Jet) else _ZERO_D
