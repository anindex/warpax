r"""Interval-valued forward-mode automatic differentiation to second order.

The certified enclosures of :mod:`warpax.energy_conditions.enclosure` need the
metric *and its first two derivatives* as rigorous enclosures over a box. Two
routes were possible: symbolic differentiation followed by interval evaluation,
or automatic differentiation carried out directly in interval arithmetic. The
symbolic route is fine for a one-component shift (Alcubierre) but does not scale
-- the three-component Rodal shift, with its ``1/r`` factors and ``log cosh``
profile, blows the expression up beyond what ``sympy`` will reduce in any
practical time.

This module takes the second route. Automatic differentiation is *exact*: it
applies the chain rule to the program, introducing no truncation error. Carrying
it out over interval arithmetic therefore yields rigorous enclosures of the
derivatives, with the same guarantee mpmath's interval type provides for the
values. The cost is that every quantity carries ``1 + 4 + 16`` interval numbers
(value, gradient, Hessian) instead of one.

Only the operations the warp metrics actually use are implemented. Every
transcendental is written so that its argument appears exactly once, which keeps
the interval extension free of dependency overestimation at the leaves.
"""

from __future__ import annotations

import mpmath
from mpmath import iv

from . import _jet

__all__ = ["Dual2", "constant", "cosh", "exp", "log", "sinh", "sqrt", "tanh", "variable"]

# The transcendentals below are the only places that name a concrete scalar type.
# Dispatching them lets Dual2 be instantiated over ``_jet.Jet`` as well as over a
# bare mpmath interval, which is what supplies the extra derivative the centered
# bound in :mod:`.enclosure` needs. Everything else here is generic in + - * /.
_sqrt, _exp, _log = _jet.sqrt, _jet.exp, _jet.log
_add, _mul = _jet.add, _jet.mul

_N = 4  # spacetime dimension


def _zeros():
    return [iv.mpf([0, 0]) for _ in range(_N)]


def _zeros2():
    return [[iv.mpf([0, 0]) for _ in range(_N)] for _ in range(_N)]


# Shared zero gradient and Hessian; see _jet._ZERO_D.
_ZERO_G = tuple(iv.mpf([0, 0]) for _ in range(_N))
_ZERO_H = tuple(tuple(iv.mpf([0, 0]) for _ in range(_N)) for _ in range(_N))


class Dual2:
    """Value, gradient and Hessian, each an mpmath interval.

    Arithmetic follows the exact chain rule, so ``d.g[i]`` encloses
    ``df/dx_i`` and ``d.h[i][j]`` encloses ``d^2 f/dx_i dx_j`` over the box on
    which the inputs were seeded.
    """

    __slots__ = ("g", "h", "v")

    def __init__(self, v, g=None, h=None):
        self.v = v
        self.g = _ZERO_G if g is None else g
        self.h = _ZERO_H if h is None else h

    #, arithmetic ---------------------------------------------------------

    def __add__(self, o):
        o = _lift(o)
        return Dual2(
            _add(self.v, o.v),
            [_add(self.g[i], o.g[i]) for i in range(_N)],
            [[_add(self.h[i][j], o.h[i][j]) for j in range(_N)] for i in range(_N)],
        )

    __radd__ = __add__

    def __neg__(self):
        return Dual2(
            -self.v,
            [-self.g[i] for i in range(_N)],
            [[-self.h[i][j] for j in range(_N)] for i in range(_N)],
        )

    def __sub__(self, o):
        return self + (-_lift(o))

    def __rsub__(self, o):
        return _lift(o) + (-self)

    def __mul__(self, o):
        o = _lift(o)
        v = _mul(self.v, o.v)
        g = [_add(_mul(self.g[i], o.v), _mul(self.v, o.g[i])) for i in range(_N)]
        h = [
            [
                # Left-associated, exactly as the un-nested version was: interval
                # addition is commutative but NOT associative under directed
                # rounding, so a different bracketing would move the value
                # enclosure by a few ulps and the jet ring would stop reproducing
                # the plain one bit for bit.
                _add(
                    _add(
                        _add(_mul(self.h[i][j], o.v), _mul(self.g[i], o.g[j])),
                        _mul(self.g[j], o.g[i]),
                    ),
                    _mul(self.v, o.h[i][j]),
                )
                for j in range(_N)
            ]
            for i in range(_N)
        ]
        return Dual2(v, g, h)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self * _reciprocal(_lift(o))

    def __rtruediv__(self, o):
        return _lift(o) * _reciprocal(self)

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


def _lift(o):
    return o if isinstance(o, Dual2) else constant(o)


def constant(c):
    c = c if hasattr(c, "a") else iv.mpf([c, c])
    return Dual2(c)


def variable(box_interval, index: int):
    """Seed the ``index``-th coordinate with its enclosure over the box."""
    g = _zeros()
    g[index] = iv.mpf([1, 1])
    return Dual2(box_interval, g)


def variable_jet(box_interval, index: int):
    """Seed coordinate ``index`` over the jet ring, carrying one extra derivative.

    The value is the coordinate itself, so its jet gradient is ``e_index``; the
    Dual2 gradient is the constant one, whose jet gradient is zero.
    """
    g = _zeros()
    g[index] = _jet.constant(1)
    return Dual2(_jet.seed(box_interval, index), g)


def _chain(d: Dual2, f0, f1, f2) -> Dual2:
    """Propagate a scalar function through value, gradient and Hessian.

    ``f0, f1, f2`` are the function and its first two derivatives evaluated on
    the interval enclosing ``d.v``.
    """
    g = [_mul(f1, d.g[i]) for i in range(_N)]
    h = [
        [_add(_mul(_mul(f2, d.g[i]), d.g[j]), _mul(f1, d.h[i][j]))
         for j in range(_N)]
        for i in range(_N)
    ]
    return Dual2(f0, g, h)


def _reciprocal(d: Dual2) -> Dual2:
    inv = 1 / d.v
    return _chain(d, inv, -(inv**2), 2 * inv**3)


def sqrt(d: Dual2) -> Dual2:
    """Interval square root, with the radicand clamped at zero from below.

    PRECONDITION: the caller must know the radicand is non-negative on the true box.
    Every use here is a norm or a sum of squares, where a negative lower endpoint is
    outward rounding and nothing else, and clamping it recovers tightness without
    losing enclosure. The clamp is NOT a domain repair: handed a box that genuinely
    straddles zero, this returns the square root of its non-negative part and says
    nothing about the rest. Callers that cannot establish the precondition must bracket
    the sign themselves before calling.
    """
    if _jet.is_jet(d.v):
        s = _sqrt(d.v)
    else:
        lo = d.v.a if d.v.a > 0 else mpmath.mpf(0)
        s = iv.sqrt(iv.mpf([lo, d.v.b]))
    return _chain(d, s, 1 / (2 * s), -1 / (4 * s**3))


def exp(d: Dual2) -> Dual2:
    e = _exp(d.v)
    return _chain(d, e, e, e)


def log(d: Dual2) -> Dual2:
    return _chain(d, _log(d.v), 1 / d.v, -1 / d.v**2)


def tanh(d: Dual2) -> Dual2:
    # Single occurrence of the argument keeps the leaf enclosure tight.
    th = 1 - 2 / (_exp(2 * d.v) + 1)
    sech2 = 1 - th**2
    return _chain(d, th, sech2, -2 * th * sech2)


def cosh(d: Dual2) -> Dual2:
    e, ei = _exp(d.v), _exp(-d.v)
    c, s = (e + ei) / 2, (e - ei) / 2
    return _chain(d, c, s, c)


def sinh(d: Dual2) -> Dual2:
    e, ei = _exp(d.v), _exp(-d.v)
    c, s = (e + ei) / 2, (e - ei) / 2
    return _chain(d, s, c, s)
