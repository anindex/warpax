"""Affine window for a null line integral, chosen by convergence, not by hand.

The published span was hard-coded at 2|x_start|, which ends at the bubble centre
for a receding bubble and integrates only half the crossing. Here the span is
doubled at fixed step density until the integral is stationary, and the achieved
span is reported with the value. Measured once per metric on the smallest impact
parameter, which sees the most of the bubble.
"""
from __future__ import annotations

__all__ = ["converged_window", "crossing_span"]


def converged_window(
    integrate,
    span0: float,
    *,
    # 1e-4: below the fourth significant figure these are reported to. Rodal's
    # integrand decays like 1/lambda^2, so 1e-6 would need span ~ 4000.
    rtol: float = 1e-4,
    atol: float = 1e-12,
    max_doublings: int = 8,
) -> tuple[float, float, bool]:
    """Double the affine span until the line integral is stationary.

    Parameters
    ----------
    integrate : callable
        ``integrate(span) -> float``. Must scale its step count with ``span``,
        or this measures quadrature error instead of truncation error.
    span0 : float
        Starting affine span.
    rtol, atol : float
        Stationarity test: ``|I(2s) - I(s)| <= atol + rtol |I(2s)|``.
    max_doublings : int
        Give up after this many and report ``converged=False``.

    Returns
    -------
    (value, span, converged)
    """
    span = float(span0)
    prev = float(integrate(span))
    for _ in range(max_doublings):
        span *= 2.0
        cur = float(integrate(span))
        if abs(cur - prev) <= atol + rtol * abs(cur):
            return cur, span, True
        prev = cur
    return prev, span, False


def _demo() -> None:
    """A decaying integrand converges; a non-decaying one is reported as such."""
    import math

    # exp(-x) on [0, s]: converges to 1
    v, s, ok = converged_window(lambda s: 1.0 - math.exp(-s), 1.0)
    assert ok and abs(v - 1.0) < 1e-6, (v, s, ok)

    # a constant integrand never converges, and must not claim to
    v, s, ok = converged_window(lambda s: s, 1.0)
    assert not ok, (v, s, ok)

    # the truncation signature this module exists to catch: half of a symmetric
    # crossing looks stable only if you never extend the window
    def half(span):
        return math.tanh(span - 16.0) * 0.5 + 0.5

    v, s, ok = converged_window(half, 16.0)
    assert ok and abs(v - 1.0) < 1e-5, (v, s, ok)
    print("_anec_window demo OK")


def crossing_span(lam, r_s, support_r: float, margin: float = 2.0) -> tuple[float, bool]:
    """Affine span covering the crossing, from the ray's own trajectory.

    Stationarity under doubling is the wrong test for a geodesic: past the
    crossing, doubling the window adds no physics and does add integrator drift,
    so the symplectic integral is stationary from span 32 to 64 and then degrades
    (-0.0772 -> -0.0834 by span 256). This test is monotone instead: the window
    is long enough once the ray is outside ``r_s = support_r`` and staying
    outside, times ``margin``.

    ``support_r`` is a truncation margin, not a support radius. ``tail_bound``
    certifies the *shape function* below 1.3e-14 outside it; the integrand
    ``T_ab k^a k^b`` is driven by that function and its derivatives and is
    suppressed by the same order, but no bound on it out there is computed
    anywhere, and a tanh wall has exponential tails. The doubling on top is what
    makes it a margin.

    Returns ``(span, left)``: the span, and whether the ray was ever seen to
    leave. A ``False`` is reported, not raised.
    """
    lam = [float(v) for v in lam]
    r_s = [float(v) for v in r_s]
    finite = [i for i, (a, b) in enumerate(zip(lam, r_s, strict=True))
              if a == a and b == b]  # drop NaN (some rays need projection)
    if not finite:
        return float(lam[-1]), False
    inside = [i for i in finite if r_s[i] < support_r]
    if not inside:
        return float(lam[finite[0]]) * margin, True
    last = max(inside)
    if last == finite[-1]:
        return float(lam[last]), False  # never left the support region
    return float(lam[last]) * margin, True


if __name__ == "__main__":
    _demo()
