"""Shared Diffrax result-code conversion and termination-reason mapping.

Diffrax >= 0.6 reports ``sol.result`` as an equinox ``EnumerationItem``
(carrying ``._value``, an int32 array) rather than a plain int, and the
member indices differ from older releases. This module centralizes the
robust conversion used by :mod:`warpax.averaged.anec` and
:mod:`warpax.averaged.awec` so a non-convertible result is *never*
silently mapped to success.
"""
from __future__ import annotations


# Sentinel for a result object we could not convert; maps to reason
# 'unknown' and must be treated as non-success by callers.
RESULT_UNKNOWN = -1

# Diffrax success code (``diffrax.RESULTS.successful``).
RESULT_SUCCESS = 0

# Human-readable termination reasons, verified against the installed
# diffrax 0.7.2 ``diffrax.RESULTS`` enumeration (equinox Enumeration;
# index -> member name via ``RESULTS._name_to_item``).
TERMINATION_REASONS: dict[int, str] = {
    RESULT_UNKNOWN: "unknown",
    0: "complete",              # successful
    1: "max_steps",             # max_steps_reached
    2: "singular",              # singular (linear solve)
    3: "breakdown",             # iterative linear-solve breakdown
    4: "stagnation",            # iterative linear-solve stagnation
    5: "conlim",                # condition-number limit exceeded
    6: "nonfinite_input",       # non-finite linear-solve input
    7: "nonlinear_max_steps",   # nonlinear_max_steps_reached
    8: "nonlinear_divergence",  # nonlinear solve diverged
    9: "nonfinite",             # non-finite values during solve
    10: "dt_min_reached",       # minimum step size reached
    11: "event_occurred",       # terminating event triggered
    12: "max_steps_rejected",   # max rejected steps reached
    13: "internal_error",       # diffrax internal error
}


def result_code_to_int(raw: object) -> int:
    """Convert a Diffrax result (EnumerationItem, array, or int) to int.

    Tries ``._value`` (equinox EnumerationItem on diffrax 0.7.x), then
    ``.value`` (older conventions), then ``int(raw)``. Returns
    :data:`RESULT_UNKNOWN` -- never the success code -- when no
    conversion applies, so an unrecognized outcome is reported as an
    incomplete geodesic rather than masked as success.
    """
    for attr in ("_value", "value"):
        v = getattr(raw, attr, None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    try:
        return int(raw)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return RESULT_UNKNOWN


def termination_reason(code: int) -> str:
    """Human-readable reason for an integer result code."""
    return TERMINATION_REASONS.get(code, "unknown")
