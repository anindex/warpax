"""Shared Diffrax result-code conversion and termination-reason mapping.

Diffrax >= 0.6 reports ``sol.result`` as an equinox ``EnumerationItem``
(carrying ``._value``, an int32 array) rather than a plain int, and the
member indices differ from older releases. This module centralizes the
robust conversion used by :mod:`warpax.averaged.anec` and
:mod:`warpax.averaged.awec` so a non-convertible result is *never*
silently mapped to success.

``RESULT_UNKNOWN`` (-1) marks a result that could not be converted; it maps
to reason ``'unknown'`` and callers must treat it as non-success.
``RESULT_SUCCESS`` is ``diffrax.RESULTS.successful``. ``TERMINATION_REASONS``
is verified against the installed diffrax 0.7.2 ``RESULTS`` enumeration
(index -> member name via ``RESULTS._name_to_item``); codes 2-8 are linear
and nonlinear solve failures, 9 is non-finite values during the solve.
"""

from __future__ import annotations

RESULT_UNKNOWN = -1

RESULT_SUCCESS = 0

TERMINATION_REASONS: dict[int, str] = {
    RESULT_UNKNOWN: "unknown",
    0: "complete",
    1: "max_steps",
    2: "singular",
    3: "breakdown",
    4: "stagnation",
    5: "conlim",
    6: "nonfinite_input",
    7: "nonlinear_max_steps",
    8: "nonlinear_divergence",
    9: "nonfinite",
    10: "dt_min_reached",
    11: "event_occurred",
    12: "max_steps_rejected",
    13: "internal_error",
}


def result_code_to_int(raw: object) -> int:
    """Convert a Diffrax result (EnumerationItem, array, or int) to int.

    Tries ``._value`` (equinox EnumerationItem on diffrax 0.7.x), then
    ``.value`` (older conventions), then ``int(raw)``. Returns
    :data:`RESULT_UNKNOWN`, never the success code, when no
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
        return int(raw)
    except (TypeError, ValueError):
        return RESULT_UNKNOWN


def termination_reason(code: int) -> str:
    """Human-readable reason for an integer result code."""
    return TERMINATION_REASONS.get(code, "unknown")
