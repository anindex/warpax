"""Shared JSON output helper for scripts: RFC 8259-compliant dumps.

``json.dump`` defaults to ``allow_nan=True``, which emits bare ``NaN`` /
``Infinity`` literals that are invalid JSON (RFC 8259) and break strict
parsers. Every script that writes ``results/*.json`` must go through
:func:`dump_json`, which recursively converts non-finite floats to
``None`` (JSON ``null``) and dumps with ``allow_nan=False``.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any


def sanitize_nonfinite(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    Returns a new structure (dicts/lists rebuilt; tuples become lists,
    matching what ``json.dump`` would emit anyway). Non-float leaves are
    passed through unchanged.
    """
    if isinstance(obj, dict):
        return {k: sanitize_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_nonfinite(v) for v in obj]
    if isinstance(obj, float):  # includes np.float64 (a float subclass)
        return obj if math.isfinite(obj) else None
    return obj


def dump_json(
    obj: Any,
    path: str | os.PathLike[str],
    *,
    indent: int = 2,
    default: Any = None,
) -> None:
    """Write ``obj`` to ``path`` as strictly valid (RFC 8259) JSON.

    Non-finite floats (``nan``, ``inf``, ``-inf``) are converted to
    ``null``; ``allow_nan=False`` guarantees no bare ``NaN`` literal can
    ever be emitted.

    Parameters
    ----------
    obj : Any
        JSON-serializable payload (after non-finite sanitization).
    path : str or os.PathLike
        Output file path (opened in text-write mode).
    indent : int
        Indentation passed to ``json.dump`` (default 2).
    default : callable or None
        Fallback serializer passed to ``json.dump`` (e.g. ``str``).
    """
    with open(path, "w") as f:
        json.dump(
            sanitize_nonfinite(obj),
            f,
            indent=indent,
            allow_nan=False,
            default=default,
        )
