"""Private persistent JIT compilation cache module.

Implements an opt-in, version-salted JAX persistent compilation cache gated by
the ``WARPAX_JIT_CACHE`` environment variable. The cache path is deterministic:

    ~/.cache/warpax/jax/<sha256(warpax_version:jaxlib_version:backend)[:16]>

where ``backend`` is ``'cpu'`` or ``'gpu'`` (derived from ``jax.devices[0].platform``).

The ``:backend`` suffix extends the cache-key salt to prevent cross-backend
cache poisoning. A compiled artifact targeting CPU XLA must never be served
to a GPU consumer (and vice versa), so the backend selector is part of the
cache-key identity.

Activation:
    WARPAX_JIT_CACHE=1 -> persistent cache enabled
    WARPAX_JIT_CACHE=0 -> explicitly disabled (equivalent to unset)
    unset -> JAX default behavior (no persistent cache)

The module is private (underscore prefix): the only user-facing surface is the
environment variable. No public symbol is added, preserving the v0.1.x API
surface contract; ``tests/test_v1_api_surface.py`` remains green.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

__all__ = ["_compute_cache_salt", "_detect_backend", "_initialize_jit_cache"]

# Module-level guard: ensure the JAX cache config is mutated at most once per
# process. Re-importing warpax (e.g. in tests via monkeypatch) resets this flag
# by reassigning at module reload time, which is the intended behavior for
# the test suite.
_CACHE_INITIALIZED: bool = False


def _compute_cache_salt(warpax_version: str, jaxlib_version: str, backend: str) -> str:
    """Deterministic 16-char hex salt keyed on ``warpax``, ``jaxlib``, ``backend``.

    Parameters
    ----------
    warpax_version
        The warpax package version string (e.g. ``"0.1.0"``).
    jaxlib_version
        The jaxlib package version string (e.g. ``"0.10.0"``).
    backend
        One of ``"cpu"`` or ``"gpu"``. Prevents cross-backend cache poisoning.

    Returns
    -------
    str
        16-character hex digest prefix of ``sha256(warpax_version:jaxlib_version:backend)``.

    Notes
    -----
    The salt is deterministic: identical ``(warpax, jaxlib, backend)`` tuples
    always produce the same 16-char prefix. This enables cache reuse across
    sessions while protecting against silent version-drift.
    """
    payload = f"{warpax_version}:{jaxlib_version}:{backend}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _detect_backend() -> str:
    """Return ``'cpu'`` or ``'gpu'`` from the first JAX device's platform.

    Falls back to ``'cpu'`` if ``jax.devices`` is empty (which is extremely
    rare - typically only happens during a failed CUDA init).
    """
    import jax

    devices = jax.devices()
    if not devices:
        return "cpu"
    platform = devices[0].platform
    # JAX reports 'cuda' for GPU; collapse to 'gpu' for the salt namespace.
    return "gpu" if platform in ("cuda", "gpu", "rocm") else "cpu"


def _initialize_jit_cache() -> None:
    """Idempotent: env-var-gated JAX persistent compilation cache initialization.

    Called at ``import warpax`` time, immediately after
    ``jax.config.update("jax_enable_x64", True)`` and before any JAX array
    creation.

    Behaviour
    ---------
    - ``WARPAX_JIT_CACHE != "1"``: early return, JAX default behavior.
    - Already initialized (``_CACHE_INITIALIZED is True``): early return.
    - Otherwise: compute salt, create ``~/.cache/warpax/jax/<salt>/`` if
      needed, and call ``jax.config.update("jax_compilation_cache_dir", ...)``.
    """
    global _CACHE_INITIALIZED

    if os.environ.get("WARPAX_JIT_CACHE") != "1":
        return
    if _CACHE_INITIALIZED:
        return

    import importlib.metadata as _metadata

    import jax

    warpax_version = _metadata.version("warpax")
    jaxlib_version = _metadata.version("jaxlib")
    backend = _detect_backend()
    salt = _compute_cache_salt(warpax_version, jaxlib_version, backend)

    cache_dir = Path.home() / ".cache" / "warpax" / "jax" / salt
    cache_dir.mkdir(parents=True, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", str(cache_dir))

    _CACHE_INITIALIZED = True
