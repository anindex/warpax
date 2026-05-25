"""Opt-in, version-salted persistent JAX JIT cache.

Activated by setting ``WARPAX_JIT_CACHE=1``. The cache directory is

    ~/.cache/warpax/jax/<sha256(warpax_version:jaxlib_version:backend)[:16]>

The backend (``"cpu"`` or ``"gpu"``) is included in the salt so a CPU-XLA
artifact never serves a GPU consumer and vice versa.

Optional tunables (forwarded to JAX, only honored when the cache is on):

- ``WARPAX_JIT_CACHE_MIN_ENTRY_SIZE_BYTES`` -> ``jax_persistent_cache_min_entry_size_bytes``.
- ``WARPAX_JIT_CACHE_MIN_COMPILE_TIME_SECS`` -> ``jax_persistent_cache_min_compile_time_secs``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

__all__ = ["_compute_cache_salt", "_detect_backend", "_initialize_jit_cache"]

# Set once per process. Reset only by module reload (which is what the
# test suite does via monkeypatch).
_CACHE_INITIALIZED: bool = False


def _compute_cache_salt(warpax_version: str, jaxlib_version: str, backend: str) -> str:
    """Deterministic 16-char hex salt over ``(warpax, jaxlib, backend)``."""
    payload = f"{warpax_version}:{jaxlib_version}:{backend}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _detect_backend() -> str:
    """Return ``'cpu'`` or ``'gpu'`` from the first JAX device's platform."""
    import jax

    devices = jax.devices()
    if not devices:
        return "cpu"
    platform = devices[0].platform
    return "gpu" if platform in ("cuda", "gpu", "rocm") else "cpu"


def _initialize_jit_cache() -> None:
    """Idempotent: enable the persistent JIT cache when ``WARPAX_JIT_CACHE=1``."""
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

    min_entry = os.environ.get("WARPAX_JIT_CACHE_MIN_ENTRY_SIZE_BYTES")
    if min_entry is not None:
        try:
            jax.config.update(
                "jax_persistent_cache_min_entry_size_bytes", int(min_entry)
            )
        except (ValueError, AttributeError):
            pass

    min_time = os.environ.get("WARPAX_JIT_CACHE_MIN_COMPILE_TIME_SECS")
    if min_time is not None:
        try:
            jax.config.update(
                "jax_persistent_cache_min_compile_time_secs", float(min_time)
            )
        except (ValueError, AttributeError):
            pass

    _CACHE_INITIALIZED = True
