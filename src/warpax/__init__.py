"""Observer-robust energy-condition verification for warp drive spacetimes.

Units: geometric (``G = c = 1``) throughout.

Environment variables:

- ``WARPAX_JIT_CACHE=1`` points JAX's persistent compilation cache at
  ``~/.cache/warpax/jax``. Override the location with
  ``JAX_COMPILATION_CACHE_DIR``. It also lowers the persistent-cache
  thresholds (compile time 0.05 s, no size floor) unless
  ``JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS`` /
  ``JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES`` are set, because JAX's
  1 s default caches none of warpax's kernels.
- ``WARPAX_BEARTYPE=1`` installs whole-package runtime type checks via
  ``beartype.claw``. Off by default so production users pay nothing.

``jax_enable_x64`` is set here, before any JAX array is created. The cache
directory is set through ``jax.config`` rather than the environment: JAX
reads ``JAX_COMPILATION_CACHE_DIR`` at its own import, which has already
happened. JAX's cache key already covers the HLO, jaxlib version, backend
and devices, so a directory is all it needs.
"""

import os

import jax

jax.config.update("jax_enable_x64", True)

if os.environ.get("WARPAX_JIT_CACHE") == "1":
    jax.config.update(
        "jax_compilation_cache_dir",
        os.environ.get("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/warpax/jax")),
    )
    if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ:
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.05)
    if "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES" not in os.environ:
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

if os.environ.get("WARPAX_BEARTYPE") == "1":
    import collections

    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    _namedtuple = collections.namedtuple

    def _namedtuple_unchecked(*args, **kwargs):
        """namedtuple whose __new__ beartype skips: pytree rebuilds pass batched or bool leaves."""
        cls = _namedtuple(*args, **kwargs)
        cls.__new__.__no_type_check__ = True
        return cls

    collections.namedtuple = _namedtuple_unchecked

    beartype_this_package(
        conf=BeartypeConf(
            warning_cls_on_decorator_exception=UserWarning,
            is_pep484_tower=True,
        ),
    )

__version__ = "1.6.0"
__author__ = "An T. Le"
__email__ = "an.thai.le97@gmail.com"

from .certify import CertifyResult, certify

__all__ = [
    "CertifyResult",
    "__author__",
    "__email__",
    "__version__",
    "certify",
    "design",
]


def __getattr__(name):
    """Lazy ``design`` import.

    ``design`` pulls in sympy, 1.6 s of the 2.5 s package import, paid by
    every script and pool worker. Uses ``importlib.import_module`` because
    ``from . import design`` re-enters this hook.
    """
    if name == "design":
        import importlib

        return importlib.import_module(".design", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
