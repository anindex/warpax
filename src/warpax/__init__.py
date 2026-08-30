"""Observer-robust energy-condition verification for warp drive spacetimes.

Units: geometric (``G = c = 1``) throughout.

Environment variables:

- ``WARPAX_JIT_CACHE=1`` points JAX's persistent compilation cache at
  ``~/.cache/warpax/jax``. Override the location with
  ``JAX_COMPILATION_CACHE_DIR``.
- ``WARPAX_BEARTYPE=1`` installs whole-package runtime type checks via
  ``beartype.claw``. Off by default so production users pay nothing.
"""

import os

import jax

# Must run before any JAX array is created.
jax.config.update("jax_enable_x64", True)

# JAX's own cache key already covers the HLO, the jaxlib version, the backend
# and the devices, so a directory is all this needs.
if os.environ.get("WARPAX_JIT_CACHE") == "1":
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.expanduser("~/.cache/warpax/jax"),
    )

if os.environ.get("WARPAX_BEARTYPE") == "1":
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    # jaxtyping subscripts (e.g. Float[Array, "4 4"]) in NamedTuple fields
    # become unresolvable forward references under PEP 563. Converting the
    # resulting decorator exception to a warning lets beartype still check
    # all decorated functions while skipping NamedTuple class annotations.
    beartype_this_package(
        conf=BeartypeConf(warning_cls_on_decorator_exception=UserWarning),
    )

__version__ = "1.4.0"
__author__ = "An T. Le"
__email__ = "an@robot-learning.de"

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
    # design pulls in sympy: 1.6 s of the 2.5 s import, paid by every script
    # and every pool worker. Import it on first use instead.
    if name == "design":
        import importlib

        # import_module, not "from . import design": the latter re-enters here.
        return importlib.import_module(".design", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
