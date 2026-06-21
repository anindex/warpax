"""Observer-robust energy-condition verification for warp drive spacetimes.

Units: geometric (``G = c = 1``) throughout.

Environment variables:

- ``WARPAX_JIT_CACHE=1`` enables the version-salted persistent JIT cache
  at ``~/.cache/warpax/jax/<hash>/`` (see :mod:`warpax._jit_cache`).
- ``WARPAX_BEARTYPE=1`` installs whole-package runtime type checks via
  ``beartype.claw``. Off by default so production users pay nothing.
"""

import os

import jax

# Must run before any JAX array is created.
jax.config.update("jax_enable_x64", True)

from ._jit_cache import _initialize_jit_cache

_initialize_jit_cache()

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

__version__ = "1.2.0"
__author__ = "An T. Le"
__email__ = "an@robot-learning.de"

from . import design as design
from .certify import CertifyResult, certify

__all__ = [
    "CertifyResult",
    "__author__",
    "__email__",
    "__version__",
    "certify",
    "design",
]
