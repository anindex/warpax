"""Observer-robust energy condition verification for warp drive spacetimes.

Units: geometric (G = c = 1) throughout.
"""

# Float64 enforcement - must happen before any JAX imports that might
# create arrays with default float32 precision. Also set defensively
# in standalone scripts for the same reason.
import os

import jax

jax.config.update("jax_enable_x64", True)

# env-var-gated persistent JAX JIT compilation cache.
# Activated only when WARPAX_JIT_CACHE=1 is set; default behavior is unchanged.
# Cache path is version-salted at ~/.cache/warpax/jax/<16hex>/ to prevent
# cross-version / cross-backend poisoning. See src/warpax/_jit_cache.py.
from ._jit_cache import _initialize_jit_cache

_initialize_jit_cache()

# Opt-in strict runtime type checking across the whole package.
# Activate via: `WARPAX_BEARTYPE=1 python ...` (or `WARPAX_BEARTYPE=1 pytest`).
# Default off: zero import-time cost, zero call-time cost for production users.
if os.environ.get("WARPAX_BEARTYPE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()

__version__ = "0.2.0"
__author__ = "An T. Le"
__email__ = "an@robot-learning.de"

# Public API surface is frozen at v1.0. Any addition, rename, or removal
# must be reflected in tests/fixtures/v1_api_surface_v1_0.json AND
# tests/fixtures/v1_api_defaults_v1_0.json (defaults snapshot).
# Regenerate via `pytest tests/test_v1_api_surface.py --regenerate` and
# commit the diff with a CHANGELOG entry.
from . import design as design

__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "design",
]
