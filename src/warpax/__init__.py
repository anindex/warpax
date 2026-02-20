"""Observer-robust energy condition verification for warp drive spacetimes.

Units: geometric (G = c = 1) throughout.
"""

# Float64 enforcement - must happen before any JAX imports that might
# create arrays with default float32 precision.  Also set defensively
# in standalone scripts for the same reason.
import jax
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
__author__ = "An T. Le"
__email__ = "an@robot-learning.de"
