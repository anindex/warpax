"""tests for the persistent JIT compilation cache module.

Covers the three env-var states (unset / '0' / '1'), salt determinism, and
backend-suffix disambiguation. All tests respect the module-level
``_CACHE_INITIALIZED`` guard by resetting it to ``False`` before each
initialization call.
"""

from __future__ import annotations

import re

import jax
import pytest

from warpax import _jit_cache
from warpax._jit_cache import _compute_cache_salt, _initialize_jit_cache


class TestJITCache:
    """Env-var-gated JIT cache activation + salt determinism."""

    def test_env_var_1_sets_cache_dir(self, monkeypatch, tmp_path):
        """WARPAX_JIT_CACHE=1 routes compile cache to a salted path.

        Redirects the user cache root to a tmp path via ``HOME`` override so
        the test leaves no residue in the developer's real ``~/.cache``.
        """
        monkeypatch.setenv("WARPAX_JIT_CACHE", "1")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(_jit_cache, "_CACHE_INITIALIZED", False)

        _initialize_jit_cache()

        cache_dir = jax.config.values.get("jax_compilation_cache_dir", "")
        assert "warpax/jax/" in cache_dir
        # Path tail must be 16 hex chars
        tail = cache_dir.rsplit("/", 1)[-1]
        assert re.match(r"^[0-9a-f]{16}$", tail), (
            f"Expected 16-char hex salt, got {tail!r}"
        )

    def test_env_var_0_leaves_cache_dir_unset(self, monkeypatch):
        """WARPAX_JIT_CACHE=0 is equivalent to unset: no config update."""
        monkeypatch.setenv("WARPAX_JIT_CACHE", "0")
        monkeypatch.setattr(_jit_cache, "_CACHE_INITIALIZED", False)

        before = jax.config.values.get("jax_compilation_cache_dir", None)
        _initialize_jit_cache()
        after = jax.config.values.get("jax_compilation_cache_dir", None)

        assert before == after, (
            "WARPAX_JIT_CACHE=0 must leave JAX cache config unchanged"
        )

    def test_env_var_unset_leaves_cache_dir_unset(self, monkeypatch):
        """Unset env var -> early-return; JAX default behavior preserved."""
        monkeypatch.delenv("WARPAX_JIT_CACHE", raising=False)
        monkeypatch.setattr(_jit_cache, "_CACHE_INITIALIZED", False)

        before = jax.config.values.get("jax_compilation_cache_dir", None)
        _initialize_jit_cache()
        after = jax.config.values.get("jax_compilation_cache_dir", None)

        assert before == after, (
            "Unset WARPAX_JIT_CACHE must leave JAX cache config unchanged"
        )

    def test_salt_deterministic(self):
        """_compute_cache_salt is pure + deterministic + 16 hex chars."""
        salt_a = _compute_cache_salt("1.1.0", "0.10.0", "cpu")
        salt_b = _compute_cache_salt("1.1.0", "0.10.0", "cpu")
        assert salt_a == salt_b
        assert re.match(r"^[0-9a-f]{16}$", salt_a)

    def test_backend_suffix_disambiguates(self):
        """CPU and GPU salts must differ to prevent cross-backend poisoning."""
        cpu = _compute_cache_salt("1.1.0", "0.10.0", "cpu")
        gpu = _compute_cache_salt("1.1.0", "0.10.0", "gpu")
        assert cpu != gpu, (
            "Backend-suffix discriminator missing from salt: "
            f"cpu={cpu!r} gpu={gpu!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
