"""Shared helpers for the test suite: script loading, flat metric, Schwarzschild coordinates, photon rocket."""

from __future__ import annotations

import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")

ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


def load_script(name: str):
    """Load ``scripts/<name>`` as a module; the scripts dir goes on ``sys.path`` for sibling imports."""
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    path = os.path.join(SCRIPTS_DIR, name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def schw_r_to_iso(r_schw, M=1.0):
    """Convert Schwarzschild radial coordinate to isotropic."""
    return (r_schw - M + np.sqrt(r_schw**2 - 2 * M * r_schw)) / 2.0


def iso_to_schw_r(r_iso, M=1.0):
    """Convert isotropic radial coordinate to Schwarzschild."""
    return r_iso * (1 + M / (2 * r_iso)) ** 2


def make_krt(G0, M0=1.0, M1=0.05, OM=0.5, scan_length=12):
    """Exact accelerating Kerr-Schild photon rocket; G0 -> 0 gives Vaidya.

    ``scan_length`` is the retarded-``u`` fixed-point iteration count; 12 converges
    at r ~ 10 M, the peeling tests at r up to a few hundred M use 80.
    """

    def worldline(u):
        sx = jnp.sinh(G0 * u) / G0 if G0 != 0 else u
        xx = (jnp.cosh(G0 * u) - 1.0) / G0 if G0 != 0 else 0.0
        return jnp.array([sx, xx, 0.0, 0.0])

    def m_of(u):
        return M0 + M1 * jnp.sin(OM * u)

    @jax.custom_jvp
    def retarded_u(X):
        def body(u, _):
            z = worldline(u)
            zdot = jax.jacfwd(worldline)(u)
            D = X - z
            r = -(ETA @ zdot) @ D
            F = -(D[0] ** 2) + D[1] ** 2 + D[2] ** 2 + D[3] ** 2
            return u - F / (2.0 * r), None

        u0 = X[0] - jnp.sqrt(X[1] ** 2 + X[2] ** 2 + X[3] ** 2)
        u, _ = jax.lax.scan(body, u0, None, length=scan_length)
        return u

    @retarded_u.defjvp
    def _jvp(primals, tangents):
        (X,), (dX,) = primals, tangents
        u = retarded_u(X)
        z = worldline(u)
        zdot = jax.jacfwd(worldline)(u)
        D = X - z
        r = -(ETA @ zdot) @ D
        l_dn = ETA @ (D / r)
        return u, -(l_dn @ dX)

    def metric_fn(X):
        u = retarded_u(X)
        z = worldline(u)
        zdot = jax.jacfwd(worldline)(u)
        D = X - z
        r = -(ETA @ zdot) @ D
        l_up = D / r
        l_dn = ETA @ l_up
        return ETA + (2.0 * m_of(u) / r) * jnp.outer(l_dn, l_dn)

    return metric_fn, m_of


def krt_cone(r, theta):
    """Point on the t = r null cone at polar angle ``theta`` in the (x, y) plane."""
    c, s = np.cos(theta), np.sin(theta)
    return jnp.array([r, r * c, r * s, 0.0])
