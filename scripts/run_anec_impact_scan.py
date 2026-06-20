"""ANEC robustness scan over impact parameter and resolution.

Backs the source-first ANEC claim in the warp-shell paper: the symplectic
geodesic-integrated null energy line integral int T_ab k^a k^b dlambda is
positive for every source-prescribed shell, and -- the invariant statement --
its SIGN is robust across impact parameter and integration resolution. Only the
sign is invariant under k^a -> lambda k^a; magnitudes use the "fixed" tangent
norm (matching figures/make_fig4_anec.py) and are parametrization-dependent.

The on-cone witness max|g_ab k^a k^b| certifies the path stayed null; it
decreases under resolution refinement toward the paper's stated 1e-4 tolerance.
"""
from __future__ import annotations

from pathlib import Path

from _json_io import dump_json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from warpax.geodesics.symplectic import null_ic_canonical, integrate_geodesic_symplectic
from warpax.averaged.anec import _extract_trajectory, anec  # noqa: F401
from warpax.metrics import fuchs_default, sshell_default, tshell_default

X0, SPAN = -30.0, 60.0
IMPACTS = [1e-3, 0.5, 1.0, 2.0, 5.0]
RESOLUTIONS = [8192, 16384, 32768]


def _anec_one(metric, b, ns):
    x0 = jnp.array([0.0, X0, float(b), 0.0])
    x0c, p0 = null_ic_canonical(metric, x0, jnp.array([1.0, 0.0, 0.0]))
    geo = integrate_geodesic_symplectic(
        metric, x0c, p0, (0.0, SPAN), num_steps=int(ns), order=4, omega=1.0,
    )
    res = anec(metric, geo, tangent_norm="fixed")
    return float(res.line_integral), float(res.max_abs_g_kk)


def scan(name, metric):
    rows = []
    for b in IMPACTS:
        for ns in RESOLUTIONS:
            li, w = _anec_one(metric, b, ns)
            rows.append({"impact": b, "n_steps": ns, "line_integral": li,
                         "witness": w, "sign": int(li > 0) - int(li < 0)})
            print(f"  {name:8} b={b:<5} NS={ns:<6} I={li:+.3e}  witness={w:.2e}")
    signs = {r["sign"] for r in rows}
    finest = [r for r in rows if r["n_steps"] == max(RESOLUTIONS)]
    return {
        "rows": rows,
        "sign_robust_positive": signs == {1},
        "min_witness": min(r["witness"] for r in rows),
        "max_witness": max(r["witness"] for r in rows),
        "line_integral_finest_b1e-3": next(
            r["line_integral"] for r in finest if r["impact"] == 1e-3),
    }


def main():
    cases = [
        ("Fuchs", fuchs_default(v_s=0.02)),
        ("S-shell", sshell_default(v_s=0.0)),
        ("T-shell", tshell_default(v_0=0.1)),
        ("T-shell_v0.2", tshell_default(v_0=0.2)),
    ]
    out = {
        "config": {"x0": X0, "span": SPAN, "impacts": IMPACTS,
                   "resolutions": RESOLUTIONS, "tangent_norm": "fixed",
                   "note": "Only the SIGN of the line integral is invariant; "
                           "magnitudes are parametrization-dependent."},
    }
    for name, m in cases:
        print(f"Scanning {name}...")
        out[name] = scan(name, m)

    out_path = Path(__file__).resolve().parents[1] / "results" / "anec" / "source_first_impact_scan.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out, out_path)
    print(f"\nSaved: {out_path}")
    for name, _ in cases:
        d = out[name]
        print(f"{name:14} sign_robust_positive={d['sign_robust_positive']}  "
              f"witness in [{d['min_witness']:.2e}, {d['max_witness']:.2e}]")


if __name__ == "__main__":
    main()
