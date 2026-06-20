"""T-shell outer-edge Type-IV: linear-in-v0 fit + three-solver / 50-digit gate.

Backs two claims in source_first_arxiv:
  (i)  the opened imaginary-eigenvalue scale grows linearly with the matter tilt
       v0 at the genuine low-density OUTER edge (r >= R2), excluding the inner
       vacuum r < R1 (a uniform-shift gauge artifact);
  (ii) the outer-edge Type-IV label survives a three-solver gate: the standard and
       generalized-pencil eigensolvers plus an independent mpmath 50-digit
       recomputation all return a complex-conjugate eigenvalue pair.

Run (after `uv sync --extra design --extra solver`):
    uv run python scripts/run_tshell_typeIV_gate.py
"""
from __future__ import annotations
import json
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import mpmath as mp

from warpax.geometry import compute_curvature_chain
from warpax.energy_conditions.frame_free import certify_point_frame_free
from warpax.metrics import tshell_default

R1, R2 = 10.0, 20.0
V0_GRID = [0.01, 0.05, 0.10, 0.20]
# Genuine low-density outer edge only: r >= R2, excluding the inner vacuum r < R1.
R_OUTER = np.linspace(R2, 26.0, 120)


def _cert(metric, r, solver="auto"):
    cur = compute_curvature_chain(metric, jnp.array([0.0, float(r), 0.0, 0.0]))
    res = certify_point_frame_free(cur.stress_energy, cur.metric, cur.metric_inv,
                                   solver=solver)
    return res, cur


def outer_edge_imag(metric):
    """max |Im eigenvalue| over the outer edge, with the radius of the max."""
    best, best_r = 0.0, R2
    for r in R_OUTER:
        res, _ = _cert(metric, r)
        m = float(np.nanmax(np.abs(np.asarray(res["eigenvalues_imag"]))))
        if m > best:
            best, best_r = m, float(r)
    return best, best_r


# (i) linear-in-v0 fit at the outer edge -----------------------------------------
rows = []
for v0 in V0_GRID:
    imax, rmax = outer_edge_imag(tshell_default(v_0=v0))
    rows.append({"v0": v0, "max_abs_imag": imax, "r_argmax": rmax})
    print(f"v0={v0:<5}  max|imag|(r>=R2)={imax:.6e}  at r={rmax:.3f}")

v = np.array([r["v0"] for r in rows])
y = np.array([r["max_abs_imag"] for r in rows])
lv, ly = np.log10(v), np.log10(y)
slope, intercept = np.polyfit(lv, ly, 1)
# standard error of the slope from the residuals
n = len(lv)
yhat = slope * lv + intercept
resid = ly - yhat
sxx = np.sum((lv - lv.mean()) ** 2)
s_err = float(np.sqrt(np.sum(resid ** 2) / (n - 2) / sxx)) if n > 2 else float("nan")
r2 = 1.0 - np.sum(resid ** 2) / np.sum((ly - ly.mean()) ** 2)
print(f"\nlog-log fit (outer edge r>=R2): slope = {slope:.4f} +/- {s_err:.4f}  (R^2={r2:.5f})")

# (ii) three-solver / 50-digit gate at the v0=0.1 outer-edge Type-IV point --------
m01 = tshell_default(v_0=0.10)
_, r_gate = outer_edge_imag(m01)
gate = {"r": r_gate, "solvers": {}}
for solver in ("standard", "generalized"):
    res, cur = _cert(m01, r_gate, solver=solver)
    imag = np.asarray(res["eigenvalues_imag"])
    gate["solvers"][solver] = {
        "he_type": int(res["he_type"]),
        "max_abs_imag": float(np.nanmax(np.abs(imag))),
    }
    print(f"gate r={r_gate:.3f} solver={solver:<12} he_type={int(res['he_type'])} "
          f"max|imag|={float(np.nanmax(np.abs(imag))):.6e}")

# independent mpmath 50-digit eigenvalues of T^a_b = g^{ac} T_{cb}
_, cur = _cert(m01, r_gate)
T_mixed = np.asarray(jnp.einsum("ac,cb->ab", cur.metric_inv, cur.stress_energy))
mp.mp.dps = 50
M = mp.matrix(T_mixed.tolist())
E, _ = mp.eig(M)
imag_parts = [abs(mp.im(e)) for e in E]
max_imag_mp = max(float(x) for x in imag_parts)
n_complex = sum(1 for x in imag_parts if float(x) > 1e-12)
gate["mpmath_50digit"] = {"dps": 50, "max_abs_imag": max_imag_mp,
                          "n_complex_eigs": int(n_complex)}
print(f"gate r={r_gate:.3f} solver=mpmath-50dps  n_complex_eigs={n_complex} "
      f"max|imag|={max_imag_mp:.6e}")

all_typeIV = (gate["solvers"]["standard"]["he_type"] == 4
              and gate["solvers"]["generalized"]["he_type"] == 4
              and n_complex >= 2)
gate["all_three_confirm_typeIV"] = bool(all_typeIV)
print(f"\nthree-solver gate confirms Type-IV at outer edge: {all_typeIV}")

out = {
    "config": {"R1": R1, "R2": R2, "v0_grid": V0_GRID,
               "outer_edge_range": [float(R2), 26.0], "note":
               "max|Im eig| over r>=R2 (outer edge), inner vacuum r<R1 excluded"},
    "linear_fit": {"slope": float(slope), "slope_stderr": s_err, "r2": float(r2),
                   "rows": rows},
    "three_solver_gate": gate,
}
dest = os.path.join(os.path.dirname(__file__), "..", "results",
                    "tshell_typeIV_gate.json")
with open(dest, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nsaved {os.path.normpath(dest)}")
