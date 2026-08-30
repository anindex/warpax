"""Slice-integrated negative-energy measures vs warp speed.

The paper reports the *pointwise* wall NEC deficit and its ``v_s^2`` speed law
(``run_ssv_bound.py``). This script is the slice-integrated companion: on the
``t = 0`` slice we integrate, per metric, over a subluminal ``v_s`` sweep,

    E_minus(v_s) = int_{rho<0} |rho| dV          (Eulerian negative-energy volume)

with the Eulerian energy density ``rho = T_ab n^a n^b`` (n the unit Eulerian
normal).  ``rho`` is well defined at every point irrespective of Hawking-Ellis
type, which is why this is the integrated measure we report.  An integrated
*NEC* deficit built from the rest-frame ``min_i(rho+p_i)`` was removed: the
rest-frame quantities exist only at Type-I points, so such an integral is
undefined on the Type-IV-dominated walls it was applied to.
The proper volume element is ``sqrt(det gamma_ij) * (cell volume)`` on the
wall-clustered grid, so the Van den Broeck conformal spatial factor (det gamma
!= 1) is handled exactly; flat-slice drives reduce to ``dx dy dz``.

For a unit-lapse, spatially flat drive the shift is linear in ``v_s`` and the
Eulerian energy density is quadratic (Hamiltonian constraint ``16 pi rho = K^2 -
K_ij K^ij ~ v_s^2`` on a flat slice), so ``E_-`` scales as ``v_s^2``: PHYSICS
EXPECTATION ``p ~ 2`` for the pure warp drives (Alcubierre, Natario, Rodal). This
connects the pointwise Santiago-Schuster-Visser deficit to Rodal's
slice-integrated negative-energy volume (arXiv:2512.18008).

One measured departure, physical, not a bug:
  * Van den Broeck carries a ``v_s``-independent conformal negative-energy offset
    (its volume distortion needs exotic matter even at ``v_s=0``), so its ``E_-``
    is nearly flat in ``v_s`` (fitted ``p ~ 0``).

Outputs
-------
- results/integrated_negative_energy.json
- ../warpax_arxiv/tables/integrated_volume.tex
"""
from __future__ import annotations

import os

from _json_io import dump_json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.benchmarks import AlcubierreMetric
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric
from warpax.geometry import evaluate_curvature_grid
from warpax.grids import wall_clustered


HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]
# (class, kwargs, bounds). Matched family R_b=1, sigma=8; Rodal at native params.
FAMILY = {
    "Alcubierre": (AlcubierreMetric, {"R": 1.0, "sigma": 8.0}, [(-3.0, 3.0)] * 3),
    "Natário": (NatarioMetric, {"R": 1.0, "sigma": 8.0}, [(-3.0, 3.0)] * 3),
    "Van den Broeck": (
        VanDenBroeckMetric,
        {"R": 1.0, "sigma": 8.0, "R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0},
        [(-3.0, 3.0)] * 3,
    ),
    "Rodal": (RodalMetric, {"R": 100.0, "sigma": 0.03}, [(-300.0, 300.0)] * 3),
}
V_S_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N = 48  # wall-clustered resolution (E_- stable to <3% from N=25->40)


def _instantiate(name, v_s):
    cls, kw, _ = FAMILY[name]
    return cls(v_s=v_s, **kw)


def _eulerian_rho(T_ab, g_inv):
    """Eulerian energy density rho = T_ab n^a n^b, n unit timelike (lapse-free)."""
    n_low = jnp.array([-1.0, 0.0, 0.0, 0.0])
    n_up = g_inv @ n_low
    n_up = n_up / jnp.sqrt(jnp.abs(n_low @ n_up))
    return n_up @ (T_ab @ n_up)


_rho_grid = jax.jit(jax.vmap(_eulerian_rho))


def _proper_dV(g_field, grid):
    """Proper volume element sqrt(det gamma_ij) * cell-volume, per grid point."""
    flat = g_field.reshape(-1, 4, 4)
    gamma = flat[:, 1:, 1:]  # spatial 3x3 block == ADM gamma_ij
    sqrtdet = np.asarray(jnp.sqrt(jnp.clip(jnp.linalg.det(gamma), min=0.0)))
    w = np.asarray(grid.volume_weights_array).ravel()
    return sqrtdet * w


def _measures_at(name, grid, v_s):
    metric = _instantiate(name, v_s)
    curv = evaluate_curvature_grid(metric, grid, batch_size=4096)
    rho = np.asarray(
        _rho_grid(curv.stress_energy.reshape(-1, 4, 4), curv.metric_inv.reshape(-1, 4, 4))
    )
    dV = _proper_dV(curv.metric, grid)
    neg = rho < 0.0
    return float(np.sum(np.abs(rho[neg]) * dV[neg]))


def _loglog_fit(vs, ys):
    """Fit y = A v_s^p on log-log; return (p, A, R^2, n)."""
    vs = np.asarray(vs, float)
    ys = np.asarray(ys, float)
    ok = np.isfinite(ys) & (ys > 0.0)
    vs, ys = vs[ok], ys[ok]
    if len(vs) < 3:
        return {"p": None, "A": None, "r_squared": None, "n": int(len(vs))}
    lv, ly = np.log(vs), np.log(ys)
    p, logA = np.polyfit(lv, ly, 1)
    pred = p * lv + logA
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"p": float(p), "A": float(np.exp(logA)), "r_squared": float(r2),
            "n": int(len(vs))}


def _f(x, nd=3):
    return f"{x:.{nd}f}" if (x is not None and np.isfinite(x)) else "--"


def write_table(results, out_path):
    lines = [
        "% Generated by scripts/run_integrated_negative_energy.py; do not edit.",
        r"\begin{tabular}{@{}l cc@{}}",
        r"  \toprule",
        r"  & \multicolumn{2}{c}{$E_-=\int_{\rho_n<0}|\rho_n|\,dV$} \\",
        r"  \cmidrule(lr){2-3}",
        r"  Metric & exponent $p$ & $R^2$ \\",
        r"  \midrule",
    ]
    for name in ORDER:
        fe = results[name]["fit_E_minus"]
        lines.append(
            f"  {name} & {_f(fe['p'],2)} & {_f(fe['r_squared'],4)} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main():
    print("=" * 72)
    print(f"SLICE-INTEGRATED NEGATIVE ENERGY vs v_s  (N={N}, wall-clustered)")
    print("=" * 72)
    results = {}
    for name in ORDER:
        # Wall-clustered grid is v_s-independent (shape geometry only): build once.
        grid = wall_clustered(_instantiate(name, 0.5), FAMILY[name][2], (N, N, N), a=1.2)
        E_list = [_measures_at(name, grid, v_s) for v_s in V_S_SWEEP]
        fit_E = _loglog_fit(V_S_SWEEP, E_list)
        results[name] = {
            "v_s": list(V_S_SWEEP),
            "E_minus": E_list,
            "fit_E_minus": fit_E,
        }
        print(f"\n{name}:")
        print("  v_s   : " + " ".join(f"{v:8.2f}" for v in V_S_SWEEP))
        print("  E_-   : " + " ".join(f"{e:8.3e}" for e in E_list))
        print(f"  fit E_-   : p={_f(fit_E['p'],3)}  R^2={_f(fit_E['r_squared'],4)}")

    out = {
        "params": {"N": N, "v_s_sweep": list(V_S_SWEEP),
                   "note": "matched family R_b=1 sigma=8; Rodal native R=100 sigma=0.03; "
                           "proper dV = sqrt(det gamma_ij)*cell_volume"},
        "order": ORDER,
        "results": results,
    }
    dump_json(out, os.path.join(RESULTS_DIR, "integrated_negative_energy.json"))
    print(f"\nWrote {os.path.join(RESULTS_DIR, 'integrated_negative_energy.json')}")
    write_table(results, os.path.join(TABLES_DIR, "integrated_volume.tex"))


if __name__ == "__main__":
    main()
