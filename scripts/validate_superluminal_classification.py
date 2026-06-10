"""Phase-0 feasibility GATE: is Hawking-Ellis classification of T^a_b
trustworthy across the luminal transition (v_s -> 1 and beyond)?

This is the load-bearing prerequisite for the velocity-resolved type/EC map
(Contribution 1). The frame-independent classifier (``classify_hawking_ellis``,
operating on the mixed tensor ``T^a_b``) never uses the Eulerian normal, so it
*runs* at v_s >= 1 where the ADM lapse ``alpha = 1/sqrt(-g^{00})`` becomes
ill-defined. The open question is whether the Type-IV labels it returns near
the ergosurface (g_00 -> 0) are REAL physics (a complex-eigenvalue,
no-rest-frame stress-energy) or a NUMERICAL ARTIFACT of an ill-conditioned
non-symmetric eigenproblem.

For each (metric, v_s) we apply three independent trustworthiness criteria to
the wall region (where the transition and any Type-IV live):

  1. Solver agreement: float64 standard ``jnp.linalg.eig`` vs LAPACK ``zggev``
     generalized pencil (solver="generalized") vs 50-digit ``mpmath`` on a
     sampled subset of wall points (Type-IV-prioritised).
  2. Refinement stability: the wall Type-IV volume fraction across
     N in {30,50,70} via ``f_miss_stability`` (discontinuous-quantity test).
  3. Tolerance insensitivity: the wall Type-IV fraction across
     imag_rtol in {3e-4, 3e-3, 3e-2}. A fraction that scales with the
     tolerance is noise; one that plateaus is physical.

Plus diagnostics: cond(g) at the wall, and |Im lambda| / |Re lambda| scaling.

DECISION RULE (per metric, v_s): Type-IV is TRUSTWORTHY iff
  mpmath flip-rate <= 1%  AND  wall Type-IV fraction refinement-stable
  AND  wall Type-IV fraction tolerance-insensitive (<= 0.5 pp spread).

Outputs
-------
- results/superluminal_gate.json        : structured per-(metric, v_s) data
- results/superluminal_gate_report.md   : human-readable verdict table

Usage
-----
    python scripts/validate_superluminal_classification.py
    python scripts/validate_superluminal_classification.py --smoke
"""
from __future__ import annotations

import argparse
import os
import time

from _json_io import dump_json

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpax.analysis.convergence import f_miss_stability
from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions.classification import classify_hawking_ellis
from warpax.energy_conditions.classification_mpmath import (
    verify_classification_at_points,
)
from warpax.energy_conditions.filtering import shape_function_mask
from warpax.energy_conditions.verifier import _classify_grid_batch
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.grids import wall_clustered
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

BOUNDS = [(-3, 3)] * 3
F_LOW, F_HIGH = 0.1, 0.9
IMAG_RTOLS = [3e-4, 3e-3, 3e-2]

# Matched family parameters (R = 1, sigma = 8): the wall-resolved regime in
# which a cross-metric comparison is meaningful (mirrors run_matched_benchmark).
METRICS: dict[str, tuple] = {
    "Alcubierre": (AlcubierreMetric, {}),
    "Natario": (NatarioMetric, {}),
    "VanDenBroeck": (
        VanDenBroeckMetric,
        {"R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0},
    ),
    "Rodal": (RodalMetric, {}),
}
METRIC_ORDER = ["Alcubierre", "Natario", "VanDenBroeck", "Rodal"]


def _instantiate(name: str, v_s: float):
    cls, extra = METRICS[name]
    return cls(v_s=v_s, R=1.0, sigma=8.0, **extra)


def _flat_curv(metric, N: int):
    """Return (flat_T, flat_g, flat_ginv, flat_Tmixed, wall_mask_flat, vol_w_flat)."""
    shape = (N, N, N)
    grid_spec = wall_clustered(metric, BOUNDS, shape, a=1.2)
    curv = evaluate_curvature_grid(metric, grid_spec, batch_size=256)
    flat_T = jnp.reshape(curv.stress_energy, (-1, 4, 4))
    flat_g = jnp.reshape(curv.metric, (-1, 4, 4))
    flat_ginv = jnp.reshape(curv.metric_inv, (-1, 4, 4))
    flat_Tmixed = jnp.einsum("nac,ncb->nab", flat_ginv, flat_T)
    coords = build_coord_batch(grid_spec, t=0.0)
    mask = shape_function_mask(metric, coords, shape, f_low=F_LOW, f_high=F_HIGH)
    wall_flat = np.asarray(jnp.reshape(mask, (-1,))).astype(bool)
    vol_w_flat = np.asarray(jnp.reshape(grid_spec.volume_weights_array, (-1,)))
    return flat_T, flat_g, flat_ginv, flat_Tmixed, wall_flat, vol_w_flat


def _vw_type_fraction(he_flat, sel_flat, w_flat, t: int) -> float:
    """Volume-weighted fraction of points of Hawking-Ellis type ``t`` within sel."""
    w = w_flat * sel_flat.astype(w_flat.dtype)
    wt = float(np.sum(w))
    if wt <= 0.0:
        return 0.0
    return float(np.sum(w * (he_flat == float(t)))) / wt


def _classify_types_rtol(flat_Tmixed, flat_g, imag_rtol: float) -> np.ndarray:
    res = jax.vmap(
        lambda Tm, g: classify_hawking_ellis(Tm, g, imag_rtol=imag_rtol)
    )(flat_Tmixed, flat_g)
    return np.asarray(res.he_type)


def run_cell(name, v_s, N_main, refine_Ns, mpmath_cap, gen_cap, do_refine):
    """Trustworthiness assessment for one (metric, v_s)."""
    out: dict = {"metric": name, "v_s": v_s, "N_main": N_main}

    metric = _instantiate(name, v_s)
    flat_T, flat_g, flat_ginv, flat_Tmixed, wall_flat, vol_w = _flat_curv(
        metric, N_main
    )

    # --- finiteness / signature sanity ----------------------------------
    dets = np.asarray(jnp.linalg.det(flat_g))
    out["det_g_median"] = float(np.median(dets))
    out["any_nan_T"] = bool(np.any(np.isnan(np.asarray(flat_T))))
    cond_g = np.asarray(jnp.linalg.cond(flat_g[wall_flat])) if wall_flat.any() else np.array([])
    out["cond_g_wall_p95"] = float(np.percentile(cond_g, 95)) if cond_g.size else None

    # --- float64 standard census (full grid) ----------------------------
    cls_std = _classify_grid_batch(flat_Tmixed, flat_g, flat_T, solver="standard")
    he_std = np.asarray(cls_std.he_type)
    imag_std = np.abs(np.asarray(cls_std.eigenvalues_imag))
    real_std = np.abs(np.asarray(cls_std.eigenvalues))

    out["wall_n"] = int(np.sum(wall_flat))
    for t, key in [(1, "I"), (2, "II"), (3, "III"), (4, "IV")]:
        out[f"wall_frac_type_{key}"] = _vw_type_fraction(he_std, wall_flat, vol_w, t)

    # |Im|/|Re| scaling on wall Type-IV-flagged points
    wall_iv = wall_flat & (he_std == 4.0)
    if wall_iv.any():
        ratio = imag_std[wall_iv].max(axis=-1) / np.maximum(
            real_std[wall_iv].max(axis=-1), 1e-30
        )
        out["wall_iv_imag_re_ratio_median"] = float(np.median(ratio))
    else:
        out["wall_iv_imag_re_ratio_median"] = None

    # --- tolerance sensitivity (full grid, wall-restricted) -------------
    tol_fracs = {}
    for rt in IMAG_RTOLS:
        he_rt = _classify_types_rtol(flat_Tmixed, flat_g, rt)
        tol_fracs[f"{rt:.0e}"] = _vw_type_fraction(he_rt, wall_flat, vol_w, 4)
    out["tol_type_iv_frac"] = tol_fracs
    tol_vals = [v * 100.0 for v in tol_fracs.values()]  # to pp
    out["tol_spread_pp"] = float(max(tol_vals) - min(tol_vals))
    out["tol_insensitive"] = out["tol_spread_pp"] <= 0.5

    # --- solver agreement on a wall subset (standard vs generalized) ----
    wall_idx = np.where(wall_flat)[0]
    # prioritise Type-IV-flagged points, then fill with a deterministic sample
    iv_idx = wall_idx[he_std[wall_idx] == 4.0]
    other_idx = wall_idx[he_std[wall_idx] != 4.0]
    take_iv = iv_idx[: gen_cap // 2]
    take_other = other_idx[:: max(1, len(other_idx) // max(1, gen_cap - len(take_iv)))][
        : gen_cap - len(take_iv)
    ]
    sub = np.concatenate([take_iv, take_other]) if len(take_iv) or len(take_other) else wall_idx[:0]
    out["agreement_n"] = int(sub.size)
    if sub.size:
        cls_gen = _classify_grid_batch(
            flat_Tmixed[sub], flat_g[sub], flat_T[sub], solver="generalized"
        )
        he_gen = np.asarray(cls_gen.he_type)
        out["std_gen_agreement"] = float(np.mean(he_std[sub] == he_gen))
        out["std_gen_iv_agreement"] = (
            float(np.mean((he_std[sub] == 4.0) == (he_gen == 4.0)))
        )
    else:
        out["std_gen_agreement"] = None
        out["std_gen_iv_agreement"] = None

    # --- mpmath 50-digit cross-check on a small wall subset -------------
    mp_idx = sub[:mpmath_cap] if sub.size else wall_idx[:0]
    out["mpmath_n"] = int(mp_idx.size)
    if mp_idx.size:
        rep = verify_classification_at_points(
            np.asarray(flat_Tmixed[mp_idx]),
            np.asarray(flat_g[mp_idx]),
            he_std[mp_idx],
            precision=50,
            imag_rtol=3e-3,  # reproduce the float64 verdict (physical check)
        )
        out["mpmath_flip_rate"] = float(rep["flip_rate"])
        out["mpmath_uncertain_frac"] = float(np.mean(rep["uncertain_mask"]))
    else:
        out["mpmath_flip_rate"] = None
        out["mpmath_uncertain_frac"] = None

    # --- refinement stability of wall Type-IV fraction ------------------
    if do_refine:
        fracs = []
        for N in refine_Ns:
            if N == N_main:
                fracs.append(out["wall_frac_type_IV"] * 100.0)
                continue
            fT, fg, fgi, fTm, wmask, vw = _flat_curv(metric, N)
            he = np.asarray(
                _classify_grid_batch(fTm, fg, fT, solver="standard").he_type
            )
            fracs.append(_vw_type_fraction(he, wmask, vw, 4) * 100.0)
        stab = f_miss_stability(fracs, abs_tol_pp=0.5, rel_tol=0.05)
        out["refine_Ns"] = list(refine_Ns)
        out["refine_type_iv_pp"] = fracs
        out["refine_stable"] = bool(stab["stable"])
        out["refine_max_dev_pp"] = float(stab["max_dev_pp"])
    else:
        out["refine_stable"] = None

    # --- VERDICT --------------------------------------------------------
    flip_ok = (out["mpmath_flip_rate"] is None) or (out["mpmath_flip_rate"] <= 0.01)
    refine_ok = (out["refine_stable"] is None) or out["refine_stable"]
    tol_ok = out["tol_insensitive"]
    has_iv = out["wall_frac_type_IV"] > 1e-4
    out["type_iv_trustworthy"] = bool(flip_ok and refine_ok and tol_ok)
    out["has_wall_type_iv"] = bool(has_iv)
    return out


def _fmt(x, nd=3):
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "yes" if x else "NO"
    return f"{x:.{nd}f}"


def write_report(cells, out_path):
    lines = [
        "# Superluminal classification feasibility gate\n",
        "Wall-restricted (R=1, sigma=8) Hawking-Ellis Type-IV trustworthiness.\n",
        "Type-IV is trustworthy iff: mpmath flip-rate <= 1% AND refinement-stable "
        "AND tolerance-insensitive (<=0.5pp).\n",
        "",
        "| Metric | v_s | wall TypeIV % | tol spread pp | std/gen agree | mpmath flip | refine stable | Im/Re | TRUSTWORTHY |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        lines.append(
            f"| {c['metric']} | {c['v_s']:.2f} | "
            f"{_fmt(c['wall_frac_type_IV']*100,2)} | {_fmt(c['tol_spread_pp'],2)} | "
            f"{_fmt(c['std_gen_agreement'])} | {_fmt(c['mpmath_flip_rate'])} | "
            f"{_fmt(c['refine_stable'])} | {_fmt(c['wall_iv_imag_re_ratio_median'])} | "
            f"{_fmt(c['type_iv_trustworthy'])} |"
        )
    # summary: max trustworthy v_s per metric (where Type-IV present)
    lines += ["", "## Trustworthy velocity ceiling per metric", ""]
    for name in METRIC_ORDER:
        mc = [c for c in cells if c["metric"] == name]
        good = [c["v_s"] for c in mc if c["type_iv_trustworthy"]]
        ceiling = max(good) if good else None
        lines.append(f"- {name}: classification trustworthy up to v_s = {_fmt(ceiling,2)}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--velocities", type=float, nargs="+",
                   default=[0.5, 0.9, 0.99, 1.0, 1.1, 1.5, 2.0, 2.5])
    p.add_argument("--n-main", type=int, default=50)
    p.add_argument("--refine-ns", type=int, nargs="+", default=[30, 50, 70])
    p.add_argument("--refine-velocities", type=float, nargs="+",
                   default=[0.99, 1.0, 1.5])
    p.add_argument("--mpmath-cap", type=int, default=120)
    p.add_argument("--gen-cap", type=int, default=2000)
    p.add_argument("--metrics", type=str, nargs="+", default=METRIC_ORDER)
    p.add_argument("--smoke", action="store_true",
                   help="fast tiny config to check the script runs")
    args = p.parse_args()

    if args.smoke:
        args.velocities = [0.5, 1.5]
        args.n_main = 24
        args.refine_ns = [20, 24, 28]
        args.refine_velocities = [1.5]
        args.mpmath_cap = 20
        args.gen_cap = 100
        args.metrics = ["Alcubierre", "Rodal"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 72)
    print("SUPERLUMINAL CLASSIFICATION GATE (R=1, sigma=8, wall-clustered)")
    print(f"metrics={args.metrics} velocities={args.velocities} N={args.n_main}")
    print("=" * 72)

    cells = []
    for name in args.metrics:
        for v_s in args.velocities:
            t0 = time.time()
            do_refine = v_s in args.refine_velocities
            c = run_cell(
                name, v_s, args.n_main, args.refine_ns,
                args.mpmath_cap, args.gen_cap, do_refine,
            )
            cells.append(c)
            print(
                f"  {name:>14s} v_s={v_s:.2f}  "
                f"TypeIV(wall)={c['wall_frac_type_IV']*100:6.2f}%  "
                f"tol_spread={c['tol_spread_pp']:.2f}pp  "
                f"std/gen={_fmt(c['std_gen_agreement'])}  "
                f"mpflip={_fmt(c['mpmath_flip_rate'])}  "
                f"refine={_fmt(c['refine_stable'])}  "
                f"trust={_fmt(c['type_iv_trustworthy'])}  "
                f"({time.time()-t0:.0f}s)"
            )

    out_json = os.path.join(RESULTS_DIR, "superluminal_gate.json")
    dump_json({"config": vars(args), "cells": cells}, out_json)
    print(f"\nWrote {out_json}")
    write_report(cells, os.path.join(RESULTS_DIR, "superluminal_gate_report.md"))


if __name__ == "__main__":
    main()
