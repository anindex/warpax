"""Per-diagnostic convergence of the wall diagnostics, wall-resolved.

For each matched-parameter drive (R=1, sigma=8, v_s=0.5) on the canonical
wall-resolved graded grid (box +-3, clustering a*=2.0; see scripts/_benchmark_grid.py)
at three resolutions N = [80, 100, 120] -> 4.5 / 5.6 / 6.7 cells across the 10-90%
wall on a radial crossing (every level clears the four-cell criterion), report a
certificate for each wall diagnostic:

- Wall Type-IV FRACTION: a non-smooth, thresholded volume fraction (integral of
  a Heaviside indicator). It is NOT a valid Richardson target; we report its
  grid stability spread across the wall-resolved ladder.
- Wall max|Im lambda|, min(rho+p_i), min(rho-|p_i|): grid-sampled EXTREMA of
  smooth autodiff-exact fields. A grid-sampled extremum undershoots the true
  continuous extremum by an O(dx^2) grid-alignment gap that does not admit a
  clean Richardson order (aliasing). We therefore report the EXACT continuous
  extremum, obtained by continuous local refinement of the exact tensor
  (warpax.analysis.extrema.refine_extremum), which is resolution-independent.
  The ladder values are shown only to exhibit the sampled values approaching it.
- Wall vorticity R_omega, WEC miss rate, DEC miss rate: bounded ratios in [0,1].
  Like the Type-IV fraction they admit no Richardson order, so we report their
  grid-stability spread across the wall-resolved ladder.

No Richardson order is asserted for any quantity: fractions are non-smooth,
extrema are exactly polished.

Outputs
- results/diagnostic_convergence.json
- ../warpax_arxiv/tables/diagnostic_convergence.tex
"""
from __future__ import annotations

import os

from _json_io import dump_json
from _benchmark_grid import benchmark_grid, wall_cells, N_LADDER, CLUSTER_A, BOX

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from types import SimpleNamespace

from warpax.benchmarks import AlcubierreMetric
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.energy_conditions import certify_grid_frame_free, type_fractions
from warpax.energy_conditions.filtering import (
    compute_wall_restricted_stats,
    shape_function_mask,
)
from warpax.energy_conditions.verifier import _eulerian_ec_point
from warpax.analysis.convergence import f_miss_stability
from warpax.analysis.extrema import refine_extremum
from warpax.analysis.shift_kinematics import compute_shift_kinematics_grid

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

V_S = 0.5


def _metrics():
    return {
        "Alcubierre": AlcubierreMetric(v_s=V_S, R=1.0, sigma=8.0),
        "Natário": NatarioMetric(v_s=V_S, R=1.0, sigma=8.0),
        "Van den Broeck": VanDenBroeckMetric(
            v_s=V_S, R=1.0, sigma=8.0, R_tilde=1.0, alpha_vdb=0.5, sigma_B=8.0
        ),
        "Rodal": RodalMetric(v_s=V_S, R=1.0, sigma=8.0),
    }


def _field_max_imag(curv):
    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)
    return np.max(np.abs(np.asarray(ff.eigenvalues_imag)), axis=-1)


def _field_typeI(curv, which):
    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)
    he = np.asarray(ff.he_types)
    margins = np.asarray(ff.nec_margins if which == "nec" else ff.dec_margins)
    return np.where(he == 1.0, margins, np.inf)  # min over Type-I only


def _grid_diagnostics(metric, N):
    """Grid-sampled wall diagnostics + argmin/argmax seeds at resolution N."""
    grid = benchmark_grid(metric, N)
    curv = evaluate_curvature_grid(metric, grid, batch_size=4096)
    ff = certify_grid_frame_free(curv.stress_energy, curv.metric, curv.metric_inv)
    coords = build_coord_batch(grid, t=0.0)
    mask = shape_function_mask(metric, coords, grid.shape)
    vol_w = grid.volume_weights_array
    fracs = type_fractions(ff, mask=mask, volume_weights=vol_w)
    m = np.asarray(mask).ravel().astype(bool)
    axes = [np.asarray(grid.axes[a]) for a in range(3)]

    imag_pt = np.max(np.abs(np.asarray(ff.eigenvalues_imag).reshape(-1, 4)), axis=-1)
    imag_w = np.where(m, imag_pt, -np.inf)
    ki = int(np.argmax(imag_w))
    max_imag = float(imag_pt[ki]) if m.any() else 0.0
    seed_imag = _flat_to_xyz(ki, grid.shape, axes)

    he = np.asarray(ff.he_types).ravel()
    nec = np.asarray(ff.nec_margins).ravel()
    dec = np.asarray(ff.dec_margins).ravel()
    typeI_wall = m & (he == 1.0)
    nec_min, seed_nec = _masked_min(nec, typeI_wall, grid.shape, axes)
    dec_min, seed_dec = _masked_min(dec, typeI_wall, grid.shape, axes)

    # Wall vorticity fraction R_omega (proper-volume-weighted ratio of the
    # vorticity scalar to the full shift-gradient magnitude). Same definition as
    # run_shift_vorticity.wall_decomposition; reported across the ladder to show
    # it is grid-stable.
    theta, sigma_sq, omega_sq = compute_shift_kinematics_grid(
        metric, grid, t=0.0, batch_size=512)
    wv = np.where(m, np.asarray(vol_w).reshape(-1), 0.0)
    wsum = float(wv.sum())
    exp = float((wv * (np.asarray(theta).reshape(-1) ** 2 / 3.0)).sum() / wsum)
    she = float((wv * np.asarray(sigma_sq).reshape(-1)).sum() / wsum)
    vor = float((wv * np.asarray(omega_sq).reshape(-1)).sum() / wsum)
    r_omega = vor / (exp + she + vor + 1e-300)

    # Wall-restricted conditional WEC/DEC miss rates: the volume-weighted
    # fraction of wall violations that a single-frame Eulerian reading misses
    # relative to the eigenvalue-robust decision. No optimizer: at a Type-I point
    # the robust margin equals the eigenvalue inequality, and Type-IV is flagged
    # by the frame-free certifier, so no observer search is needed. The Eulerian
    # margins are the single-frame contractions from _eulerian_ec_point.
    eul = jax.vmap(_eulerian_ec_point)(
        jnp.reshape(curv.stress_energy, (-1, 4, 4)),
        jnp.reshape(curv.metric, (-1, 4, 4)),
        jnp.reshape(curv.metric_inv, (-1, 4, 4)),
    )
    ec = SimpleNamespace(
        he_types=ff.he_types, nec_margins=ff.nec_margins,
        wec_margins=ff.wec_margins, sec_margins=ff.sec_margins,
        dec_margins=ff.dec_margins, eigenvalues=ff.eigenvalues)
    eul_m = {c: jnp.reshape(eul[c], grid.shape) for c in ("nec", "wec", "sec", "dec")}
    wr = compute_wall_restricted_stats(
        ec, mask, eulerian_margins=eul_m, volume_weights=vol_w)

    return {
        "type_iv_frac": fracs["frac_type_iv"],
        "max_imag": max_imag, "seed_imag": seed_imag,
        "nec_min": nec_min, "seed_nec": seed_nec,
        "dec_min": dec_min, "seed_dec": seed_dec,
        "r_omega": r_omega,
        "wec_miss": wr.wec_miss_rate, "dec_miss": wr.dec_miss_rate,
    }


def _flat_to_xyz(k, shape, axes):
    i0, i1, i2 = np.unravel_index(k, shape)
    return [float(axes[0][i0]), float(axes[1][i1]), float(axes[2][i2])]


def _masked_min(field, sel, shape, axes):
    if not sel.any():
        return None, None
    masked = np.where(sel, field, np.inf)
    k = int(np.argmin(masked))
    return float(masked[k]), _flat_to_xyz(k, shape, axes)


def _polish(metric, seed, field_fn, mode):
    if seed is None:
        return None
    res = refine_extremum(metric, seed, field_fn, mode=mode,
                          half_width=0.15, n=9, levels=7)
    return res["value"]


def _f(x, nd=3):
    return f"{x:.{nd}f}" if (x is not None and np.isfinite(x)) else "--"


def write_table(results, wall_info, out_path):
    ns = N_LADDER
    hdr = " & ".join(f"$N{{=}}{n}$" for n in ns)
    lines = [
        r"\begin{tabular}{@{}l l ccc c@{}}",
        r"  \toprule",
        rf"  Metric & Diagnostic & {hdr} & Certification \\",
        r"  \midrule",
    ]
    labels = [
        ("type_iv_frac", "Wall Type-IV frac.", "fraction"),
        ("max_imag", r"Wall $\max|\mathrm{Im}\,\lambda|$", "extremum"),
        ("nec_min", r"Wall $\min(\rho+p_i)$", "extremum"),
        ("dec_min", r"Wall $\min(\rho-|p_i|)$", "extremum"),
        ("r_omega", r"Vorticity $\mathcal{R}_\omega$", "fraction"),
        ("wec_miss", r"Wall WEC miss rate", "fraction"),
        ("dec_miss", r"Wall DEC miss rate", "fraction"),
    ]
    for name, diags in results.items():
        first = True
        for key, label, kind in labels:
            c = diags[key]
            v = c["ladder"]
            if kind == "fraction":
                cert = f"stable ({_f(c['max_dev_pp'], 2)}~pp)"
            else:
                cert = f"polished {_f(c['polished'], 3)}"
            mcol = name if first else ""
            first = False
            cells = " & ".join(_f(x) for x in v)
            lines.append(f"  {mcol} & {label} & {cells} & {cert} \\\\")
        lines.append(r"  \midrule")
    lines[-1] = r"  \bottomrule"
    lines.append(r"\end{tabular}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main():
    print("=" * 72)
    print(f"WALL-RESOLVED PER-DIAGNOSTIC CONVERGENCE (N={N_LADDER}, a*={CLUSTER_A}, box +-{BOX})")
    print("=" * 72)
    results = {}
    wall_info = {}
    for name, metric in _metrics().items():
        print(f"\n{name}:")
        wc = {N: wall_cells(metric, N) for N in N_LADDER}
        wall_info[name] = {str(N): {"cells": wc[N][0], "dx_wall": wc[N][1]} for N in N_LADDER}
        print("  wall cells: " + ", ".join(f"N={N}:{wc[N][0]}c/dx={wc[N][1]:.3f}" for N in N_LADDER))
        series = {k: [] for k in ("type_iv_frac", "max_imag", "nec_min", "dec_min",
                                   "r_omega", "wec_miss", "dec_miss")}
        # Track the most extreme SAMPLE across the whole ladder (not just the
        # finest grid): a coarser grid can catch a deeper/higher sample the finest
        # grid misses by alignment, and the polish must be seeded from that.
        best = {"max_imag": (-np.inf, None), "nec_min": (np.inf, None),
                "dec_min": (np.inf, None)}
        for N in N_LADDER:
            d = _grid_diagnostics(metric, N)
            for k in series:
                series[k].append(d[k])
            if d["max_imag"] > best["max_imag"][0]:
                best["max_imag"] = (d["max_imag"], d["seed_imag"])
            if d["nec_min"] is not None and d["nec_min"] < best["nec_min"][0]:
                best["nec_min"] = (d["nec_min"], d["seed_nec"])
            if d["dec_min"] is not None and d["dec_min"] < best["dec_min"][0]:
                best["dec_min"] = (d["dec_min"], d["seed_dec"])
            print(f"  N={N}: TypeIV={d['type_iv_frac']:.3f} max|Im|={d['max_imag']:.4f} "
                  f"NECmin={_f(d['nec_min'])} DECmin={_f(d['dec_min'])}")
        # Continuum polish, seeded from the ladder's most extreme sample and
        # guaranteed at least as extreme as every sample (a grid sample is a valid
        # field value, so the continuum extremum cannot be less extreme than it).
        pi = _polish(metric, best["max_imag"][1], _field_max_imag, "max")
        pol_imag = max(best["max_imag"][0], pi) if pi is not None else best["max_imag"][0]
        pn = _polish(metric, best["nec_min"][1], lambda c: _field_typeI(c, "nec"), "min")
        pol_nec = (min(best["nec_min"][0], pn) if pn is not None
                   else (best["nec_min"][0] if np.isfinite(best["nec_min"][0]) else None))
        pd = _polish(metric, best["dec_min"][1], lambda c: _field_typeI(c, "dec"), "min")
        pol_dec = (min(best["dec_min"][0], pd) if pd is not None
                   else (best["dec_min"][0] if np.isfinite(best["dec_min"][0]) else None))
        print(f"  polished: max|Im|={_f(pol_imag,4)} NECmin={_f(pol_nec)} DECmin={_f(pol_dec)}")

        st = f_miss_stability([v * 100 for v in series["type_iv_frac"]])
        stw = f_miss_stability([v * 100 for v in series["r_omega"]])
        # Miss rates are ratios in [0,1] (or None if the wall has no violations of
        # that condition); certify their grid stability like the other fractions.
        stwec = f_miss_stability([(v or 0.0) * 100 for v in series["wec_miss"]])
        stdec = f_miss_stability([(v or 0.0) * 100 for v in series["dec_miss"]])
        print(f"  WEC miss={_f(series['wec_miss'][-1])} DEC miss={_f(series['dec_miss'][-1])} "
              f"(spread {_f(stwec['max_dev_pp'],2)}/{_f(stdec['max_dev_pp'],2)} pp)")
        results[name] = {
            "type_iv_frac": {"ladder": series["type_iv_frac"],
                             "max_dev_pp": st["max_dev_pp"], "stable": bool(st["stable"])},
            "max_imag": {"ladder": series["max_imag"], "polished": pol_imag},
            "nec_min": {"ladder": series["nec_min"], "polished": pol_nec},
            "dec_min": {"ladder": series["dec_min"], "polished": pol_dec},
            "r_omega": {"ladder": series["r_omega"],
                        "max_dev_pp": stw["max_dev_pp"], "stable": bool(stw["stable"])},
            "wec_miss": {"ladder": series["wec_miss"],
                         "max_dev_pp": stwec["max_dev_pp"], "stable": bool(stwec["stable"])},
            "dec_miss": {"ladder": series["dec_miss"],
                         "max_dev_pp": stdec["max_dev_pp"], "stable": bool(stdec["stable"])},
        }
    dump_json({"ladder_N": N_LADDER, "cluster_a": CLUSTER_A, "box": BOX, "v_s": V_S,
               "wall_info": wall_info, "results": results},
              os.path.join(RESULTS_DIR, "diagnostic_convergence.json"))
    write_table(results, wall_info, os.path.join(TABLES_DIR, "diagnostic_convergence.tex"))


if __name__ == "__main__":
    main()
