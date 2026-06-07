"""Shift vorticity as the geometric control of the Hawking-Ellis type.

For each retained drive we compute the wall-restricted, proper-volume-weighted
irreducible decomposition of the ADM shift gradient (expansion, shear,
vorticity) and the dimensionless vorticity fraction R_omega, then pair it with
the cached wall Type-IV fractions from ``results/velocity_sweep.json``.

The result: the irrotational Rodal drive (R_omega = 0) is the unique globally
Type-I geometry, the zero-expansion Natario drive (theta_beta = 0) still carries
vorticity and is Type-IV-walled, and Alcubierre / Van den Broeck likewise have
rotational shifts and Type-IV walls. R_omega is independent of v_s because the
shift scales linearly with the warp speed, so it is a per-drive fingerprint.

Outputs
-------
- results/shift_vorticity.json
- ../warpax_arxiv/tables/shift_vorticity.tex
- ../warpax_arxiv/figures/shift_vorticity.pdf
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.analysis.shift_kinematics import compute_shift_kinematics_grid
from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions.filtering import shape_function_mask
from warpax.geometry.grid import build_coord_batch
from warpax.grids import wall_clustered
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric
from warpax.visualization.shift_vorticity_plots import plot_shift_vorticity

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")
# Figures are generated into the codebase, then copied to the paper folder
# by reproduce_all.sh (sync_figures_to_paper).
FIG_DIR = os.path.join(HERE, "..", "figures")

BOUNDS = [(-3, 3)] * 3
F_LOW, F_HIGH = 0.1, 0.9
METRICS = {
    "Alcubierre": (AlcubierreMetric, {}),
    "Natário": (NatarioMetric, {}),
    "Van den Broeck": (VanDenBroeckMetric,
                       {"R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0}),
    "Rodal": (RodalMetric, {}),
}
ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]


def _instantiate(name, v_s):
    cls, extra = METRICS[name]
    return cls(v_s=v_s, R=1.0, sigma=8.0, **extra)


def wall_decomposition(name, v_s, N):
    """Wall proper-volume-weighted (<theta^2/3>, <sigma^2>, <omega^2>)."""
    metric = _instantiate(name, v_s)
    shape = (N, N, N)
    grid = wall_clustered(metric, BOUNDS, shape, a=1.2)
    theta, sigma_sq, omega_sq = compute_shift_kinematics_grid(
        metric, grid, t=0.0, batch_size=512
    )
    coords = build_coord_batch(grid, t=0.0)
    mask = np.asarray(jnp.reshape(
        shape_function_mask(metric, coords, shape, f_low=F_LOW, f_high=F_HIGH),
        (-1,))).astype(bool)
    w = np.asarray(grid.volume_weights_array).reshape(-1)
    w = np.where(mask, w, 0.0)
    wsum = float(w.sum())
    if wsum == 0.0:
        raise ValueError(
            f"Wall mask is empty for metric '{name}' at v_s={v_s} (N={N}); "
            "check F_LOW/F_HIGH and the grid extent."
        )
    exp = float((w * (np.asarray(theta).reshape(-1) ** 2 / 3.0)).sum() / wsum)
    she = float((w * np.asarray(sigma_sq).reshape(-1)).sum() / wsum)
    vor = float((w * np.asarray(omega_sq).reshape(-1)).sum() / wsum)
    return exp, she, vor


def write_table(fingerprint, type_iv_range, out_path):
    lines = [
        r"\begin{tabular}{@{}l ccc c@{}}",
        r"  \toprule",
        r"  & \multicolumn{3}{c}{Shift-gradient fraction} & Wall Type~IV \\",
        r"  \cmidrule(lr){2-4}",
        r"  Metric & Expansion & Shear & Vorticity $\mathcal{R}_\omega$ "
        r"& range (\%) \\",
        r"  \midrule",
    ]
    for m in ORDER:
        f = fingerprint[m]
        lo, hi = type_iv_range[m]
        lines.append(
            f"  {m} & {f['expansion']:.3f} & {f['shear']:.3f} & "
            f"{f['vorticity']:.3f} & {lo:.0f}--{hi:.0f} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fingerprint-velocities", type=float, nargs="+",
                   default=[0.1, 0.5, 0.9, 1.5, 2.5])
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.fingerprint_velocities = [0.5]
        args.N = 24

    # Cached Type-IV wall fractions (the expensive eigenstructure).
    with open(os.path.join(RESULTS_DIR, "velocity_sweep.json")) as f:
        sweep = json.load(f)
    type_iv = {}
    for r in sweep["rows"]:
        type_iv.setdefault(r["metric"], []).append(
            (r["v_s"], r["wall_frac_type_iv"] * 100.0))
    for m in type_iv:
        type_iv[m].sort()
    missing = [m for m in ORDER if m not in type_iv]
    if missing:
        raise KeyError(
            f"velocity_sweep.json is missing Type-IV data for {missing}; "
            "rerun run_velocity_sweep.py for the full metric set first."
        )

    print("=" * 72)
    print(f"SHIFT VORTICITY (R=1, sigma=8, N={args.N}, wall-clustered)")
    print("=" * 72)

    # Per-(metric, v_s) decomposition; confirm R_omega is v_s-independent.
    # R_omega here is the ratio of wall-averaged scalars,
    # <omega^2> / (<theta^2/3> + <sigma^2> + <omega^2>), the wall fingerprint.
    # This is the volume aggregate of the pointwise analysis.rotationality(),
    # and equals the reported vorticity fraction by construction.
    raw = {m: [] for m in ORDER}
    for name in ORDER:
        for v_s in args.fingerprint_velocities:
            exp, she, vor = wall_decomposition(name, v_s, args.N)
            tot = exp + she + vor + 1e-300
            r_omega_avg = vor / tot
            raw[name].append({"v_s": v_s, "expansion": exp / tot,
                              "shear": she / tot, "vorticity": vor / tot,
                              "rotationality": r_omega_avg})
            print(f"  {name:>15} v_s={v_s:.2f}  "
                  f"exp={exp/tot:.3f} shear={she/tot:.3f} vort={vor/tot:.3f}  "
                  f"R_omega={r_omega_avg:.4f}")

    # Velocity-averaged fingerprint (R_omega is v_s-independent to numerics).
    fingerprint = {}
    for m in ORDER:
        rows = raw[m]
        fingerprint[m] = {
            k: float(np.mean([r[k] for r in rows]))
            for k in ("expansion", "shear", "vorticity")
        }
    type_iv_range = {m: (min(t for _, t in type_iv[m]),
                         max(t for _, t in type_iv[m])) for m in ORDER}

    # Figure sweep data: pair each metric's mean R_omega with all cached Type-IV.
    fig_sweep = {}
    for m in ORDER:
        r_mean = fingerprint[m]["vorticity"]
        fig_sweep[m] = [(v, r_mean, t) for v, t in type_iv[m]]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "shift_vorticity.json"), "w") as f:
        json.dump({"config": vars(args), "raw": raw, "fingerprint": fingerprint,
                   "type_iv_range": type_iv_range}, f, indent=2)
    print(f"\nWrote {os.path.join(RESULTS_DIR, 'shift_vorticity.json')}")

    if not args.smoke:
        write_table(fingerprint, type_iv_range,
                    os.path.join(TABLES_DIR, "shift_vorticity.tex"))
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR, "shift_vorticity.pdf")
        plot_shift_vorticity(fingerprint, fig_sweep, ORDER, save_path=out)
        print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
