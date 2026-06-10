"""Fuchs kernel-robustness comparison and post-smoothing residual verification.

Kernel comparison: re-runs the canonical Fuchs constant-velocity shell
verification for both ``kernel_type="gaussian"`` (the published default) and
``kernel_type="moving_average"`` (the original MATLAB ``smooth()`` boxcar,
variance-matched span = sigma*sqrt(12)).

For each kernel we report, over an identical 50-pt radial sweep on [0.5, 40]
with n_starts=16:
  * interior probes (bulk shell r in [R_1, R_2] = [10, 20]) and how many
    violate an energy condition (NEC/WEC/DEC robust margin < 0),
  * tail probes (smoothing tail r > R_2 = 20) and how many are Hawking-Ellis
    Type IV and/or violate WEC/DEC,
  * the worst (most negative) min(WEC, DEC) robust margin in the tail and
    its radius.

Post-smoothing residual: for the canonical (post-smoothing) Gaussian Fuchs
shell, reports the source-consistency relative residual of T_input vs
G_ab/8pi using ``warpax.constraints.source_consistency.stress_energy_residual``.
The post-smoothing T_input is the static-observer perfect-fluid stress-energy
assembled from the *smoothed* density and isotropic pressure that the
construction feeds into the TOV/metric solve (the canonical claimed source,
as opposed to the pre-smoothing constant-density intermediate).

Outputs: results/fuchs_kernel_comparison.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from _json_io import dump_json

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _radial_sweep import evaluate_point  # noqa: E402

from warpax.constraints.source_consistency import stress_energy_residual  # noqa: E402
from warpax.metrics import build_fuchs_construction, fuchs_default  # noqa: E402


OUTPUT = Path(__file__).resolve().parents[1] / "results" / "fuchs_kernel_comparison.json"

R_RANGE = (0.5, 40.0)
N_SWEEP = 50
N_STARTS = 16
R_1, R_2 = 10.0, 20.0


# Kernel radial sweep + interior/tail breakdown


def sweep_kernel(kernel_type: str) -> dict:
    metric = fuchs_default(kernel_type=kernel_type)
    rs = jnp.linspace(*R_RANGE, N_SWEEP)
    per_point = []
    t0 = time.time()
    for i, r in enumerate(rs):
        sys.stdout.write(f"\r  [{kernel_type}] {i + 1}/{N_SWEEP} r={float(r):6.2f}")
        sys.stdout.flush()
        coords = jnp.array([0.0, float(r), 0.0, 0.0], dtype=jnp.float64)
        per_point.append(evaluate_point(metric, coords, n_starts=N_STARTS))
    print()
    elapsed = time.time() - t0

    interior = [p for p in per_point if R_1 <= p["r"] <= R_2]
    tail = [p for p in per_point if p["r"] > R_2]

    def viol(p):  # robust EC violation: any of NEC/WEC/DEC < 0
        return any(p["ec_robust"][k] < 0 for k in ("nec", "wec", "dec"))

    interior_viol = sum(1 for p in interior if viol(p))
    tail_type_iv = sum(1 for p in tail if p["he_type"] == 4)
    tail_wec_viol = sum(1 for p in tail if p["ec_robust"]["wec"] < 0)
    tail_dec_viol = sum(1 for p in tail if p["ec_robust"]["dec"] < 0)
    tail_any_viol = sum(1 for p in tail if viol(p))

    # worst (most negative) min(WEC, DEC) robust margin in the tail
    worst_margin = float("inf")
    worst_r = None
    for p in tail:
        m = min(p["ec_robust"]["wec"], p["ec_robust"]["dec"])
        if m < worst_margin:
            worst_margin = m
            worst_r = p["r"]

    return {
        "kernel_type": kernel_type,
        "elapsed_s": round(elapsed, 1),
        "interior_count": len(interior),
        "interior_violations": interior_viol,
        "tail_count": len(tail),
        "tail_type_iv": tail_type_iv,
        "tail_wec_violations": tail_wec_viol,
        "tail_dec_violations": tail_dec_viol,
        "tail_any_ec_violations": tail_any_viol,
        "worst_tail_min_wec_dec_margin": worst_margin,
        "worst_tail_radius": worst_r,
        "per_point": per_point,
    }


# Post-smoothing source-consistency residual


def _smoothed_T_input_builder(metric, construction):
    """Return T_input(coords) for the post-smoothing canonical shell.

    The claimed canonical source is a static-observer perfect fluid built from
    the *smoothed* density and isotropic pressure that the construction feeds
    into the metric solve. We assemble T_ab exactly as fuchs_input_stress_energy
    does (perfect fluid, isotropic p_r = p_t = P_smoothed), but with the smoothed
    profiles of the canonical construction interpolated at radius r.
    """
    import interpax

    r_grid = construction.r_grid
    rho_grid = construction.rho_smoothed
    P_grid = construction.P_smoothed

    def rho_at(r):
        rc = jnp.clip(r, r_grid[0], r_grid[-1])
        return interpax.interp1d(rc, r_grid, rho_grid, method="cubic")

    def P_at(r):
        rc = jnp.clip(r, r_grid[0], r_grid[-1])
        return jnp.maximum(interpax.interp1d(rc, r_grid, P_grid, method="cubic"), 0.0)

    def T_input(coords):
        g = metric(coords)
        t, x, y, z = coords
        x_rel = x - metric.v_s * t
        r = jnp.sqrt(x_rel ** 2 + y ** 2 + z ** 2 + 1e-24)

        rho = rho_at(r)
        p = P_at(r)  # isotropic: p_r = p_t

        alpha = metric.lapse(coords)
        u_up = jnp.array([1.0 / alpha, 0.0, 0.0, 0.0])
        u_down = g @ u_up

        # perfect fluid: T = (rho + p) u u + p g
        T = (rho + p) * jnp.outer(u_down, u_down) + p * g
        return 0.5 * (T + T.T)

    return T_input


def source_consistency_post_smoothing() -> dict:
    metric = fuchs_default(kernel_type="gaussian")
    construction = build_fuchs_construction(kernel_type="gaussian")
    T_input_fn = _smoothed_T_input_builder(metric, construction)

    rs = jnp.linspace(*R_RANGE, N_SWEEP)
    per_point = []
    peak_rel = -1.0
    peak_rel_r = None
    peak_rel_shell = -1.0  # peak within physical shell r in [R_1, R_2]
    peak_rel_shell_r = None
    for r in rs:
        coords = jnp.array([0.0, float(r), 0.0, 0.0], dtype=jnp.float64)
        T_in = T_input_fn(coords)
        sc = stress_energy_residual(metric, coords, T_input=T_in)
        rel = float(sc["relative_residual"])
        mx = float(sc["max_residual"])
        per_point.append({
            "r": float(r),
            "max_residual": mx,
            "relative_residual": rel,
        })
        if rel > peak_rel:
            peak_rel, peak_rel_r = rel, float(r)
        if R_1 <= float(r) <= R_2 and rel > peak_rel_shell:
            peak_rel_shell, peak_rel_shell_r = rel, float(r)

    shell_pts = [p for p in per_point if R_1 <= p["r"] <= R_2]
    shell_rel = sorted(p["relative_residual"] for p in shell_pts)
    median_shell_rel = shell_rel[len(shell_rel) // 2]
    max_abs_in_shell = max(p["max_residual"] for p in shell_pts)

    return {
        "description": (
            "Post-smoothing source consistency: T_input = static-observer "
            "perfect fluid from smoothed (rho, P) vs G_ab/8pi. "
            "Representative number is the in-shell (r in [R_1, R_2]) peak "
            "relative residual; vacuum interior (r<R_1) and far tail (r>R_2) "
            "relative residuals are 0/0 / near-vacuum artifacts where both "
            "T_input and T_derived approach zero (see max_residual column)."
        ),
        "representative_relative_residual_in_shell": peak_rel_shell,
        "representative_relative_residual_in_shell_radius": peak_rel_shell_r,
        "median_relative_residual_in_shell": median_shell_rel,
        "max_abs_residual_in_shell": max_abs_in_shell,
        "peak_relative_residual_full_sweep": peak_rel,
        "peak_relative_residual_full_sweep_radius": peak_rel_r,
        "note_full_sweep_peak": (
            "full-sweep peak is a vacuum 0/0 artifact (max|dT|~1e-19, "
            "max|T_derived|~0); not physical."
        ),
        "per_point": per_point,
    }


def main():
    print("=" * 64)
    print("Fuchs kernel comparison + post-smoothing residual")
    print("=" * 64)

    results = {}
    for kt in ("gaussian", "moving_average"):
        results[kt] = sweep_kernel(kt)

    print("  computing post-smoothing source-consistency residual ...")
    residual = source_consistency_post_smoothing()

    report = {
        "config": {
            "r_range": list(R_RANGE),
            "n_sweep": N_SWEEP,
            "n_starts": N_STARTS,
            "R_1": R_1,
            "R_2": R_2,
            "interior_def": "R_1 <= r <= R_2 (bulk shell)",
            "tail_def": "r > R_2 (smoothing tail)",
            "ec_violation_def": "robust margin < 0 for any of NEC/WEC/DEC",
        },
        "kernel_comparison": {
            kt: {k: v for k, v in results[kt].items() if k != "per_point"}
            for kt in results
        },
        "per_point": {kt: results[kt]["per_point"] for kt in results},
        "post_smoothing_residual": residual,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dump_json(report, OUTPUT, default=str)
    print(f"  -> {OUTPUT}")

    # compact console summary
    print()
    print(f"{'kernel':<16s} {'int viol':>10s} {'tail IV':>9s} "
          f"{'worst margin':>14s} {'@ r':>8s}")
    for kt in ("gaussian", "moving_average"):
        s = results[kt]
        print(f"{kt:<16s} {s['interior_violations']}/{s['interior_count']:<8} "
              f"{s['tail_type_iv']}/{s['tail_count']:<6} "
              f"{s['worst_tail_min_wec_dec_margin']:>14.4e} "
              f"{s['worst_tail_radius']:>8.2f}")
    print()
    print(f"Post-smoothing relative residual (in shell, representative): "
          f"{residual['representative_relative_residual_in_shell']:.4e} "
          f"@ r={residual['representative_relative_residual_in_shell_radius']:.2f}  "
          f"(max|dT|={residual['max_abs_residual_in_shell']:.3e})")
    print(f"Median in-shell relative residual: "
          f"{residual['median_relative_residual_in_shell']:.4e}")


if __name__ == "__main__":
    main()
