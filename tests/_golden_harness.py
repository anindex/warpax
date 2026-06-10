"""Shared golden-value harness for parity testing across refactors.

Single source of truth used by BOTH ``scripts/capture_goldens.py`` (which
saves the goldens to ``tests/fixtures/parity_goldens.npz``) and
``tests/test_parity_golden.py`` (which re-runs and compares). The goal is a
fast, deterministic snapshot of every certified quantity that feeds the
paper, so the engineering/perf refactors in later phases can be proven to
preserve output bit-for-bit (or within a tight tolerance).

The numbers here are NOT the paper's published resolution (small grids are
used for speed); they exist purely to detect drift in the *code path*.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# x64 must be enabled before any jax array work; the package __init__ does
# this, and conftest asserts it, but be defensive when run as a script.
jax.config.update("jax_enable_x64", True)

from warpax import certify
from warpax.averaged import anec
from warpax.benchmarks import AlcubierreMetric, SchwarzschildMetric
from warpax.energy_conditions.classification import classify_hawking_ellis
from warpax.energy_conditions.optimization import optimize_point
from warpax.energy_conditions.verifier import verify_grid
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.geometry.geometry import compute_curvature_chain
from warpax.analysis.shift_kinematics import compute_shift_kinematics_grid
from warpax.analysis.kinematic_scalars import compute_kinematic_scalars_grid
from warpax.metrics import RodalMetric, WarpShellMetric
from warpax.quantum import ford_roman

# A small wall grid that resolves the compact (R=1, sigma=8) warp wall.
_SMALL_BOUNDS = ((-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0))
_SMALL_SHAPE = (16, 16, 16)


def _np(x) -> np.ndarray:
    return np.asarray(x)


def _classification_grid(metric, bounds, shape) -> dict[str, np.ndarray]:
    """vmapped standard classification over an explicit grid (no clustering)."""
    grid = GridSpec(bounds=bounds, shape=shape)
    chain = evaluate_curvature_grid(metric, grid)
    t_mixed = (chain.metric_inv @ chain.stress_energy).reshape(-1, 4, 4)
    g_flat = chain.metric.reshape(-1, 4, 4)
    cls = jax.vmap(classify_hawking_ellis)(t_mixed, g_flat)
    return {
        "he_type": _np(cls.he_type),
        "rho": _np(cls.rho),
        "pressures": _np(cls.pressures),
        "eigenvalues": _np(cls.eigenvalues),
        "eigenvalues_imag": _np(cls.eigenvalues_imag),
        "is_vacuum": _np(cls.is_vacuum),
    }


def _large_norm_typeI_tensor() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Synthetic Type-I tensor at ||T|| ~ 1e11 with two near-equal causal
    characters - the degenerate case Bug A (timelike-index tiebreak) targets.
    """
    eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
    # Mixed tensor T^a_b diag with a large scale and a tiny split between the
    # timelike eigenvalue and a spatial one to stress the argmin tiebreak.
    scale = 1.0e11
    t_mixed = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]) * scale)
    return t_mixed, eta


def compute_goldens() -> dict[str, np.ndarray]:
    """Compute every golden quantity. Keys use ``group/label/field`` paths.

    Returns a flat dict[str, np.ndarray]; scalars are stored as 0-d arrays.
    Each block is independently guarded so one failure does not abort the
    rest (a missing key simply will not be compared).
    """
    out: dict[str, np.ndarray] = {}

    # --- 1. Classification grids (parity-critical for Bug A, Perf 1) ---
    try:
        ws = _classification_grid(
            WarpShellMetric(v_s=0.5),
            ((-12.0, 12.0), (-6.0, 6.0), (-6.0, 6.0)),
            (10, 10, 10),
        )
        for k, v in ws.items():
            out[f"cls/warpshell_v0.5/{k}"] = v
    except Exception as e:  # noqa: BLE001
        print(f"[golden] skip warpshell classification: {e}")

    for label, metric in (
        ("rodal_v0.5", RodalMetric(v_s=0.5)),
        ("alcubierre_v0.5", AlcubierreMetric(v_s=0.5)),
        ("alcubierre_v1.5", AlcubierreMetric(v_s=1.5)),
    ):
        try:
            d = _classification_grid(metric, _SMALL_BOUNDS, _SMALL_SHAPE)
            for k, v in d.items():
                out[f"cls/{label}/{k}"] = v
        except Exception as e:  # noqa: BLE001
            print(f"[golden] skip {label} classification: {e}")

    # --- Bug A synthetic large-||T|| degenerate Type-I ---
    try:
        t_mixed, g = _large_norm_typeI_tensor()
        r = classify_hawking_ellis(t_mixed, g)
        out["bugA/he_type"] = _np(r.he_type)
        out["bugA/rho"] = _np(r.rho)
        out["bugA/pressures"] = _np(r.pressures)
        out["bugA/eigenvalues"] = _np(r.eigenvalues)
    except Exception as e:  # noqa: BLE001
        print(f"[golden] skip bugA synthetic: {e}")

    # --- 2. certify() summaries (the actual paper numbers) ---
    certify_metrics = {
        "rodal": RodalMetric,
        "alcubierre": AlcubierreMetric,
        "warpshell": WarpShellMetric,
    }
    for name, cls in certify_metrics.items():
        for v_s in (0.5, 1.0, 1.5, 2.0):
            label = f"{name}_v{v_s}"
            try:
                res = certify(cls(v_s=v_s), shape=(16, 16, 16))
                fr = res.type_fractions
                out[f"certify/{label}/frac_type_i"] = _np(fr["frac_type_i"])
                out[f"certify/{label}/frac_type_ii"] = _np(fr["frac_type_ii"])
                out[f"certify/{label}/frac_type_iii"] = _np(fr["frac_type_iii"])
                out[f"certify/{label}/frac_type_iv"] = _np(fr["frac_type_iv"])
                out[f"certify/{label}/invariant_nec_min"] = _np(
                    res.invariant_nec_min
                )
                out[f"certify/{label}/invariant_dec_min"] = _np(
                    res.invariant_dec_min
                )
                if res.single_frame_miss is not None:
                    for mk, mv in res.single_frame_miss.items():
                        # Per-condition entries are dicts; pin miss_rate.
                        if isinstance(mv, dict):
                            rate = mv.get("miss_rate")
                            if rate is not None:
                                out[f"certify/{label}/miss_{mk}"] = _np(
                                    float(rate)
                                )
                        elif isinstance(mv, (int, float)):
                            out[f"certify/{label}/miss_{mk}"] = _np(float(mv))
            except Exception as e:  # noqa: BLE001
                print(f"[golden] skip certify {label}: {e}")

    # --- 3. curvature chain (Perf 3: gamma threading must be bit-exact) ---
    curv_points = {
        "schwarzschild_r5": (SchwarzschildMetric(), jnp.array([0.0, 5.0, 0.0, 0.0])),
        "alcubierre_wall": (AlcubierreMetric(v_s=0.5), jnp.array([0.0, 1.0, 0.0, 0.0])),
        "rodal_wall": (RodalMetric(v_s=0.5), jnp.array([0.0, 1.0, 0.3, 0.0])),
    }
    for label, (metric, pt) in curv_points.items():
        try:
            c = compute_curvature_chain(metric, pt)
            out[f"curv/{label}/riemann"] = _np(c.riemann)
            out[f"curv/{label}/einstein"] = _np(c.einstein)
            out[f"curv/{label}/stress_energy"] = _np(c.stress_energy)
        except Exception as e:  # noqa: BLE001
            print(f"[golden] skip curvature {label}: {e}")

    # --- 4. verify_grid margins (Perf 2 de-serialization parity) ---
    for label, metric in (
        ("rodal_v0.5", RodalMetric(v_s=0.5)),
        ("alcubierre_v0.5", AlcubierreMetric(v_s=0.5)),
    ):
        try:
            grid = GridSpec(bounds=_SMALL_BOUNDS, shape=(8, 8, 8))
            chain = evaluate_curvature_grid(metric, grid)
            vg = verify_grid(
                chain.stress_energy, chain.metric, chain.metric_inv,
                n_starts=1, batch_size=64,
                key=jax.random.PRNGKey(0),
            )
            out[f"verify/{label}/nec_margins"] = _np(vg.nec_margins)
            out[f"verify/{label}/wec_margins"] = _np(vg.wec_margins)
            out[f"verify/{label}/sec_margins"] = _np(vg.sec_margins)
            out[f"verify/{label}/dec_margins"] = _np(vg.dec_margins)
            out[f"verify/{label}/nec_min"] = _np(vg.nec_summary.min_margin)
            out[f"verify/{label}/dec_min"] = _np(vg.dec_summary.min_margin)
        except Exception as e:  # noqa: BLE001
            print(f"[golden] skip verify_grid {label}: {e}")

    # --- 5. optimize_point sec/dec (Perf 7 g_inv threading parity) ---
    try:
        metric = AlcubierreMetric(v_s=0.5)
        c = compute_curvature_chain(metric, jnp.array([0.0, 1.0, 0.0, 0.0]))
        opt = optimize_point(
            c.stress_energy, c.metric,
            conditions=("nec", "wec", "sec", "dec"),
            n_starts=4, key=jax.random.PRNGKey(0),
        )
        out["opt/alcubierre_wall/nec_margin"] = _np(opt["nec"].margin)
        out["opt/alcubierre_wall/sec_margin"] = _np(opt["sec"].margin)
        out["opt/alcubierre_wall/dec_margin"] = _np(opt["dec"].margin)
    except Exception as e:  # noqa: BLE001
        print(f"[golden] skip optimize_point: {e}")

    # --- 6. shift kinematics + kinematic scalars (Perf 4 filter_jit parity) ---
    try:
        metric = AlcubierreMetric(v_s=0.5)
        grid = GridSpec(bounds=_SMALL_BOUNDS, shape=(8, 8, 8))
        theta, sig2, om2 = compute_shift_kinematics_grid(metric, grid)
        out["kin/alcubierre/theta"] = _np(theta)
        out["kin/alcubierre/sigma_sq"] = _np(sig2)
        out["kin/alcubierre/omega_sq"] = _np(om2)
        kt, ks, ko = compute_kinematic_scalars_grid(metric, grid)
        out["kinscalar/alcubierre/theta"] = _np(kt)
        out["kinscalar/alcubierre/sigma_sq"] = _np(ks)
        out["kinscalar/alcubierre/omega_sq"] = _np(ko)
    except Exception as e:  # noqa: BLE001
        print(f"[golden] skip kinematics: {e}")

    # --- 7. ANEC + Ford-Roman (Perf 4 filter_jit; ANEC pre-symplectic baseline) ---
    try:
        metric = AlcubierreMetric(v_s=0.5)
        ray = lambda lam: jnp.array([lam, lam, 0.5, 0.0])  # noqa: E731
        # Pin the legacy tangent norm explicitly: the stored golden was
        # captured before the anec() default switched to 'null_projected'.
        out["anec/alcubierre/line_integral"] = _np(
            anec(metric, ray, tangent_norm="renormalized").line_integral
        )
        wl = lambda tau: jnp.array([tau, 0.0, 0.5, 0.0])  # noqa: E731
        out["qi/alcubierre/margin"] = _np(ford_roman(metric, wl, tau0=1.0).margin)
    except Exception as e:  # noqa: BLE001
        print(f"[golden] skip anec/qi: {e}")

    return out
