"""Shared radial-sweep helper for the per-metric verification scripts.

Provides a single per-point evaluator (HE classification, robust + Eulerian
energy conditions, constraint residuals) and an aggregator (HE census,
violation counts, worst margins) used by ``run_*.py`` callers.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp


def evaluate_point(metric, coords, *, n_starts: int = 16) -> dict:
    """Single-point physics evaluation: HE type, robust EC, Eulerian EC, constraints."""
    from warpax.constraints import normalized_residuals
    from warpax.energy_conditions import (
        classify_mixed_tensor,
        compute_eulerian_ec,
        verify_point,
    )
    from warpax.geometry import compute_curvature_chain

    curv = compute_curvature_chain(metric, coords)
    T, g, gi = curv.stress_energy, curv.metric, curv.metric_inv
    cls = classify_mixed_tensor(T, g, gi)
    ec = verify_point(T, g, gi, n_starts=n_starts)
    eul = compute_eulerian_ec(T, g, gi)
    try:
        cr = normalized_residuals(metric, coords)
        eps_H = float(cr["epsilon_H"])
        eps_M = float(cr["epsilon_M"])
    except Exception:
        eps_H = eps_M = float("nan")
    return {
        "r": float(coords[1]),
        "he_type": int(cls.he_type),
        "T_norm": float(jnp.max(jnp.abs(T))),
        "ec_robust": {k: float(getattr(ec, f"{k}_margin"))
                      for k in ("nec", "wec", "sec", "dec")},
        "ec_eulerian": {k: float(eul[k]) for k in ("nec", "wec", "sec", "dec")},
        "constraints": {"epsilon_H": eps_H, "epsilon_M": eps_M},
    }


def aggregate(per_point: list[dict]) -> dict:
    """Cross-point summary: HE census, violation counts, worst margins, max residual."""
    he = {"1": 0, "2": 0, "3": 0, "4": 0}
    v_rob = {"nec": 0, "wec": 0, "sec": 0, "dec": 0}
    v_eul = {"nec": 0, "wec": 0, "sec": 0, "dec": 0}
    m_rob = {k: float("inf") for k in ("nec", "wec", "sec", "dec")}
    eps_H_max = eps_M_max = 0.0
    for p in per_point:
        he[str(p["he_type"])] = he.get(str(p["he_type"]), 0) + 1
        for k in v_rob:
            if p["ec_robust"][k] < 0:
                v_rob[k] += 1
            if p["ec_eulerian"][k] < 0:
                v_eul[k] += 1
            m_rob[k] = min(m_rob[k], p["ec_robust"][k])
        eH = p["constraints"]["epsilon_H"]
        eM = p["constraints"]["epsilon_M"]
        if eH == eH:
            eps_H_max = max(eps_H_max, eH)
        if eM == eM:
            eps_M_max = max(eps_M_max, eM)
    return {
        "n_points": len(per_point),
        "he_type_census": he,
        "violated_robust": v_rob,
        "violated_eulerian": v_eul,
        "min_margins_robust": m_rob,
        "max_epsilon_H": eps_H_max,
        "max_epsilon_M": eps_M_max,
    }


def radial_sweep(
    metric,
    *,
    r_range: tuple[float, float] = (0.5, 40.0),
    n_sweep: int = 50,
    n_starts: int = 16,
    progress: bool = True,
) -> dict:
    """Sweep ``metric`` along the +x axis and return per-point + summary."""
    jax.config.update("jax_enable_x64", True)
    rs = jnp.linspace(*r_range, n_sweep)
    per_point = []
    t0 = time.time()
    for i, r in enumerate(rs):
        if progress:
            sys.stdout.write(f"\r  {i + 1}/{n_sweep} r={float(r):6.2f}")
            sys.stdout.flush()
        coords = jnp.array([0.0, float(r), 0.0, 0.0], dtype=jnp.float64)
        per_point.append(evaluate_point(metric, coords, n_starts=n_starts))
    if progress:
        print()
    return {
        "per_point": per_point,
        "summary": aggregate(per_point),
        "elapsed_s": time.time() - t0,
    }


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  -> {path}")
