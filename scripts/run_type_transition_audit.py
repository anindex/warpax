"""Sweep analytic families through the Type-I / Type-II / Type-IV transition.

Why this exists
---------------
The Hawking--Ellis types I and IV do not exhaust the four. In the momentum plane,
with ``A = [[-rho, j], [-j, S_par]]`` and ``Delta = (rho + S_par)^2 - 4 j^2``, a
continuous transition from ``Delta > 0`` to ``Delta < 0`` at fixed ``j != 0`` passes
through ``Delta = 0``, where the repeated eigenvalue has a single *null* eigenvector:
the point is Type II. So a decision procedure that consults the algebraic type cannot
be complete, and the classifier's Type-II and Type-III strata are exactly the ones a
floating-point eigensolver cannot resolve (Martin-Moruno and Visser show these two
types are unstable under perturbation while I and IV are stable).

The manuscript's answer is that the energy-condition verdict never consults the type:
it comes from a 4x4 linear matrix inequality. This script measures that claim on
analytic families where the ground truth is known in closed form, rather than on a
grid where it is not. The previous audit ran on a numerical wall whose Type-III branch
tested *zero* points, which made half of it vacuous.

The families
------------
All are written in the Minkowski orthonormal frame, so the classifier, the LMI and a
brute-force observer scan see literally the same matrix and no tetrad step can hide a
discrepancy.

``momentum_aligned``
    ``rho = S_par = 1``, ``p_perp = 0``, ``j`` sweeping through 1. ``Delta = 4(1-j^2)``:
    Type I below, Type IV above, and at ``j = 1`` the momentum block is a nilpotent
    ``J_2(0)``, Type II exactly. The null deficit is ``1 - j^2`` below and
    ``2 - 2j`` above, continuous and vanishing at the locus.

``momentum_decoupled``
    Same ``Delta`` and the same Type II at ``j = 1``, but ``p_perp = -3`` puts the
    null-cone minimum at ``-2 - j^2/4``, which never changes sign. The label flips
    from I to IV while nothing physical happens. This is the family that shows the
    algebraic type is not a proxy for an energy condition.

``transverse``
    The manuscript's own Corollary-1 witness: a transverse coupling ``m`` opens a
    complex pair through the cubic discriminant with ``Delta = 3 > 0`` throughout, so
    Type IV occurs where the momentum discriminant says it should not.

``type_iii_chain``
    A genuine Segre [3,1] family, ``f`` sweeping down to where float64 loses it.
    Ground truth comes from the 50-digit mpmath gate, which separates [3,1] from
    [2,1,1] by the Jordan defect rather than by counting distinct eigenvalues.

Run:  JAX_PLATFORMS=cpu python scripts/run_type_transition_audit.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

import warpax  # noqa: F401  (installs the x64 config on import)
import jax.numpy as jnp
from warpax.energy_conditions.classification import classify_hawking_ellis
from warpax.energy_conditions.slemma import certify_point, noise_floor, null_deficit

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
TABLES = pathlib.Path(__file__).resolve().parents[2] / "warpax_arxiv" / "tables"

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
BRUTE_N = 400_000
BRUTE_SEED = 0
TOLS = (1e-10, 2e-6, 5e-6)


# --------------------------------------------------------------------------- #
# families
# --------------------------------------------------------------------------- #

def momentum_block(j, rho=1.0, s_par=1.0, p_perp=0.0):
    """Covariant ``T_ab`` whose mixed momentum block is the report's ``A``."""
    return np.array([
        [rho, -j, 0.0, 0.0],
        [-j, s_par, 0.0, 0.0],
        [0.0, 0.0, p_perp, 0.0],
        [0.0, 0.0, 0.0, p_perp],
    ])


def transverse_block(m, rho=1.0, j=0.5, s_x=1.0):
    """Corollary-1 witness: Type IV opened by a transverse coupling at ``Delta > 0``."""
    return np.array([
        [rho, -j, 0.0, 0.0],
        [-j, s_x, m, 0.0],
        [0.0, m, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])


def segre31_block(f, r=0.7, p=0.3):
    """Genuine Segre [3,1]: a size-three Jordan chain for every ``f != 0``."""
    return np.array([
        [r, 0.0, -f, 0.0],
        [0.0, -r, f, 0.0],
        [-f, f, -r, 0.0],
        [0.0, 0.0, 0.0, p],
    ])


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #

def _brute_sphere_min(T_ab, directions):
    """Minimum of ``T_ab k^a k^b`` over the null cone, by direct sampling.

    With ``eta`` the frame metric and ``k = (1, s)``, ``|s| = 1``, the contraction is
    ``rho + 2 b.s + s^T S s`` in the notation of the appendix. Independent of the LMI
    and of the classifier, so agreement is evidence and not a tautology.
    """
    rho = T_ab[0, 0]
    b = -T_ab[0, 1:]
    S = T_ab[1:, 1:]
    return float((rho + 2.0 * directions @ b
                  + np.einsum("ni,ij,nj->n", directions, S, directions)).min())


def _mixed(T_ab):
    """``T^a_b = g^{ac} T_cb``. With ``eta`` its own inverse this is ``eta @ T_ab``.

    For the momentum family this is exactly the report's block
    ``A = [[-rho, j], [-j, S_par]]``.
    """
    return ETA @ T_ab


def _label(T_ab, tol):
    res = classify_hawking_ellis(jnp.asarray(_mixed(T_ab)), jnp.asarray(ETA), tol=tol)
    return int(res.he_type)


def _labels(T_ab):
    """Float64 label at three tolerances, to show the label is a function of ``tol``."""
    out = {}
    for tol in TOLS:
        try:
            out[f"tol_{tol:g}"] = _label(T_ab, tol)
        except Exception:
            out[f"tol_{tol:g}"] = -1
    return out


def sweep(name, builder, params, directions):
    rows = []
    for t in params:
        T = builder(t)
        Tj, gj = jnp.asarray(T), jnp.asarray(ETA)
        margins = certify_point(Tj, gj)
        rows.append({
            "param": float(t),
            "nec_margin": float(margins["nec"]),
            "wec_margin": float(margins["wec"]),
            "null_deficit": float(null_deficit(Tj, gj)),
            "brute_sphere_min": _brute_sphere_min(T, directions),
            "noise_floor": float(noise_floor(Tj, gj, condition="nec")),
            "labels": _labels(T),
        })

    margins = np.array([r["nec_margin"] for r in rows])
    ps = np.array([r["param"] for r in rows])
    # Discrete Lipschitz constant of the margin: finite means continuous across the
    # locus, which is the whole point, the label is not.
    lip = float(np.max(np.abs(np.diff(margins)) / np.maximum(np.diff(ps), 1e-30)))
    primary = np.array([r["labels"][f"tol_{TOLS[0]:g}"] for r in rows])
    jumps = [int(i) for i in np.flatnonzero(np.diff(primary) != 0)]
    err = float(np.max(np.abs(2.0 * margins
                              - np.array([r["brute_sphere_min"] for r in rows]))))
    return {
        "param_name": name,
        "n": len(rows),
        "margin_lipschitz": lip,
        "label_jump_indices": jumps,
        "n_labelled_type_ii": int((primary == 2).sum()),
        "n_labelled_type_iii": int((primary == 3).sum()),
        "max_lmi_vs_brute_abs_err": err,
        "rows": rows,
    }


def type_iii_arm(fs, directions):
    """Where does the float64 label lose a genuine Type III, and does the LMI care?"""
    from warpax.energy_conditions.classification_mpmath import (
        classify_hawking_ellis_mpmath,
    )

    rows = []
    for f in fs:
        T = segre31_block(f)
        Tj, gj = jnp.asarray(T), jnp.asarray(ETA)
        m = certify_point(Tj, gj)
        try:
            exact = int(classify_hawking_ellis_mpmath(_mixed(T), ETA,
                                                      precision=50)["he_type"])
        except Exception:
            exact = -1
        rows.append({
            "f": float(f),
            "he_type_mpmath_50digit": exact,
            "labels": _labels(T),
            "nec_margin": float(m["nec"]),
            "noise_floor": float(noise_floor(Tj, gj, condition="nec")),
            "brute_sphere_min": _brute_sphere_min(T, directions),
        })

    def _first(pred):
        for r in rows:
            if pred(r):
                return r["f"]
        return None

    # Smallest chain strength at which float64 still agrees with 50 digits, and the
    # smallest at which the LMI still certifies a violation. The gap between them is
    # the point: the verdict outlives the label by orders of magnitude.
    agree = [r["f"] for r in rows
             if r["labels"][f"tol_{TOLS[0]:g}"] == r["he_type_mpmath_50digit"]]
    certifies = [r["f"] for r in rows if r["nec_margin"] < -r["noise_floor"]]
    return {
        "n": len(rows),
        "n_mpmath_type_iii": sum(1 for r in rows if r["he_type_mpmath_50digit"] == 3),
        "f_label_agrees_down_to": min(agree) if agree else None,
        "f_lmi_certifies_down_to": min(certifies) if certifies else None,
        "rows": rows,
    }


def write_table(out, path):
    def row(key, label):
        d = out["families"][key]
        return (f"  {label} & {d['n']} & {d['n_labelled_type_ii']} & "
                f"{d['margin_lipschitz']:.3g} & {d['max_lmi_vs_brute_abs_err']:.1e} \\\\")

    lines = [
        r"% Generated by scripts/run_type_transition_audit.py; do not edit.",
        r"\begin{tabular}{@{}l cccc@{}}",
        r"  \toprule",
        r"  Family & points & labelled II & margin Lip.\ & $|2\mathcal{N}_{\rm LMI}"
        r" - \min_{|s|=1}q|$ \\",
        r"  \midrule",
        row("momentum_aligned", r"Momentum, $\Delta\to0$"),
        row("momentum_decoupled", r"Momentum, decoupled"),
        row("transverse", r"Transverse ($\Delta>0$)"),
        r"  \bottomrule",
        r"\end{tabular}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def main():
    rng = np.random.default_rng(BRUTE_SEED)
    dirs = rng.normal(size=(BRUTE_N, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # 401 points, so j = 1, the Type-II locus, is hit exactly.
    js = np.linspace(0.0, 2.0, 401)
    ms = np.linspace(0.0, 3.0, 301)
    fs = np.logspace(-6.0, 0.0, 121)

    out = {
        "brute_force": {"n_samples": BRUTE_N, "seed": BRUTE_SEED},
        "tolerances": list(TOLS),
        "families": {
            "momentum_aligned": sweep("j", lambda j: momentum_block(j), js, dirs),
            "momentum_decoupled": sweep(
                "j", lambda j: momentum_block(j, p_perp=-3.0), js, dirs),
            "transverse": sweep("m", transverse_block, ms, dirs),
        },
        "type_iii_chain": type_iii_arm(fs, dirs),
    }

    print("=" * 72)
    print("TYPE-TRANSITION AUDIT   (LMI vs classifier across the Type-II locus)")
    print("=" * 72)
    for k, d in out["families"].items():
        print(f"  {k:20s} n={d['n']:4d}  labelled II={d['n_labelled_type_ii']:3d}  "
              f"margin Lipschitz={d['margin_lipschitz']:.4g}  "
              f"max|LMI-brute|={d['max_lmi_vs_brute_abs_err']:.2e}")
    t3 = out["type_iii_chain"]
    print(f"  type_iii_chain       n={t3['n']:4d}  "
          f"mpmath says III at {t3['n_mpmath_type_iii']} of them; "
          f"float64 label agrees down to f={t3['f_label_agrees_down_to']}, "
          f"LMI certifies down to f={t3['f_lmi_certifies_down_to']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "type_transition_audit.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nWrote {path}")
    write_table(out, TABLES / "type_transition.tex")


if __name__ == "__main__":
    main()
