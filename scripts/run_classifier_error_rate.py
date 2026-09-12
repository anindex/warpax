"""Numerical label contradictions and Jordan perturbation examples.

Eigenvalues can vary as delta^(1/m) near a defective size-m block; the
Jordan labels themselves are discontinuous. The recorded perturbation fits
and floating-point splits describe these fixtures and solver settings.

Exact Types III and IV violate NEC. A positive binary64 LMI margin above
its recorded floor at such a numerical label is a detected contradiction
between numerical routes, not an independently certified label error.

Run: JAX_PLATFORMS=cpu python scripts/run_classifier_error_rate.py
"""

from __future__ import annotations

import json
import pathlib

import jax.numpy as jnp
import numpy as np

import warpax  # noqa: F401  (installs the x64 config on import)
from warpax import certify
from warpax.energy_conditions.classification import classify_hawking_ellis
from warpax.metrics.warpshell import WarpShellMetric

# WarpShell's shell sits at R_1=10 to R_2=20, so the +/-3 box used elsewhere in
# the pipeline contains no shell at all. The bounds are recorded in the output.
V_S = 0.5
SHAPE = (50, 50, 50)
BOUNDS = [(-25.0, 25.0)] * 3

ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


def jordan_split_scale():
    """Measured eigenvalue splitting of exact J_2 and J_3 blocks in float64."""
    out = {}
    eta = np.diag([-1.0, 1, 1, 1])
    rng = np.random.default_rng(0)

    # Measure conditioning: perturb by a symmetric delta and fit the eigenvalue
    # displacement, whose slope in log-log is 1/m for a J_m block. A single
    # unperturbed split understates it, since LAPACK can return dyadic entries exactly.
    def exponent(builder, lam_of, m, n=48):
        """Fitted slope of log(eigenvalue displacement) against log(perturbation).

        The measured quantity is ``max_k |lambda_k - lambda_repeated|``, not
        ``|Im lambda|``. A perturbed defective block splits either into two real
        eigenvalues or into a conjugate pair, roughly half the time each, so the
        imaginary part is zero on about half the draws and its median says nothing.
        The displacement is the quantity the delta^(1/m) law is actually about, and
        it is what determines whether any tolerance can separate the cluster.
        """
        deltas = np.array([1e-14, 1e-12, 1e-10, 1e-8])
        med = []
        for d in deltas:
            vals = []
            for _ in range(n):
                params = rng.uniform(0.5, 2.0, size=3)
                T = builder(*params)
                E = rng.normal(size=(4, 4))
                ev = np.linalg.eigvals(eta @ (T + d * (E + E.T) / 2.0))
                lam0 = lam_of(*params)
                # only the m eigenvalues of the defective cluster; the decoupled
                # eigenvalue sits O(1) away and would swamp the fit.
                near = np.sort(np.abs(ev - lam0))[:m]
                vals.append(float(near.max()))
            med.append(max(np.median(vals), 1e-300))
        slope = float(np.polyfit(np.log10(deltas), np.log10(med), 1)[0])
        return slope, float(med[-1])

    j2 = lambda mu, f, p: np.array(
        [[mu + f, f, 0, 0], [f, -mu + f, 0, 0], [0, 0, p, 0], [0, 0, 0, p]]
    )
    j3 = lambda rho, f, p: np.array(
        [[rho, 0, -f, 0], [0, -rho, f, 0], [-f, f, -rho, 0], [0, 0, 0, p + 2.0]]
    )

    # unperturbed repeated eigenvalues: J_2 block has (lam + mu)^2, J_3 has (lam + rho)^3
    out["J2_exponent"], out["J2_split_at_1e-8"] = exponent(j2, lambda mu, f, p: -mu, 2)
    out["J3_exponent"], out["J3_split_at_1e-8"] = exponent(j3, lambda rho, f, p: -rho, 3)

    # a fixed generic Type III, for the label check below
    rho, f3, p = 1.0, 1.0, 3.0
    T3 = np.array([[rho, 0, -f3, 0], [0, -rho, f3, 0], [-f3, f3, -rho, 0], [0, 0, 0, p]])
    A3 = eta @ T3
    out["J3_split_rounding_only"] = float(np.max(np.abs(np.linalg.eigvals(A3).imag)))
    eps = float(np.finfo(np.float64).eps)
    out["eps"] = eps
    out["eps_sqrt"] = eps**0.5
    out["eps_cbrt"] = eps ** (1.0 / 3.0)
    # what the classifier returns for the generic Type III, at default tolerance
    r = classify_hawking_ellis(jnp.asarray(A3), ETA, T_ab=jnp.asarray(T3))
    out["generic_type_iii_label"] = int(r.he_type)
    return out


def compare_grid():
    """Type census on the WarpShell wall, plus the LMI cross-check."""
    metric = WarpShellMetric(v_s=V_S)
    res = certify(metric, shape=SHAPE, bounds=BOUNDS)
    ff = res.frame_free
    he = np.asarray(ff.he_types).ravel()

    counts = {f"type_{t}": int((he == t).sum()) for t in (1, 2, 3, 4)}
    counts["vacuum"] = int(ff.n_vacuum)
    counts["total"] = int(he.size)

    # LMI comparison on the non-Type-I points: Types III and IV must violate everything.
    nec = np.asarray(ff.nec_margins).ravel()
    wec = np.asarray(ff.wec_margins).ravel()
    comparison = {}
    for t in (3, 4):
        sel = he == t
        n = int(sel.sum())
        if n == 0:
            comparison[f"type_{t}"] = {"n": 0}
            continue
        # Count contradictions only above the recorded binary64 noise floor.
        # This comparison cannot identify which numerical route failed.
        floor = np.asarray(ff.nec_noise_floor).ravel()[sel]
        clean = int(np.sum(nec[sel] > floor))
        comparison[f"type_{t}"] = {
            "n": n,
            "numerically_detected_label_contradictions": clean,
            "contradiction_rate_percent": 100.0 * clean / n,
            "wec_violated_percent": 100.0 * float(np.mean(wec[sel] < 0.0)),
        }
    return counts, comparison


def main():
    print("== Hawking-Ellis classifier comparison ==\n")

    split = jordan_split_scale()
    print("Resolution limit (measured, float64):")
    print(f"  machine epsilon                    {split['eps']:.3e}")
    print(f"  eps^(1/2)                          {split['eps_sqrt']:.3e}")
    print(f"  eps^(1/3)                          {split['eps_cbrt']:.3e}")
    print("  eigenvalue displacement under a perturbation delta: fitted slope of")
    print("  log max|lambda - lambda_rep| vs log delta  (theory: 1/m for a J_m block)")
    print(f"    J_2 (Type II):   {split['J2_exponent']:.3f}   (expect 0.500)")
    print(f"    J_3 (Type III):  {split['J3_exponent']:.3f}   (expect 0.333)")
    print(
        f"  J_3 split from rounding alone      {split['J3_split_rounding_only']:.3e}"
        f"   (cf. eps^(1/3) = {split['eps_cbrt']:.3e})"
    )
    print(
        f"  generic Type III is labelled Type  {split['generic_type_iii_label']}"
        "   <- must read 3; 4 means unresolvable"
    )
    print()

    counts, comparison = compare_grid()
    print(
        f"WarpShell v_s={V_S}, grid {SHAPE}, bounds {BOUNDS[0]} (recorded, "
        "so the census is reproducible):"
    )
    for k, v in counts.items():
        print(f"  {k:>10s}: {v}")
    print()
    print("LMI cross-check on the non-Type-I labels")
    print("  Numerically detected label contradictions above the recorded NEC floor:")
    for k, v in comparison.items():
        if v.get("n", 0) == 0:
            print(f"  {k}: none present")
            continue
        print(
            f"  {k}: n={v['n']}, numerically detected label contradictions: "
            f"{v['numerically_detected_label_contradictions']} points "
            f"({v['contradiction_rate_percent']:.1f}%), WEC violated at "
            f"{v['wec_violated_percent']:.1f}%"
        )

    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "classifier_error_rate.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "v_s": V_S,
        "shape": list(SHAPE),
        "bounds": BOUNDS[0],
        "resolution_limit": split,
        "counts": counts,
        "lmi_agreement": comparison,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
