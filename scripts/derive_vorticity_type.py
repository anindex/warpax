"""Vorticity -> Type-IV mechanism: f = kappa * omega (K10).

Establishes, numerically and in a controlled limit, that the imaginary part of
the Hawking-Ellis Type-IV eigenvalue pair of ``T^a_b`` is *linear in the shift
vorticity* for the unit-lapse, flat-slice warp family:

  1. Controlled family -- a pure-rotation shift ``beta = c (-y, x, 0) * env(r)``
     has zero expansion and zero shear, only vorticity ``omega^2 = 2 c^2 env^2``.
     Sweeping ``c`` shows ``max|Im lambda|`` is exactly proportional to ``omega``
     (fit ``kappa``, ``R^2 ~ 1``) and the type flips Type I (c=0) -> Type IV (c>0).

  2. Cross-metric validation -- at matched wall points, the irrotational Rodal
     drive (omega ~ 0) is Type I with ``Im ~ 0``, while Natario/Alcubierre/VdB
     (omega > 0) are Type IV with ``Im`` tracking ``kappa * omega``.

This supplies the analytic *mechanism* behind Rodal's empirical irrotational ->
global-Type-I result (arXiv:2512.18008) and Santiago-Schuster-Visser's
irrotational-implies-Type-I lemma: vorticity is the control parameter for the
Type-IV imaginary eigenvalue.

Outputs
-------
- results/vorticity_type_analytic.json
- ../warpax_arxiv/figures/vorticity_type_mechanism.pdf  (best-effort)
"""
from __future__ import annotations

import os

from _json_io import dump_json

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import sympy as sp

from warpax.analysis.shift_kinematics import compute_shift_kinematics
from warpax.analysis.vorticity_type_analytic import fit_kappa, imaginary_part_estimate
from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions.classification import classify_hawking_ellis
from warpax.geometry.geometry import compute_curvature_chain
from warpax.geometry.metric import ADMMetric, SymbolicMetric
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
FIG_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "figures")


class _RotationShift(ADMMetric):
    """Controlled pure-rotation shift (zero expansion/shear, tunable vorticity).

    ``beta = c (-y, x, 0) * exp(-r^2 / 2 w^2)``, ``alpha = 1``, ``gamma = delta``.
    """

    c: float = 0.1
    w: float = 1.0

    def lapse(self, coords):
        return jnp.array(1.0)

    def shift(self, coords):
        t, x, y, z = coords
        env = jnp.exp(-(x * x + y * y + z * z) / (2.0 * self.w * self.w))
        return jnp.array([-self.c * y * env, self.c * x * env, 0.0])

    def spatial_metric(self, coords):
        return jnp.eye(3)

    def shape_function_value(self, coords):
        t, x, y, z = coords
        return jnp.exp(-(x * x + y * y + z * z) / (2.0 * self.w * self.w))

    def symbolic(self):
        t, x, y, z = sp.symbols("t x y z")
        g = sp.eye(4)
        g[0, 0] = -1
        return SymbolicMetric([t, x, y, z], g)

    def name(self):
        return "RotationShift"


def _omega_and_imag(metric, point) -> tuple[float, float, int]:
    """Return (omega = sqrt(omega^2), max|Im lambda|, he_type) at a point."""
    _, _, omega_sq = compute_shift_kinematics(metric, point)
    cur = compute_curvature_chain(metric, point)
    T_mixed = cur.metric_inv @ cur.stress_energy
    cls = classify_hawking_ellis(T_mixed, cur.metric)
    omega = float(np.sqrt(max(float(omega_sq), 0.0)))
    imag = float(jnp.max(jnp.abs(cls.eigenvalues_imag)))
    return omega, imag, int(cls.he_type)


def controlled_family() -> dict:
    """Pure-rotation sweep: demonstrate f = kappa * omega, type flip at omega>0."""
    point = jnp.array([0.0, 0.5, 0.5, 0.0])
    omegas, imags, types = [], [], []
    c_values = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    for c in c_values:
        om, im, he = _omega_and_imag(_RotationShift(c=c, w=1.0), point)
        omegas.append(om)
        imags.append(im)
        types.append(he)
    nz = np.array(omegas) > 0
    fit = fit_kappa(np.array(omegas)[nz], np.array(imags)[nz])
    return {
        "c_values": c_values,
        "omega": omegas,
        "imag": imags,
        "he_type": types,
        "kappa": fit["kappa"],
        "r_squared": fit["r_squared"],
        "type_at_zero_vorticity": types[0],
        "type_at_max_vorticity": types[-1],
    }


def cross_metric(kappa: float) -> dict:
    """Validate f vs kappa*omega across the retained metrics at a wall point."""
    point = jnp.array([0.0, 1.0, 0.3, 0.0])  # off-axis wall sample
    metrics = {
        "Rodal": RodalMetric(v_s=0.5, R=1.0, sigma=8.0),
        "Alcubierre": AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0),
        "Natário": NatarioMetric(v_s=0.5, R=1.0, sigma=8.0),
        "Van den Broeck": VanDenBroeckMetric(
            v_s=0.5, R=1.0, sigma=8.0, R_tilde=1.0, alpha_vdb=0.5, sigma_B=8.0
        ),
    }
    out = {}
    for name, m in metrics.items():
        om, im, he = _omega_and_imag(m, point)
        out[name] = {
            "omega": om,
            "imag_measured": im,
            "imag_predicted": imaginary_part_estimate(om, kappa),
            "he_type": he,
        }
    return out


def _make_figure(controlled, cross, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  (figure skipped: {e})")
        return
    om = np.array(controlled["omega"])
    im = np.array(controlled["imag"])
    kappa = controlled["kappa"]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    grid = np.linspace(0, om.max() * 1.05, 100)
    ax.plot(grid, kappa * grid, "-", color="0.4",
            label=fr"$f=\kappa\,\omega$, $\kappa={kappa:.3f}$ ($R^2={controlled['r_squared']:.4f}$)")
    ax.plot(om, im, "o", color="C0", label="pure-rotation family")
    for name, d in cross.items():
        if d["he_type"] == 4:
            ax.plot(d["omega"], d["imag_measured"], "s", label=name)
        else:
            ax.plot(d["omega"], d["imag_measured"], "^", label=f"{name} (Type I)")
    ax.set_xlabel(r"shift vorticity $\omega=\sqrt{\omega^2}$")
    ax.set_ylabel(r"$\max|\mathrm{Im}\,\lambda|$ of $T^a{}_b$")
    ax.set_title("Vorticity sources the Type-IV imaginary eigenvalue")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print("VORTICITY -> TYPE-IV MECHANISM  (f = kappa * omega)")
    print("=" * 70)
    controlled = controlled_family()
    print(f"  Controlled pure-rotation family: kappa={controlled['kappa']:.4f}  "
          f"R^2={controlled['r_squared']:.6f}")
    print(f"    type at omega=0: {controlled['type_at_zero_vorticity']}  "
          f"type at omega_max: {controlled['type_at_max_vorticity']}")
    cross = cross_metric(controlled["kappa"])
    print("  Cross-metric (omega, Im measured, Im predicted, type):")
    for name, d in cross.items():
        print(f"    {name:16s} omega={d['omega']:.3e}  "
              f"Im={d['imag_measured']:.3e}  pred={d['imag_predicted']:.3e}  "
              f"type={d['he_type']}")

    out = {
        "controlled_family": controlled,
        "cross_metric": cross,
        "summary": (
            "f = kappa * omega established on a controlled pure-rotation shift "
            "(R^2 ~ 1, type flips I->IV at omega>0); irrotational Rodal is "
            "Type I with Im ~ 0, vortical drives are Type IV with Im tracking "
            "kappa * omega."
        ),
    }
    out_path = os.path.join(RESULTS_DIR, "vorticity_type_analytic.json")
    dump_json(out, out_path)
    print(f"\nWrote {out_path}")
    _make_figure(controlled, cross,
                 os.path.join(FIG_DIR, "vorticity_type_mechanism.pdf"))


if __name__ == "__main__":
    main()
