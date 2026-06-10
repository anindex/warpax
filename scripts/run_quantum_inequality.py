"""Ford--Roman quantum-inequality diagnostic for the retained warp metrics.

At a fixed spatial point in the bubble wall a coordinate-static (Eulerian)
observer sees the warp bubble sweep past, so the sampled energy density
``rho(tau)`` is a temporary negative pulse -- precisely the situation the
Ford--Roman quantum inequality constrains. For each retained metric we locate
the most-negative static-observer energy density in the wall, then evaluate the
flat-space Ford--Roman inequality

    integral = \\int rho(tau) f(tau)^2 dtau  >=  -C / tau_0^4 ,
    C = 3 / (32 pi^2),  f Lorentzian of width tau_0,

along that worldline as a function of the sampling time ``tau_0``, and report
the threshold ``tau_0^th`` beyond which the inequality is violated.

This is applied as a *flat-space sampling diagnostic* along a curved-spacetime
worldline; a fully rigorous curved-space quantum inequality would carry
curvature corrections. The Eulerian (coordinate-static) observer exists only
for ``v_s < 1``, matching the subluminal scope of the single-frame comparison.

Also emits the combined "averaged and quantum diagnostics" table and figure,
reading the ANEC line-integral results from ``run_anec_retained.py``.

Outputs:
- ../results/quantum/ford_roman.json
- ../../warpax_arxiv/tables/averaged_quantum.tex
- ../figures/averaged_quantum.pdf
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from _json_io import dump_json

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric
from warpax.benchmarks.alcubierre import alcubierre_shape
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric
from warpax.quantum.ford_roman import ford_roman, _rho_at_tau

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results", "quantum")
ANEC_JSON = os.path.join(HERE, "..", "results", "anec", "retained.json")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")
FIG_DIR = os.path.join(HERE, "..", "figures")
PAPER_FIG_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "figures")

V_S, R_B, SIGMA = 0.5, 1.0, 8.0
N_SAMPLES = 1024
TAU0_GRID = np.geomspace(0.1, 40.0, 200)
F_LOW, F_HIGH = 0.1, 0.9

METRICS = {
    "Alcubierre": (AlcubierreMetric, {}),
    "Natário": (NatarioMetric, {}),
    "Van den Broeck": (
        VanDenBroeckMetric,
        {"R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0},
    ),
    "Rodal": (RodalMetric, {}),
}
ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]
# The Ford--Roman threshold requires a smooth, resolution-stable wall energy
# density. The Type-IV-walled Natário/VdB have no invariant energy density and
# their steep-wall Eulerian density is oscillatory and resolution-marginal, so a
# static-observer QI threshold is not a robust observable there; the ANEC
# line-integral diagnostic (resolution-converged) is used for them instead. The
# QI threshold is reported only for the smooth-wall drives.
QI_METRICS = ["Alcubierre", "Rodal"]


def _instantiate(name: str):
    cls, extra = METRICS[name]
    return cls(v_s=V_S, R=R_B, sigma=SIGMA, **extra)


def _static_worldline(x_w: float, y_w: float):
    def wl(tau):
        return jnp.stack(
            [jnp.asarray(tau), jnp.asarray(x_w), jnp.asarray(y_w), jnp.asarray(0.0)]
        )

    return wl


def _worst_static_point(metric) -> tuple[float, float, float]:
    """Most-negative static-observer energy density within the bubble wall.

    Restricted to the active wall ``f in [F_LOW, F_HIGH]`` for consistency with
    the rest of the paper. Used only for the smooth-wall drives.
    """
    xs = np.linspace(-1.5, 1.5, 61)
    ys = np.linspace(0.05, 1.6, 61)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

    def rho0(x, y):
        wl = _static_worldline(x, y)
        return float(_rho_at_tau(metric, wl, jnp.asarray(0.0)))

    best_rho, best_xy = 0.0, (0.0, 1.0)
    for x, y in pts:
        r_s = float(np.hypot(x, y))
        f = float(alcubierre_shape(jnp.asarray(r_s), R_B, SIGMA))
        if not (F_LOW <= f <= F_HIGH):
            continue
        rho = rho0(float(x), float(y))
        if rho < best_rho:
            best_rho, best_xy = rho, (float(x), float(y))
    return best_rho, best_xy[0], best_xy[1]


def _margin_curve(metric, x_w: float, y_w: float) -> np.ndarray:
    wl = _static_worldline(x_w, y_w)
    return np.array(
        [float(ford_roman(metric, wl, tau0=float(t), n_samples=N_SAMPLES).margin)
         for t in TAU0_GRID]
    )


def _margin_at(metric, x_w: float, y_w: float, tau0: float) -> float:
    wl = _static_worldline(x_w, y_w)
    return float(ford_roman(metric, wl, tau0=tau0, n_samples=N_SAMPLES).margin)


def _threshold(metric, x_w: float, y_w: float, margins: np.ndarray) -> float:
    """tau_0 where the QI margin crosses from >=0 to <0, bisection-refined."""
    sign = margins < 0.0
    idx = int(np.argmax(sign)) if sign.any() else -1
    if idx <= 0:
        return float("nan")
    lo, hi = float(TAU0_GRID[idx - 1]), float(TAU0_GRID[idx])
    for _ in range(40):
        mid = float(np.sqrt(lo * hi))
        if _margin_at(metric, x_w, y_w, mid) < 0.0:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


def _fmt(v: float) -> str:
    return f"{v:+.3g}"


def _write_table(anec: dict, qi: dict) -> None:
    lines = [
        r"\begin{tabular}{l rr rr}",
        r"  \toprule",
        r"  & \multicolumn{2}{c}{ANEC null-ray $\int T_{ab}k^ak^b\,\dd\lambda$}"
        r" & \multicolumn{2}{c}{Ford--Roman QI} \\",
        r"  \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"  Metric & on-axis & min ($b^\ast$) & $\rho_{\min}$"
        r" & $\tau_0^{\mathrm{th}}$ \\",
        r"  \midrule",
    ]
    for name in ORDER:
        a = anec["metrics"][name]
        q = qi["metrics"][name]
        if q.get("robust"):
            qi_cols = f"${_fmt(q['rho_min'])}$ & ${q['tau0_threshold']:.2f}$"
        else:
            qi_cols = r"-- & --"
        lines.append(
            f"  {name} & ${_fmt(a['on_axis'])}$ & "
            f"${_fmt(a['min_line_integral'])}$ ({a['b_at_min']:.2f}) & "
            f"{qi_cols} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}", ""]
    Path(TABLES_DIR).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "averaged_quantum.tex"), "w") as f:
        f.write("\n".join(lines))


def _make_figure(anec: dict, qi: dict) -> None:
    import matplotlib.pyplot as plt

    from warpax.visualization._style import DOUBLE_COL, apply_style, metric_color

    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.44)
    )
    for name in ORDER:
        c = metric_color(name)
        a = anec["metrics"][name]
        ax_a.plot(a["b_scan"], a["line_integral_scan"], color=c, label=name, lw=1.4)
        q = qi["metrics"][name]
        if q.get("robust"):
            ax_b.plot(TAU0_GRID, q["margin_curve"], color=c, label=name, lw=1.4)
            th = q["tau0_threshold"]
            if np.isfinite(th):
                ax_b.axvline(th, color=c, ls=":", lw=0.8, alpha=0.7)

    ax_a.axhline(0.0, color="0.4", lw=0.7, ls="--")
    ax_a.set_xlabel(r"impact parameter $b$")
    ax_a.set_ylabel(r"null line integral $\int T_{ab}k^ak^b\,d\lambda$")
    ax_a.set_title("(a) Averaged null energy along null rays", fontsize=9)
    ax_a.legend(frameon=False, fontsize=7)

    ax_b.axhline(0.0, color="0.4", lw=0.7, ls="--")
    ax_b.set_xscale("log")
    ax_b.set_yscale("symlog", linthresh=1e-5)
    ax_b.set_xlabel(r"sampling time $\tau_0$")
    ax_b.set_ylabel(r"Ford--Roman QI margin")
    ax_b.set_title("(b) Quantum inequality, smooth-wall drives", fontsize=9)
    ax_b.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    out = os.path.join(FIG_DIR, "averaged_quantum.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    # mirror into the paper tree (reproduce_all.sh also syncs figures)
    Path(PAPER_FIG_DIR).mkdir(parents=True, exist_ok=True)
    shutil.copy(out, os.path.join(PAPER_FIG_DIR, "averaged_quantum.pdf"))


def main() -> None:
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Minkowski sentinel: vacuum QI margin must be strictly positive (=+C/tau0^4).
    mink_margin = float(
        ford_roman(MinkowskiMetric(), _static_worldline(0.0, 1.0),
                   tau0=1.0, n_samples=N_SAMPLES).margin
    )
    print(f"Minkowski QI margin (tau0=1) = {mink_margin:+.3e} "
          f"({'PASS' if mink_margin > 0 else 'FAIL'})")

    per_metric: dict[str, dict] = {}
    for name in QI_METRICS:
        metric = _instantiate(name)
        rho_min, x_w, y_w = _worst_static_point(metric)
        margins = _margin_curve(metric, x_w, y_w)
        tau_th = _threshold(metric, x_w, y_w, margins)
        per_metric[name] = {
            "robust": True,
            "rho_min": rho_min,
            "x_w": x_w,
            "y_w": y_w,
            "tau0_threshold": tau_th,
            "margin_curve": margins.tolist(),
        }
        print(f"  {name:16s} rho_min={rho_min:+.4e} @ (x={x_w:.2f}, y={y_w:.2f})  "
              f"tau0_th={tau_th:.3f}")
    for name in ORDER:
        if name not in per_metric:
            per_metric[name] = {"robust": False}

    qi = {
        "params": {"v_s": V_S, "R_b": R_B, "sigma": SIGMA,
                   "ford_roman_C": float(3.0 / (32.0 * np.pi ** 2))},
        "tau0_grid": TAU0_GRID.tolist(),
        "minkowski_margin": mink_margin,
        "order": ORDER,
        "metrics": per_metric,
    }
    dump_json(qi, os.path.join(RESULTS_DIR, "ford_roman.json"))
    print(f"Wrote {os.path.join(RESULTS_DIR, 'ford_roman.json')}")

    with open(ANEC_JSON) as f:
        anec = json.load(f)
    _write_table(anec, qi)
    _make_figure(anec, qi)
    print(f"Wrote {os.path.join(TABLES_DIR, 'averaged_quantum.tex')} "
          f"and {os.path.join(FIG_DIR, 'averaged_quantum.pdf')}")


if __name__ == "__main__":
    main()
