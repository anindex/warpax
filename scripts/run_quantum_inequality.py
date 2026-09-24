"""Flat-space Lorentzian-sampling diagnostics at coordinate-static wall points.

The numerical curves use G=c=hbar=1 and R_b=1. Their roots are dimensionless
pulse diagnostics, not coefficients that can be rescaled to a macroscopic bubble.
Separately, the short-window approximation rho_static(tau) ~= rho_static(0)
gives tau0 ~= c_short sqrt(l_P R_b), where
c_short = (C / abs(rho_static(0) R_b^2))^(1/4).

The sampled observer is u=partial_t/sqrt(-g00), not the Eulerian normal.
The flat-space massless-scalar inequality is used as an illustrative diagnostic
on a curved worldline; neither the finite integration window nor curvature
corrections are bounded here. Null integrals read from run_anec_retained.py are
finite-segment diagnostics and local minima found.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from _json_io import dump_json
from _json_io import write_table as write_tex_table
from _paper_metrics import instantiate

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from warpax.benchmarks import MinkowskiMetric
from warpax.benchmarks.alcubierre import alcubierre_shape
from warpax.quantum.ford_roman import _rho_at_tau, ford_roman

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

ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]
# These two coordinate-static density extrema have retained resolution checks.
# Omitting the other metrics makes no claim about their smoothness or algebraic type.
QI_METRICS = ["Alcubierre", "Rodal"]


def _static_worldline(x_w: float, y_w: float):
    def wl(tau):
        return jnp.stack([jnp.asarray(tau), jnp.asarray(x_w), jnp.asarray(y_w), jnp.asarray(0.0)])

    return wl


def _worst_static_point(metric) -> tuple[float, float, float]:
    """Basin-local minimum found for static-observer energy density in the wall.

    Restricted to the active wall ``f in [F_LOW, F_HIGH]`` for consistency with
    the rest of the paper. Used only for the smooth-wall drives.
    """

    def rho0(x, y):
        wl = _static_worldline(x, y)
        return float(_rho_at_tau(metric, wl, jnp.asarray(0.0)))

    def scan(x0, x1, y0, y1, n):
        best_rho, best_xy = 0.0, None
        for x in np.linspace(x0, x1, n):
            for y in np.linspace(y0, y1, n):
                f = float(alcubierre_shape(jnp.asarray(float(np.hypot(x, y))), R_B, SIGMA))
                if not (F_LOW <= f <= F_HIGH):
                    continue
                rho = rho0(float(x), float(y))
                if rho < best_rho:
                    best_rho, best_xy = rho, (float(x), float(y))
        return best_rho, best_xy

    # Coarse scan, then zoom. The raw 61x61 argmin missed the minimum by 2.8%
    # and put it at the wrong x.
    best_rho, best_xy = scan(-1.5, 1.5, 0.05, 1.6, 61)
    if best_xy is None:
        return 0.0, 0.0, 1.0
    hw = 3.0 / 60
    for _ in range(6):
        x, y = best_xy
        r, xy = scan(x - hw, x + hw, max(y - hw, 1e-3), y + hw, 9)
        if xy is not None and r < best_rho:
            best_rho, best_xy = r, xy
        hw *= 0.34
    return best_rho, best_xy[0], best_xy[1]


def _margin_curve(metric, x_w: float, y_w: float) -> np.ndarray:
    wl = _static_worldline(x_w, y_w)
    return np.array(
        [
            float(ford_roman(metric, wl, tau0=float(t), n_samples=N_SAMPLES).margin)
            for t in TAU0_GRID
        ]
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


def _short_window_coefficient(rho_static_min: float, radius: float) -> float:
    """Coefficient of sqrt(l_P R_b) in the constant-density short-window estimate."""
    return (3.0 / (32.0 * np.pi**2 * abs(rho_static_min * radius**2))) ** 0.25


def _fmt(v: float) -> str:
    return f"{v:+.3g}"


def _write_table(anec: dict, qi: dict) -> None:
    lines = [
        r"\begin{tabular}{l rr rr}",
        r"  \toprule",
        r"  & \multicolumn{2}{c}{Finite-segment $\int T_{ab}k^ak^b\,\dd\lambda$}"
        r" & \multicolumn{2}{c}{Short-window estimate} \\",
        r"  \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"  Metric & $b=0.001$ & min found ($b^\ast$) & $R_b^2\rho_{\rm static,min}$"
        r" & $c_{\rm metric}$ \\",
        r"  \midrule",
    ]
    for name in ORDER:
        a = anec["metrics"][name]
        q = qi["metrics"][name]
        if q.get("robust"):
            qi_cols = (
                f"${_fmt(q['rho_static_min'] * R_B**2)}$ & ${q['short_window_coefficient']:.2f}$"
            )
        else:
            qi_cols = r"-- & --"
        lines.append(
            f"  {name} & ${_fmt(a['on_axis'])}$ & "
            f"${_fmt(a['min_line_integral'])}$ ({a['b_at_min']:.2f}) & "
            f"{qi_cols} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}", ""]
    Path(TABLES_DIR).mkdir(parents=True, exist_ok=True)
    write_tex_table(
        os.path.join(TABLES_DIR, "averaged_quantum.tex"),
        lines,
        script="scripts/run_quantum_inequality.py",
        sources=["results/quantum/ford_roman.json", "results/anec/retained.json"],
    )


def _make_figure(anec: dict, qi: dict) -> None:
    import matplotlib

    # Headless by default: reproduce_all.sh runs on machines with no display,
    # and matplotlib otherwise falls back to Tk and dies here, *after* the
    # results JSON has already been written, so the run looks like a physics
    # failure when it is only a missing display.
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from warpax.visualization._style import DOUBLE_COL, apply_style, metric_color

    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.44))
    for name in ORDER:
        c = metric_color(name)
        a = anec["metrics"][name]
        ax_a.plot(a["b_scan"], a["line_integral_scan"], color=c, label=name, lw=1.4)
        q = qi["metrics"][name]
        if q.get("robust"):
            ax_b.plot(
                TAU0_GRID / R_B, np.array(q["margin_curve"]) * R_B**2, color=c, label=name, lw=1.4
            )
            th = q["tau0_threshold"] / R_B
            if np.isfinite(th):
                ax_b.axvline(th, color=c, ls=":", lw=0.8, alpha=0.7)

    ax_a.axhline(0.0, color="0.4", lw=0.7, ls="--")
    ax_a.set_xlabel(r"impact parameter $b$")
    ax_a.set_yscale("symlog", linthresh=1e-2)
    ax_a.set_ylabel(r"null line integral $\int T_{ab}k^ak^b\,d\lambda$")
    ax_a.set_title("(a) Finite-segment null integrals", fontsize=9)
    ax_a.legend(frameon=False, fontsize=7)

    ax_b.axhline(0.0, color="0.4", lw=0.7, ls="--")
    ax_b.set_xscale("log")
    ax_b.set_yscale("symlog", linthresh=1e-5)
    ax_b.set_xlabel(r"sampling width $\tau_0/R_b$")
    ax_b.set_ylabel(r"dimensionless sampling margin")
    ax_b.set_title(r"(b) Flat-space diagnostic, $\hbar/R_b^2=1$", fontsize=9)
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
        ford_roman(
            MinkowskiMetric(), _static_worldline(0.0, 1.0), tau0=1.0, n_samples=N_SAMPLES
        ).margin
    )
    print(
        f"Minkowski QI margin (tau0=1) = {mink_margin:+.3e} "
        f"({'PASS' if mink_margin > 0 else 'FAIL'})"
    )

    per_metric: dict[str, dict] = {}
    for name in QI_METRICS:
        metric = instantiate(name, V_S, R_B, SIGMA)
        rho_min, x_w, y_w = _worst_static_point(metric)
        margins = _margin_curve(metric, x_w, y_w)
        tau_th = _threshold(metric, x_w, y_w, margins)
        per_metric[name] = {
            "robust": True,
            "rho_min": rho_min,  # Legacy key; the observer is coordinate-static.
            "rho_static_min": rho_min,
            "short_window_coefficient": _short_window_coefficient(rho_min, R_B),
            "x_w": x_w,
            "y_w": y_w,
            "tau0_threshold": tau_th,
            "margin_curve": margins.tolist(),
        }
        print(
            f"  {name:16s} rho_static_min={rho_min:+.4e} @ (x={x_w:.2f}, y={y_w:.2f})  "
            f"dimensionless pulse root={tau_th / R_B:.3f}; "
            f"c_short={_short_window_coefficient(rho_min, R_B):.3f}"
        )
    for name in ORDER:
        if name not in per_metric:
            per_metric[name] = {"robust": False}

    qi = {
        "params": {
            "v_s": V_S,
            "R_b": R_B,
            "sigma": SIGMA,
            "ford_roman_C": float(3.0 / (32.0 * np.pi**2)),
            "hbar": 1.0,
            "hbar_over_R_b_squared": 1.0 / R_B**2,
            "observer": "u=partial_t/sqrt(-g00), future-directed where g00<0",
            "density": "rho_static=T00/(-g00); not Eulerian rho_n",
            "field": "massless scalar, four-dimensional flat-space bound",
            "sampling": "Lorentzian kernel in accumulated proper time",
            "n_samples": N_SAMPLES,
            "proper_time_half_span_over_tau0": 10.0,
            "curve_status": "finite-window diagnostic; no tail or curved-space error bound",
            "tau0_threshold_definition": "root of hbar=1 pulse diagnostic; not a macroscopic scaling coefficient",
            "short_window_definition": (
                "c_short=(C/abs(rho_static_min*R_b^2))^(1/4); "
                "tau0 approximately c_short*sqrt(l_P*R_b), assuming nearly constant "
                "density over the short sampling window"
            ),
        },
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
    print(
        f"Wrote {os.path.join(TABLES_DIR, 'averaged_quantum.tex')} "
        f"and {os.path.join(FIG_DIR, 'averaged_quantum.pdf')}"
    )


if __name__ == "__main__":
    main()
