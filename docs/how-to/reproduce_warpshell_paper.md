# Reproducing the warp-shell admissibility paper

This guide records scripts and the **v1.1.1 dataset** for the
[companion shell study](https://arxiv.org/abs/2605.25417). These numbers are not
v1.5.0 benchmarks: current metric definitions and numerical methods can change
them, especially for Fuchs. Use the corresponding tagged source to reproduce
v1.1.1 values; use the commands below for new calculations with the current
implementation. See [boundary cost](../explanation/boundary_cost.md) for scope.

Current all-observer decisions use Type-I eigenvalue inequalities where reliable
and LMI tests at other types. Bounded-rapidity BFGS outputs are search diagnostics.
Null-energy integrals cover specified finite segments, not complete geodesics.

## Capabilities used by this paper

| Capability | warpax package |
|---|---|
| Frame-free Hawking-Ellis certifier + cap-free Type-I slacks | `warpax.certify`, `energy_conditions.frame_free` |
| Type-I principal-axis boost formula; Type-IV solver comparison | `energy_conditions.worst_observer_analytic`, `energy_conditions.classification` |
| Hamiltonian + momentum constraint residuals $\epsilon_{\mathcal{H}}, \epsilon_{\mathcal{M}}$ | `warpax.constraints` |
| Anisotropic TOV equilibrium; S-/T-shell constraint solvers | `warpax.constraints`, `metrics.sshell`, `metrics.tshell` |
| Israel-Darmois surface stress-energy; ADM mass + $1/r$ falloff | `warpax.junction`, `warpax.adm` |
| Finite-segment symplectic null-energy integral and null-norm drift | `averaged.anec.anec_rigorous`, `geodesics.symplectic` |
| Ford-Roman quantum-inequality diagnostic (flat-space) | `warpax.quantum.ford_roman` |
| Geodesic deviation, specified-observer blueshift, coordinate-time diagnostics | `warpax.transport`, `warpax.geodesics` |

## Metric entry points

| Metric | Class | Entry point |
|---|---|---|
| S-shell (Class I, shift-free, source-first) | `metrics.SShellMetric` / `sshell_default` | constraint-derived lapse from a prescribed isotropic source |
| T-shell (Class II, tilted-flow, source-first) | `metrics.TShellMetric` / `tshell_default` | constraint-derived lapse + shift from a prescribed source + velocity profile |
| Fuchs | `metrics.fuchs_default` | boxcar smoothing by default; `kernel_type="gaussian"` selects the Gaussian kernel |

## Companion figure and table sources

| Figure | Underlying warpax computation |
|---|---|
| Fig. 1 (geometries) | `AlcubierreMetric`, `sshell_default(v_s=0)`, `tshell_default(v_0=0.1)` |
| Fig. 2 (admissibility diagnostics) | `constraint_residual_verification.py`; frame-free radial sweeps |
| Fig. 3a (S-shell) | `sshell_default` + S-shell sweep (`scripts/run_sshell_sweep.py`) |
| Fig. 3b (T-shell) | `tshell_default` + T-shell sweep (`examples/10_phase_diagram.py --full`) |
| Fig. 4 (finite null-energy integrals) | `averaged.anec.anec_rigorous` |
| Fig. 5 (boundary-cost contributions) | Outer-edge ($r\ge R_2$) Type-IV imag. scale vs $v_0$ + cap-free Type-I floors; fit from `scripts/run_tshell_typeIV_onset.py` |
| Table I (8-proposal grid) | `scripts/verify_proposals.py` |
| Table II (3-shell summary) | `scripts/run_criterion_e_verification.py`, `scripts/verify_fuchs.py` |

## Numerical values (v1.1.1)

Constraint residuals include the prescribed Eulerian source. Full
metric-versus-source stress residuals test additional Einstein equations and
can be larger. The table retains the v1.1.1 values and their source scripts;
profile and impact-parameter surveys do not establish universal bounds.

| Claim | Value | Source |
|---|---|---|
| S-shell residuals | $\epsilon_{\mathcal{H}}\approx2\times10^{-6}$, $\epsilon_{\mathcal{M}}\equiv0$ | `scripts/constraint_residual_verification.py` |
| T-shell residuals | $\epsilon_{\mathcal{H}}\approx3\times10^{-6}$, $\epsilon_{\mathcal{M}}\approx4\times10^{-4}$ | `scripts/constraint_residual_verification.py` |
| Fuchs residuals (v1.1.1 Gaussian model) | $\epsilon_{\mathcal{H}}\approx3\times10^{-8}$, $\epsilon_{\mathcal{M}}\approx4\times10^{-4}$ | `scripts/constraint_residual_verification.py` |
| Source-consistency residual (full metric-vs-source stress) | S-shell deep interior $\sim10^{-3}$ (mean $3.7\times10^{-4}$); T-shell deep interior $\sim0.16$; Fuchs $\approx0.4$ inner edge, $0.14$ shell-averaged. Distinct from (and larger than) the constraint residuals $\epsilon_{\mathcal H},\epsilon_{\mathcal M}$. | `scripts/constraint_residual_verification.py` (`source_consistency_deep_2pct`) |
| Fuchs pre-smoothing source mismatch | $\sim640\times$ | `scripts/verify_fuchs.py` |
| Fuchs frame-free types | bulk $[R_1,R_2]$: 0/13 violate, all Type-I; tail $r>R_2$: 22/25 Type-IV | `scripts/verify_fuchs.py` |
| Inner-edge Type-I DEC deficit | Type-I slack $\approx-4.4\times10^{-4}$ at $r=R_1$ (S- and T-shell) | frame-free sweep |
| Sampled profile dependence | No resolved dependence on sampled $v_0$ or metric width; profiles: Bernstein $-1.2\times10^{-4}$, parabolic $-2.2\times10^{-4}$, smoothstep $-4.4\times10^{-4}$ | frame-free sweep; `run_v0_ablation.py` |
| T-shell tilt/type association | Type-I in bulk, Type-IV in the low-density outer edge ($r\ge R_2$) for $v_0>0$; imag. eigenvalue scale linear in $v_0$ (outer-edge log-log slope $1.01\pm0.01$, $=0$ at $v_0=0$); standard + generalized-pencil + 50-digit checks agree on Type IV | `scripts/run_tshell_typeIV_onset.py` |
| Interior DEC slacks (positive) | S-shell $+9.4\times10^{-5}$, T-shell $+9.3\times10^{-5}$ | frame-free sweep |
| ADM / source masses | Fuchs $2.51$ (integrated ADM mass, converged once the Gaussian tail decays by $r\approx25$; the surface integral at $r=R_2$ gives a finite-radius $2.98$), S-shell $3.09$, T-shell $3.12$ | `scripts/run_criterion_e_verification.py` |
| Finite-segment null-energy integrals | Fuchs $+1.9\times10^{-3}$, S-shell $+2.9\times10^{-3}$, T-shell $+4.6\times10^{-3}$ ($v_0=0.1$), $+5.4\times10^{-3}$ ($v_0=0.2$); positive on tested rays and resolutions; null-norm drift $\lesssim2\times10^{-4}$ on coarse grids, $<10^{-4}$ at the finest | `averaged.anec.anec_rigorous`; `scripts/run_anec_impact_scan.py` |
| 0/600 phase-diagram admissibility (frame-free verdict) | T-shell + S-shell $20\times15$ sweeps, EC verdict from the frame-free Hawking-Ellis certifier (`ec_feasibility_frame_free`), probes covering the smoothstep tails; the coarse grid resolves the sign but under-resolves the boundary-peak magnitude | `examples/10_phase_diagram.py --full`, `scripts/run_sshell_sweep.py` |
| Cross-proposal counts | Rodal 9/50 NEC, 46/50 DEC; Lentz 1/50 NEC, 2/50 DEC; Alcubierre/Natário/VdB 18/22/25, 29/30/35, 16/24/25 | `scripts/verify_proposals.py` |

## Current calculation commands

```bash
uv sync --extra design --extra solver --extra viz   # interpax + scipy are required

# 1. Constraint residuals + criterion-E masses/transport
uv run python scripts/constraint_residual_verification.py
uv run python scripts/run_criterion_e_verification.py

# 2. Cross-proposal grid and Fuchs verification
uv run python scripts/verify_proposals.py
uv run python scripts/verify_fuchs.py

# 3. Full phase-diagram sweeps (expensive)
uv run python examples/10_phase_diagram.py --full     # T-shell
uv run python scripts/run_sshell_sweep.py             # S-shell

# 4. Velocity / convergence / angular robustness
uv run python scripts/run_v0_ablation.py
uv run python scripts/run_tshell_convergence.py
uv run python scripts/run_tshell_kterm_angular.py

# 5. Outer-edge type checks and finite-segment impact-parameter scan
uv run python scripts/run_tshell_typeIV_onset.py
uv run python scripts/run_anec_impact_scan.py
```

Pointwise classification and a finite null-energy integral use these APIs:

```python
import warpax  # enables float64 before arrays are created
import jax.numpy as jnp
from warpax.geometry import compute_curvature_chain
from warpax.energy_conditions.frame_free import certify_point_frame_free
from warpax.averaged.anec import anec_rigorous
from warpax.metrics import tshell_default

m = tshell_default(v_0=0.1)
cur = compute_curvature_chain(m, jnp.array([0.0, 20.6, 0.0, 0.0]))   # outer edge
print(certify_point_frame_free(cur.stress_energy, cur.metric, cur.metric_inv)["he_type"])

anec = anec_rigorous(m, jnp.array([0.0, -30.0, 1e-3, 0.0]), jnp.array([1.0, 0.0, 0.0]),
                     affine_bounds=(0.0, 60.0), num_steps=16384)
print(float(anec.symplectic.line_integral), float(anec.symplectic.max_abs_g_kk))
```

## Interpretation

Use geometric units and signature $(-+++)$, and state the observer frame for
margin magnitudes. An integral's sign is unchanged by positive affine rescaling
when endpoints represent the same physical segment. `max_abs_g_kk` is a numerical
null-norm diagnostic; it does not certify zero path error or bound omitted tails.
The reported 0/600 admissibility result concerns a finite parameter survey.
