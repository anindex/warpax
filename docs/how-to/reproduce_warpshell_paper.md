# Reproducing the warp-shell admissibility paper

This guide maps every figure, table, and quantitative claim in the companion
note (*On the boundary cost of source-consistent warp shells*,
[arXiv:2605.25417](https://arxiv.org/abs/2605.25417)) to the warpax modules and
scripts that produce them, at **warpax v1.1.1**.

The paper is distributed separately; this document covers the warpax-side
computations. Conceptual background is in
[The boundary cost of source consistency](../explanation/boundary_cost.md).

> **Certification is frame-free (v1.1.0).** Energy-condition verdicts come from
> `warpax.certify` / `energy_conditions.frame_free`: the Hawking--Ellis
> classification of `T^a_b` plus the Type-I eigenvalue slacks, which are exact
> and **cap-free**. Non-Type-I points (no rest frame) are reported by algebraic
> type and imaginary-eigenvalue scale; the bounded-rapidity BFGS optimizer
> (`ζ_max=5`) is retained only as a labelled one-sided severity diagnostic.

## Capabilities used by this paper

| Capability | warpax package |
|---|---|
| Frame-free Hawking--Ellis certifier + cap-free Type-I slacks | `warpax.certify`, `energy_conditions.frame_free` |
| Closed-form Type-I worst observer; Type-IV three-solver gate | `energy_conditions.worst_observer_analytic`, `energy_conditions.classification` |
| Hamiltonian + momentum constraint residuals $\epsilon_{\mathcal{H}}, \epsilon_{\mathcal{M}}$ | `warpax.constraints` |
| Anisotropic TOV equilibrium; S-/T-shell constraint solvers | `warpax.constraints`, `metrics.sshell`, `metrics.tshell` |
| Israel--Darmois surface stress-energy; ADM mass + $1/r$ falloff | `warpax.junction`, `warpax.adm` |
| Rigorous symplectic geodesic-integrated ANEC (on-cone witness) | `averaged.anec.anec_rigorous`, `geodesics.symplectic` |
| Ford--Roman quantum-inequality diagnostic (flat-space) | `warpax.quantum.ford_roman` |
| Invariant transport ($\delta\tau$, geodesic deviation, blueshift) | `warpax.transport`, `warpax.geodesics` |

## New metrics

| Metric | Class | Entry point |
|---|---|---|
| S-shell (Class I, shift-free, source-first) | `metrics.SShellMetric` / `sshell_default` | constraint-derived lapse from a prescribed isotropic source |
| T-shell (Class II, tilted-flow, source-first) | `metrics.TShellMetric` / `tshell_default` | constraint-derived lapse + shift from a prescribed source + velocity profile |
| Fuchs canonical (Gaussian-smoothed) | `metrics.fuchs_default` | five-step iterative construction, variance-matched Gaussian kernel |

## Figure-by-figure data sources

| Figure | Underlying warpax computation |
|---|---|
| Fig. 1 (geometries) | `AlcubierreMetric`, `sshell_default(v_s=0)`, `tshell_default(v_0=0.1)` |
| Fig. 2 (admissibility diagnostics) | `constraint_residual_verification.py`; frame-free radial sweeps |
| Fig. 3a (S-shell hero) | `sshell_default` + S-shell sweep (`scripts/run_sshell_sweep.py`) |
| Fig. 3b (T-shell hero) | `tshell_default` + T-shell sweep (`examples/10_phase_diagram.py --full`) |
| Fig. 4 (symplectic geodesic ANEC) | `averaged.anec.anec_rigorous` (see `figures/make_fig4_anec.py` in the paper tree) |
| Fig. 5 (boundary-cost contributions) | Outer-edge ($r\ge R_2$) Type-IV imag. scale vs $v_0$ + cap-free Type-I floors (`figures/make_fig5_ec_scaling.py`; gate + fit `scripts/run_tshell_typeIV_gate.py`) |
| Table I (8-proposal grid) | `scripts/verify_proposals.py` |
| Table II (3-shell summary) | `scripts/run_criterion_e_verification.py`, `scripts/verify_fuchs.py` |

## Quantitative claims (v1.1.1)

Constraint residuals are **source-aware** (evaluated against the prescribed
Eulerian source). The much larger *vacuum* residual that omits the source term
(e.g. $\sim 5\times10^{-3}$) is not the constraint residual and is not quoted.

| Claim | Value | Source |
|---|---|---|
| S-shell residuals | $\epsilon_{\mathcal{H}}\approx2\times10^{-6}$, $\epsilon_{\mathcal{M}}\equiv0$ | `scripts/constraint_residual_verification.py` |
| T-shell residuals | $\epsilon_{\mathcal{H}}\approx3\times10^{-6}$, $\epsilon_{\mathcal{M}}\approx4\times10^{-4}$ | `scripts/constraint_residual_verification.py` |
| Fuchs residuals (canonical, source-aware) | $\epsilon_{\mathcal{H}}\approx3\times10^{-8}$, $\epsilon_{\mathcal{M}}\approx4\times10^{-4}$ | `scripts/constraint_residual_verification.py` |
| Source-consistency residual (full metric-vs-source stress) | S-shell deep interior $\sim10^{-3}$ (mean $3.7\times10^{-4}$); T-shell deep interior $\sim0.16$; Fuchs $\approx0.4$ inner edge, $0.14$ shell-averaged. Distinct from (and larger than) the constraint residuals $\epsilon_{\mathcal H},\epsilon_{\mathcal M}$. | `scripts/constraint_residual_verification.py` (`source_consistency_deep_2pct`) |
| Fuchs pre-smoothing source mismatch | $\sim640\times$ | `scripts/verify_fuchs.py` |
| Fuchs frame-free types | bulk $[R_1,R_2]$: 0/13 violate, all Type-I; tail $r>R_2$: 22/25 Type-IV | `scripts/verify_fuchs.py` |
| **Inner-edge DEC deficit (the binding cap-free violation)** | Type-I slack $\approx-4.4\times10^{-4}$ at $r=R_1$ (S- and T-shell) | frame-free sweep |
| Geometric-floor invariance | $v_0$-independent and metric-width-independent; profile-class only: Bernstein $-1.2\times10^{-4}$, parabolic $-2.2\times10^{-4}$, smoothstep $-4.4\times10^{-4}$ | frame-free sweep; `run_v0_ablation.py` |
| **T-shell vorticity $\to$ Type-IV** | Type-I in bulk, Type-IV in the low-density outer edge ($r\ge R_2$) for $v_0>0$; imag. eigenvalue scale linear in $v_0$ (outer-edge log--log slope $1.01\pm0.01$, $=0$ at $v_0=0$); standard + generalized-pencil + 50-digit gate all Type-IV | `scripts/run_tshell_typeIV_gate.py` |
| Interior DEC slacks (positive) | S-shell $+9.4\times10^{-5}$, T-shell $+9.3\times10^{-5}$ | frame-free sweep |
| ADM / source masses | Fuchs $2.51$ (integrated ADM mass, converged once the Gaussian tail decays by $r\approx25$; the surface integral at $r=R_2$ gives a finite-radius $2.98$), S-shell $3.09$, T-shell $3.12$ | `scripts/run_criterion_e_verification.py` |
| Symplectic geodesic ANEC (sign is the invariant) | Fuchs $+1.9\times10^{-3}$, S-shell $+2.9\times10^{-3}$, T-shell $+4.6\times10^{-3}$ ($v_0=0.1$), $+5.4\times10^{-3}$ ($v_0=0.2$); sign robust across $b\in[10^{-3},5]$ and resolution; on-cone witness $\lesssim2\times10^{-4}$ on coarse grids, $<10^{-4}$ at the finest | `averaged.anec.anec_rigorous`; `scripts/run_anec_impact_scan.py` |
| 0/600 phase-diagram admissibility (frame-free verdict) | T-shell + S-shell $20\times15$ sweeps, EC verdict from the frame-free Hawking--Ellis certifier (`ec_feasibility_frame_free`), probes covering the smoothstep tails; the coarse grid resolves the sign but under-resolves the boundary-peak magnitude | `examples/10_phase_diagram.py --full`, `scripts/run_sshell_sweep.py` |
| Cross-proposal counts | Rodal 9/50 NEC, 46/50 DEC; Lentz 1/50 NEC, 2/50 DEC; Alcubierre/Natário/VdB 18/22/25, 29/30/35, 16/24/25 | `scripts/verify_proposals.py` |

## One-shot reproduction

```bash
uv sync --extra design --extra solver --extra viz   # interpax + scipy are required

# 1. Constraint residuals + criterion-E masses/transport
uv run python scripts/constraint_residual_verification.py
uv run python scripts/run_criterion_e_verification.py

# 2. Cross-proposal grid and Fuchs verification
uv run python scripts/verify_proposals.py
uv run python scripts/verify_fuchs.py

# 3. Phase-diagram sweeps (slow: ~1 h per shell class on GPU for 20x15)
uv run python examples/10_phase_diagram.py --full     # T-shell
uv run python scripts/run_sshell_sweep.py             # S-shell

# 4. Velocity / convergence / angular robustness
uv run python scripts/run_v0_ablation.py
uv run python scripts/run_tshell_convergence.py
uv run python scripts/run_tshell_kterm_angular.py

# 5. Outer-edge Type-IV gate (slope 1.01 +/- 0.01) + ANEC impact-parameter scan
uv run python scripts/run_tshell_typeIV_gate.py
uv run python scripts/run_anec_impact_scan.py
```

Frame-free Type-IV classification and the symplectic shell-ANEC line integrals
(Fig. 4) are computed with the public APIs:

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from warpax.geometry import compute_curvature_chain
from warpax.energy_conditions.frame_free import certify_point_frame_free
from warpax.averaged.anec import anec_rigorous
from warpax.metrics import tshell_default

m = tshell_default(v_0=0.1)
cur = compute_curvature_chain(m, jnp.array([0.0, 20.6, 0.0, 0.0]))   # outer edge
print(certify_point_frame_free(cur.stress_energy, cur.metric, cur.metric_inv)["he_type"])  # -> 4

anec = anec_rigorous(m, jnp.array([0.0, -30.0, 1e-3, 0.0]), jnp.array([1.0, 0.0, 0.0]),
                     affine_bounds=(0.0, 60.0), num_steps=16384)
print(float(anec.symplectic.line_integral), float(anec.symplectic.max_abs_g_kk))  # +4.6e-3, ~5e-5
```

The paper's figure-plotting scripts (`make_fig4_anec.py`, `make_fig5_ec_scaling.py`)
ship in the paper source tree and call these same APIs.

## Conventions

- 64-bit JAX (`jax.config.update('jax_enable_x64', True)`); geometric units, signature $(-{+}{+}{+})$.
- **Verdict** = frame-free Hawking--Ellis type + cap-free Type-I slacks (no rapidity cap, valid at all $v_s$). The $\zeta_{\max}=5$ BFGS optimizer is a labelled non-Type-I severity diagnostic only.
- ANEC: symplectic null geodesic (`anec_rigorous`); only the **sign** of the line integral is invariant under $k^a\to\lambda k^a$. The on-cone witness $\max|g_{ab}k^ak^b|$ certifies the path stayed null.
