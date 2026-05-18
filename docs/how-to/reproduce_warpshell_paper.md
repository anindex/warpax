# Reproducing the warp-shell admissibility paper

This guide maps every figure, table, and quantitative claim in the warp-shell
admissibility paper (*Positive-energy warp shells and the boundary cost of
source consistency*) to the warpax modules, example scripts, and verification
scripts that produce them.

The paper itself is distributed separately; this document covers only the
warpax-side computations that feed the paper. Plotting scripts that consume the
output JSON / `.npz` files are bundled with the paper sources.

## Capabilities introduced for this paper

The warp-shell paper relies on five capabilities added to warpax for
source-consistency verification of warp shells:

| Capability | warpax package |
|---|---|
| Hamiltonian + momentum constraint residuals $\epsilon_{\mathcal{H}}, \epsilon_{\mathcal{M}}$ | `warpax.constraints` |
| Anisotropic TOV equilibrium for prescribed sources | `warpax.tov` |
| Israel--Darmois surface stress-energy at shell junctions | `warpax.junction` |
| ADM mass with $1/r$ asymptotic falloff verification | `warpax.adm` |
| Invariant transport diagnostics (geodesic deviation, null round-trip asymmetry $\delta\tau$, blueshift hazard $\mathcal{B}$) | `warpax.transport` |

## New metrics

| Metric | Class | Entry point |
|---|---|---|
| S-shell (Class I, shift-free, source-first) | `warpax.metrics.SShellMetric` / `sshell_default` | constraint-derived lapse from prescribed isotropic source |
| T-shell (Class II, tilted-flow, source-first) | `warpax.metrics.TShellMetric` / `tshell_default` | constraint-derived lapse + shift from prescribed source + velocity profile |
| Fuchs canonical (Gaussian-smoothed) | `warpax.metrics.fuchs_default` | five-step iterative construction with variance-matched Gaussian kernel |
| Fuchs pre-smoothing intermediate | `warpax.metrics._fuchs_analytical_default` | constant-density TOV intermediate, used only as the legacy comparison baseline |

## Figure-by-figure data sources

Each figure in the paper consumes output produced by a warpax script or
example. The plotting code itself is shipped with the paper source tree.

| Figure | Underlying warpax computation |
|---|---|
| Fig. 1 (geometries) | `AlcubierreMetric`, `SShellMetric(v_s=0)`, `TShellMetric(v_0=0.1)` |
| Fig. 2 (admissibility diagnostics) | `fuchs_default`, `sshell_default`, `tshell_default`; radial sweeps with `n_starts=16` |
| Fig. 3a (S-shell hero) | `sshell_default` + S-shell phase sweep (`scripts/run_sshell_sweep.py`) |
| Fig. 3b (T-shell hero) | `tshell_default` + T-shell phase sweep (`examples/10_phase_diagram.py --full`) |
| Fig. 4 (null-energy contraction along coordinate ray) | `scripts/run_anec_profiles.py`; off-axis null ray at $y=10^{-3}$ |
| Fig. 5 (EC scaling) | T-shell DEC margin vs $v_0$ for two profile families |
| Table I (8-proposal grid) | `scripts/verify_proposals.py` → `results/proposals_verification_report.json` |
| Table II (3-shell summary) | `scripts/verify_fuchs.py` + S/T-shell construction outputs |

## Quantitative claims

| Claim | Source |
|---|---|
| Fuchs constraint residual $\epsilon_{\mathcal{H}} \approx 4\times10^{-3}$ (canonical), $0.165$ (pre-smoothing) | `scripts/verify_fuchs.py` → `results/fuchs_verification_report.json` |
| Fuchs source-consistency mismatch $\sim 640\times$ | `scripts/verify_fuchs.py` → `source_consistency.max_relative_residual` |
| 0/13 interior, 22/25 exterior tail violations under observer-robust certification | `scripts/verify_fuchs.py` → `ec_summary` |
| S/T-shell constraint residual $\epsilon_{\mathcal{H}} \approx 5\times10^{-3}$ | `examples/09_admissibility_diagnostics.py` for either shell |
| 0/600 phase-diagram admissibility | `examples/10_phase_diagram.py --full` (T-shell) and `scripts/run_sshell_sweep.py` (S-shell) |
| Grid convergence at $n_{\rm grid} \in \{256, 512, 1024\}$ | `scripts/run_convergence.py` |
| Null-ray ANEC line integrals (Fuchs $+5.0\times10^{-4}$, S-shell $+6.5\times10^{-4}$, T-shell $+3.8\times10^{-3}$) | `scripts/run_anec_profiles.py` |
| Geodesic-integrated ANEC cross-check ($|g_{ab}k^ak^b| < 10^{-6}$) | `scripts/run_anec_geodesic_check.py` |
| Cross-proposal verification (Rodal $9/50$ NEC, $46/50$ DEC; Lentz $1/50$ NEC, $2/50$ DEC) | `scripts/verify_proposals.py` → `results/proposals_verification_report.json` |

## One-shot reproduction

```bash
# 1. Cross-proposal verification reports
.venv/bin/python scripts/verify_fuchs.py
.venv/bin/python scripts/verify_proposals.py

# 2. Phase-diagram sweeps (slow: ~1 hour per shell class on GPU for 20x15)
.venv/bin/python examples/10_phase_diagram.py --full     # T-shell sweep
.venv/bin/python scripts/run_sshell_sweep.py             # S-shell sweep

# 3. ANEC line integrals and geodesic cross-check
.venv/bin/python scripts/run_anec_profiles.py
.venv/bin/python scripts/run_anec_geodesic_check.py

# 4. Grid convergence study
.venv/bin/python scripts/run_convergence.py
```

After these complete, the JSON / `.npz` files in `results/` and `output/`
contain every datapoint the paper figures and tables consume.

## Numerical tolerances and conventions

- 64-bit JAX arithmetic throughout (`jax.config.update('jax_enable_x64', True)`).
- Geodesic integration: Diffrax adaptive Runge--Kutta with $\mathrm{rtol} = \mathrm{atol} = 10^{-10}$; null-tangent renormalization at every step keeps $|g_{ab}k^ak^b|<10^{-6}$ for source-prescribed shells along the integration path.
- Multi-start BFGS: $n_{\rm starts} = 8$ for the phase sweep, $n_{\rm starts} = 16$ for single-metric certification; rapidity cap $\zeta_{\max} = 5$ (maximum Lorentz factor $\gamma_{\max} \approx 74$).
- Geometric units ($G = c = 1$) with signature $(-{+}{+}{+})$ throughout.
