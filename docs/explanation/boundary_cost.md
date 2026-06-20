# The boundary cost of source consistency

This page consolidates the physics of the source-first warp shells and the
companion note *On the boundary cost of source-consistent warp shells*
([arXiv:2605.25417](https://arxiv.org/abs/2605.25417)), as certified by warpax
v1.1.0. For the reproduction recipe and numbers see
[Reproducing the warp-shell admissibility paper](../how-to/reproduce_warpshell_paper.md).

## The question

Metric-first ("$G$-method") warp drives prescribe $g_{ab}$ and *read off*
$T_{ab}=G_{ab}/8\pi$, which Barzegar, Buchert & Vigneron argue can yield
"fantastic" matter with no physical interpretation. The companion note asks the
inverse: **fix physical matter, solve the Einstein constraints for the metric,
and ask whether the result can satisfy the energy conditions.** Two
*source-first* shells answer it:

- **S-shell** (Class I): a shift-free, isotropic-fluid shell; lapse and radial
  potential from the Hamiltonian constraint + anisotropic TOV equilibrium.
- **T-shell** (Class II): a tilted matter flow whose shift $\beta^x(r)$ is the
  solution of the momentum constraint (an $\ell=1$ vector-harmonic boundary-value
  problem), not a postulated coordinate ansatz.

Both are graded by a five-criterion **admissibility standard** (regularity,
constraint satisfaction, explicit matter, frame-independent EC margins, global
diagnostics).

## Certification is frame-free

Verdicts use the Hawking--Ellis classification of $T^a{}_b$ with the Type-I
eigenvalue slacks
$\mathrm{NEC}=\min_i(\rho+p_i)$, $\mathrm{WEC}=\min(\rho,\min_i(\rho+p_i))$,
$\mathrm{DEC}=\min_i(\rho-|p_i|)$. These are **exact and cap-free**: the certifier
builds no Eulerian normal, so it is valid at all warp speeds (including
$v_s\ge1$). At a Type-I point the worst observer is closed-form,
$\rho_{\rm obs}(\zeta)=\rho+(\rho+p_i)\sinh^2\zeta$; if some $\rho+p_i<0$ the
boosted density is unbounded below, so **at non-Type-I points there is no
invariant margin** — only the algebraic type and the imaginary-eigenvalue scale.
A bounded-rapidity ($\zeta_{\max}=5$) optimizer survives only as a labelled
one-sided severity *diagnostic*.

## Three results

**1. The bulk is clean; the cost is at the boundary.** Every source-prescribed
shell is Hawking--Ellis Type-I and energy-condition compliant in the
matter-filled interior (Fuchs: 0/13 interior probes violate). The single
observer-independent violation is a **Type-I dominant-energy deficit at the inner
shell edge**, $\approx-4.4\times10^{-4}$.

**2. The inner-edge floor is a geometric invariant.** That deficit is
independent of the bubble velocity *and* of the metric smoothing width; it
depends only on the **regularity class of the source profile** —
$-4.4\times10^{-4}$ (smoothstep), $-2.2\times10^{-4}$ (parabolic),
$-1.2\times10^{-4}$ (Bernstein), a factor $\sim3.7$ — and no finite-regularity
polynomial family removes it. It is a cap-free Type-I slack, hence a genuine
invariant rather than an optimizer artifact.

**3. Transport carries a change of algebraic type (vorticity $\to$ Type-IV).**
The T-shell's constraint-derived shift $\beta^i=\beta(r)\hat x^i$ is **not
curl-free**; its vorticity $\propto\beta'$ drives the stress-energy to
**Hawking--Ellis Type-IV** (no rest frame) wherever the matter density thins and
the momentum flux dominates the energy block, i.e. in the low-density transition
edges. The opened imaginary eigenvalue part is **linear in the matter tilt
$v_0$** (log--log slope $1.01$, vanishing at $v_0=0$) and is confirmed by the
three-solver gate. This instantiates, in a source-consistent shell, the
shift-vorticity $\to$ type control ($f=\kappa\omega$) the certification paper
establishes for metric-first drives: the boundary cost of transport is a
transition from Type-I to Type-IV, not merely a larger margin. Likewise the Fuchs
smoothing halo ($r>R_2$) is Type-IV (22/25 probes); the often-quoted
$-7.9\times10^{-3}$ there is the $\zeta_{\max}=5$ diagnostic, not an invariant.

## Averaged level

A rigorous symplectic geodesic-integrated ANEC (Tao-2016 extended phase space,
Yoshida-4; on-cone witness $\lesssim10^{-4}$, versus $O(0.1)$ drift for adaptive
Runge--Kutta) is **positive for every source-prescribed shell** — Fuchs
$+1.9\times10^{-3}$, S-shell $+2.9\times10^{-3}$, T-shell $+4.6\times10^{-3}$
($v_0=0.1$). Only the sign is invariant under $k^a\to\lambda k^a$, and it is
robustly positive across resolution and impact parameter. So the pointwise
transition failures need not appear in the average. A full average over a
geodesic family, and a curved-space (Fewster-type) quantum inequality, remain
open; the Ford--Roman comparison is an explicitly flat-space estimate.

## Verdict

A $20\times15$ compactness–thickness scan finds **0 of 600** configurations
strictly admissible in either shell class: source consistency is achievable
(criteria A–C, E), but the smooth source–vacuum transition exacts an
energy-condition cost that this family of profiles does not remove. The result
is consistent with the Lobo--Visser obstruction across the surveyed family; it is
not a no-go theorem.
