# Boundary cost of source-consistent shells

The [companion note](https://arxiv.org/abs/2605.25417) studies shell metrics
constructed from prescribed matter. Its v1.1.1 numerical results and
reproduction commands are listed in the
[companion guide](../how-to/reproduce_warpshell_paper.md).

Metric-first constructions infer $T_{ab}=G_{ab}/8\pi$ from a chosen metric.
Source-first constructions instead prescribe matter and solve the Einstein
constraints. The remaining questions are whether the full stress agrees with
that source, whether equilibrium holds, and whether the energy conditions hold.
Small constraint residuals alone do not settle these questions.

| Construction | Source and geometry |
|---|---|
| S-shell | Flow orthogonal to the slice; zero shift; metric potentials from the Hamiltonian constraint and TOV equilibrium |
| T-shell | Tilted matter flow; radial shift profile from the momentum constraint |
| Fuchs | Smoothed shell construction; the moving metric modifies covariant $g_{01}=g_{10}$, with ADM fields recovered afterward |

## What the diagnostics establish

Type-I rest-frame slacks give cap-free energy-condition inequalities. The
all-observer LMI tests also apply at other algebraic types. Bounded-rapidity
observer searches measure violations within their stated cap; their magnitude
is not a frame-invariant measure of severity.

The companion scans found positive interior slacks and negative DEC slacks near
source-vacuum transitions. In the tested profile families, the inner-edge
Type-I deficit changed with the source profile. This finite survey does not
prove that every smooth profile has a nonzero deficit or that regularity class
alone fixes its value.

Tilted T-shell scans also found Type-IV regions near low-density edges and an
approximately linear dependence of the imaginary eigenvalue scale on matter
tilt. This is an observed association within that family. Vorticity alone does
not determine Hawking-Ellis type.

## Averaged and global checks

The symplectic integrator computes $\int T_{ab}k^ak^b\,d\lambda$ on a specified
finite null segment. Positive values on selected rays do not establish ANEC
on complete geodesics or an entire geodesic family. Step refinement and
$\max|g(k,k)|$ assess numerical behavior on the retained segment; neither bounds
omitted tails. The Ford-Roman comparison is a flat-space diagnostic with its
own sampling assumptions.

The reported compactness-thickness survey found no configuration satisfying
all of its admissibility criteria. It constrains the surveyed shell families
and parameter range; it is not a general no-go theorem for source-first metrics.
