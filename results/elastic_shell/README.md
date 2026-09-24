# Elastic-shell examples and data

Data for *Relativistic elastic shells: material support and cavity geometry*,
by An T. Le. These files belong to the `warpax` repository and are not an
arXiv ancillary attachment.

From the repository root, install the solver extra and run
[example 11](../../examples/11_elastic_shell.py):

```sh
pip install -e ".[solver]"
python examples/11_elastic_shell.py
```

The example regenerates the three CSV profiles and `numerics.json` in this
directory. It reads `centrifugal_response.json`, whose exact coefficients
solve the six free-traction equations in the article, and compares them
with independent numerical boundary-value solutions. The example runs
without a manuscript checkout. The [shell reproduction guide](../../docs/how-to/reproduce_warpshell_paper.md)
describes the current calculation and its relation to earlier shell models.

The files contain numerical equilibrium and frame-dragging profiles,
exact centrifugal-response coefficients, and numerical comparisons.
The article supplies the constitutive law, boundary conditions, and
derivations. CSV files have one header row and decimal values; JSON files
use descriptive keys. Fractions stored as JSON strings are exact rational
numbers, readable with Python's `fractions.Fraction`.

## Units and parameters

The numerical profiles use `G=c=1`. For a physical length unit `L0`, multiply
lengths by `L0`, geometrized masses by `c² L0/G_N`, energy densities and
pressures by `c⁴/(G_N L0²)`, and angular frequencies by `c/L0`. Here `G_N`
is Newton's constant in physical units. Lapse, strain, and speed in units
of `c` are dimensionless.

The elastic benchmark has reference radii `(A,B)=(10,20)`, relaxed energy
density `m0=1e-6`, bulk modulus `kappa=m0/5`, and shear modulus `mu0=m0/10`.
Thus `beta=B/A=2`, `k=kappa/m0=1/5`, `mu=mu0/m0=1/10`, and
`chi=G*m0*A²=1e-4`. Rotation is parametrized by
`epsilonOmega=(A*varpi)²`, where `varpi` is the material angular velocity
relative to time normalized at infinity.

## Static elastic shell

`elastic_static.csv` contains 1001 samples through the material.

| Column | Meaning |
|---|---|
| `R` | Reference material radius, from `A` to `B`. |
| `r` | Physical areal radius. Its endpoint values are the free-face radii `a,b`. |
| `m` | Enclosed geometrized mass; `m(a)=0`, `m(b)=M`. |
| `n_r`, `n_t` | Radial and tangential principal particle densities. The corresponding proper stretches are `1/n_r` and `1/n_t=r/R`. |
| `rho` | Energy density in the material rest frame. |
| `p_r`, `p_t` | Radial and tangential pressure, positive in compression. |
| `Phi_unnormalized` | Lapse potential with its inner value set to zero. |

Normalize the last column before calculating clock rates:

```python
Phi = Phi_unnormalized - Phi_unnormalized[-1] + 0.5*log(1 - 2*M/b)
alpha = exp(Phi)
```

The cavity lapse is the inner value of `alpha`; the exterior lapse is
`sqrt(1-2*M/r)`. The benchmark has `a≈9.98529313`, `b≈19.98317724`,
`M≈0.02929678149`, and cavity lapse `≈0.99811338`. Figure 1 uses these
profiles. Floating-point residuals can leave face pressures very close to,
rather than exactly, zero.

## Linear frame dragging

`elastic_dragging.csv` uses the same elastic benchmark.

| Column | Meaning |
|---|---|
| `r` | Areal radius. |
| `omega_over_varpi` | Inertial-frame angular velocity divided by material angular velocity. |
| `domega_dr_over_varpi` | Its radial derivative, in inverse length units. |
| `region` | `0`: vacuum cavity; `1`: elastic material; `2`: vacuum exterior. |

The cavity ratio is `≈0.00251496`; outside the shell,
`omega/varpi=2*(J/varpi)/r³` with `J/varpi≈5.18585` in cubed length units.
These are first-order coefficients about the spherical equilibrium.

## Prescribed tangential support

`static_counterexample.csv` belongs to the separate anisotropic shell in
Appendix A, with `(a,b,M)=(10,20,1)`. It is not the elastic benchmark.
The columns `r,m,rho,p_r,p_t` have the meanings above, with `p_r=0` and
`p_t=rho*m/[2*(r-2*m)]`. `DEC_slack=rho-p_t` is the slack in the dominant
energy condition for this profile. `circular_speed_squared=m/(r-2*m)` is
the squared local orbital speed used in the circular-orbit interpretation.

## Centrifugal response and cavity observables

`centrifugal_response.json` contains the leading centrifugal displacement
at `G=0` and the resulting leading weak-gravity cavity moments. The keys
`benchmark` and `reversed_tide` give the two materials used in the article;
`cases` contains 28 discrete parameter triples, including those two.
`reversed_tide` has `beta=2` and `k=mu=2/5`. Both materials are causal near
relaxation; the article's explicit finite strain box concerns the benchmark.

For each case:

| Key | Meaning |
|---|---|
| `beta`, `bulk_ratio`, `shear_ratio` | `B/A`, `k`, and `mu`. |
| `coefficients` | `[h0,j0,a2,b2,c2,d2]`, in the general profiles of the appendix "Centrifugal response coefficients". |
| `flattening_coefficients` | `[F(1),F(beta)]`; multiply by `epsilonOmega` for leading inner and outer fractional flattenings. |
| `divergence_moment` | `C=integral(D2/s, s=1..beta)`. |
| `active_moment_parts` | `[T, -3*k*C, -(1-k)*(beta²-1)/2]`, the displacement, pressure-trace, and motion terms. |
| `face_bulk_moment_parts` | `[2*(F(1)-F(beta))/3, -(1+3*k)*C, -(1-k)*(beta²-1)/2]`, the bars in Figure 2. |
| `minus_q_over_pi_G_m0_A2_Omega2` | `Qc/pi`, where the leading tidal coefficient is `q=-G*m0*A²*varpi²*Qc`. |
| `clock_shift_over_pi_chi_epsilon` | `Zc/pi`, where the leading central lapse change is `chi*epsilonOmega*Zc`. |

Here `s=R/A`, `D2=U2'+2*U2/s-6*V2/s`, and `F(s)=-3*U2/(2*s)`.
Either three-term partition sums to `5*Qc/(4*pi)`; its separate terms
depend on the chosen reference material coordinates. The resulting tide
is the physical observable. In a local orthonormal frame its leading
electric Weyl tensor is `diag(-q,-q,2*q)`, and relative free-fall
acceleration is minus this tensor times the separation vector.

Positive `Zc` means that rotation makes the central clock run faster than
in the static shell with the same material content. The benchmark has
`Zc=16472*pi/2625`; the second material has `Zc=-15929*pi/7350`.
Both gain total mass at leading rotational order.

These are asymptotic coefficients, with no finite-rotation remainder bound
or enclosure of a continuous parameter range. Figure 2 exaggerates the
shape at `epsilonOmega=0.02`: the benchmark's leading flattenings are
about 35% and 17%. That drawing is not a controlled finite-strain solution.

## Numerical comparisons

`numerics.json` records calculations at the parameters above. `elastic`
lists shooting solutions at three tolerances; `elastic_collocation`
compares them with a separate boundary-value solution. `mass_and_clock`
contains ADM and Komar masses, material energies, and the normalized lapse.
`linear_rotation` and `weak_cavity_comparison` give the dragging response
and its weak-gravity comparison. `centrifugal_collocation` compares the
28 exact coefficient sets with numerical displacement and moment integrals.
The remaining entries concern isotropic inner pressure, the prescribed
tangential shell, the radial mass variation, and compatible radial data.

Tolerances, sampled residuals, and relative errors describe numerical
comparisons. In particular, `radial_diagnostic` is a sampled sufficient
criterion in inverse squared length units, not a continuum stability bound.
The article gives the analytical arguments and the scope of its stability
results. No time evolution is supplied in these files. Figure 3 uses the
exact coordinate transformation in the article at `v/c=0.4` and does not
require an additional numerical dataset.
