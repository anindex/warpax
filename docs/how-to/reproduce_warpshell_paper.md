# Relativistic elastic shells: numerical examples

The paper [Relativistic elastic shells: material support and cavity
geometry](https://arxiv.org/abs/2605.25417v3) uses one constitutive action to determine the material support,
rotation response and cavity observables of an elastic annulus with two free
vacuum faces. Its numerical example and datasets live in this repository.
The manuscript contains the assumptions and theoretical derivations; its
arXiv package needs only the LaTeX source, style and figures.

## Run the elastic-shell example

From a repository checkout:

```sh
pip install -e ".[solver]"
python examples/11_elastic_shell.py
```

[Example 11](https://github.com/anindex/warpax/blob/main/examples/11_elastic_shell.py)
uses NumPy, SciPy and SymPy. It runs without SageMath, JAX compilation or a
sibling manuscript directory. Output paths are relative to the checkout,
so the command also works when invoked by its absolute path.

The calculation includes nonlinear static shooting and independent
collocation, ADM and Komar mass integrals, the cavity redshift, first-order
frame dragging, and leading centrifugal displacement and cavity moments.
It also evaluates the sampled radial stability criterion, the constrained
mass variation, and compatible radial initial data. Assertions compare the
solutions and enforce the stated numerical tolerances.

The [data guide](https://github.com/anindex/warpax/blob/main/results/elastic_shell/README.md)
defines units, parameters, columns and normalizations. All files are under
`results/elastic_shell/`:

| File | Contents |
|---|---|
| `elastic_static.csv` | Material strains, stresses, geometry and unnormalized lapse through the elastic shell |
| `elastic_dragging.csv` | Linear dragging through the cavity, material and exterior |
| `static_counterexample.csv` | Separate prescribed anisotropic shell from Appendix A |
| `centrifugal_response.json` | Exact leading coefficients for 28 discrete choices of thickness and moduli |
| `numerics.json` | Numerical equilibrium, mass, clock, rotation and radial comparisons |

The example regenerates the CSV files and `numerics.json`. It reads the
exact centrifugal coefficients supplied with the checkout and compares all
28 cases with independent boundary-value solutions. The coefficients come
from the six free-traction equations in the article's appendix
*Centrifugal response coefficients*. They are also included in the source
distribution so the example retains its required input there.

The two materials at `beta=2` give opposite leading tidal and central-clock
shifts, despite two oblate faces and positive rotational mass increments.
The coefficients describe weak gravity and slow rotation. They do not
provide a finite-rotation strain or remainder bound, full nonspherical
stability, or a nonlinear rotating solution at the numerical benchmark.

## Paper figures and further checks

When `warpshell_arxiv/` is available beside the checkout, its `README.md`
gives the paper build and source-only submission commands. Its
`verify/README.md` describes the additional exact and interval checks.
That suite calls example 11 and generates the three paper figures from
the datasets here. The independent exact calculation also regenerates
`centrifugal_response.json` from the traction equations.

The package's `SShellMetric` and `TShellMetric` constructors implement the
earlier prescriptions below. They do not implement the elastic action or
its rigidly rotating branch. Example 11 solves the revised spherical
Einstein–elastic boundary-value problem and the stated perturbative
responses directly.

## Limits of the S-/T-shell prescriptions

An isotropic static shell with nonnegative density and pressure, a regular empty
cavity, and no surface stress cannot satisfy its inner vacuum boundary. Vacuum
matching requires `p(a)=0`, while the TOV equation makes nonnegative pressure
nonincreasing outward. A nontrivial shell under these assumptions is impossible.
Tangential stress or an elastic constitutive law changes the problem.

The T-shell's scalar radial shift equation is not the full Cartesian-vector
momentum constraint. Even on a flat unit-lapse slice,

$$
\mathcal P_i=\tfrac12(\Delta\beta_i-\partial_i\partial_j\beta^j),\qquad
\beta^i=b(r)\delta_x^i,
$$

has transverse components. Its averaged x component is
`(b''+2b'/r)/3`, without the scalar-harmonic term `-2b/r^2`.
Constraint residuals also do not enforce the spatial Einstein equations or
constitutive stress. A true perfect fluid has a timelike eigenvector under
every subluminal tilt. Type IV in the reconstructed stress therefore cannot
also describe the prescribed perfect fluid.

The old boundary deficits and parameter surveys establish no universal
source–vacuum energy-condition cost. Open-path coordinate-time asymmetries are
not invariant transport observables, and positive finite null integrals do not
establish complete-geodesic ANEC.

The S-/T-shell calculations belong to the earlier
[boundary-cost preprint](https://arxiv.org/abs/2605.25417v2).
The [v1.1.1 software](https://github.com/anindex/warpax/tree/v1.1.1) retains
those experiment scripts. For current results, use example 11 and identify
the metric, material model and approximation order when comparing outputs.
