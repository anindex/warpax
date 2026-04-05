# Theory: ADM 3+1 and Hawking-Ellis types

A compact primer on the two physics formalisms that warpax uses everywhere.
Full derivations live in the CQG paper
(`warpax_arxiv/main.tex`, Sections 2–3).

## ADM 3+1 decomposition

Every `ADMMetric` subclass supplies three objects:

- **Lapse** $\alpha(t, \vec{x})$ - a scalar.
- **Shift** $\beta^i(t, \vec{x})$ - a 3-vector.
- **Spatial metric** $\gamma_{ij}(t, \vec{x})$ - a symmetric $3\times 3$
  positive-definite tensor.

The full 4-metric is reconstructed as

$$
g_{ab} \,=\,
\begin{pmatrix}
-\alpha^2 + \beta_i \beta^i & \beta_j \\[4pt]
\beta_i & \gamma_{ij}
\end{pmatrix},
$$

where $\beta_i = \gamma_{ij} \beta^j$ lowers the shift index. This is the
standard Arnowitt-Deser-Misner decomposition; see Baumgarte & Shapiro
(2010) for a textbook treatment.

warpax's `ADMMetric` base class auto-reconstructs $g_{ab}$ from the
user-supplied `lapse`, `shift`, and `spatial_metric` - subclasses never
need to assemble the 4-metric directly.

## Einstein field equations

Given $g_{ab}$, the curvature chain

$$
g_{ab}
\,\xrightarrow{\partial}\, \Gamma^a{}_{bc}
\,\xrightarrow{\partial}\, R^a{}_{bcd}
\,\to\, R_{ab}
\,\to\, R
\,\to\, G_{ab}
\,=\, T_{ab}
$$

is implemented with JAX forward-mode autodiff (no finite differences). The
final step $G_{ab} = 8\pi T_{ab}$ reads the stress-energy tensor straight
off the Einstein tensor in geometric units $G = c = 1$.

See [`compute_curvature_chain`](../reference/index.md) for the full
signature.

## Hawking-Ellis types I–IV

The stress-energy tensor $T^a{}_b$ (mixed index) admits an algebraic
classification based on the structure of its eigendecomposition
(Hawking & Ellis 1973, §4.3):

| Type | Eigenstructure | Physical interpretation |
|------|---------------|-------------------------|
| I | Diagonalizable, 4 real eigenvalues, 1 timelike + 3 spacelike | Perfect fluid; anisotropic fluid |
| II | 2×2 null Jordan block + 2 real eigenvalues | Pure radiation (null dust) |
| III | 3×3 null Jordan block + 1 real eigenvalue | Rare; pathological |
| IV | Complex-conjugate pair of eigenvalues | No real timelike eigenvector |

Type I admits closed-form algebraic energy condition checks on the
eigenvalues. Types II–IV do **not** - they require a continuous search
over the timelike observer manifold, which warpax performs via
Optimistix BFGS over a bounded rapidity parameter.

The observer-robust EC check in warpax thus has two tiers:

1. Classify the point as Type I–IV.
2. For Type II–IV, run the BFGS observer-space optimization (and for
   Type I, run it anyway as a verification - the algebraic check should
   agree with the optimizer's worst-case margin).

See `warpax.energy_conditions.classify_hawking_ellis` and
`warpax.energy_conditions.verify_point` for the implementation.

## Why it matters for warp drives

The Alcubierre-wall region is dominated by **Type IV** stress-energy
points (complex eigenvalue pairs). An Eulerian-frame only analysis misses
a large fraction of true NEC / WEC / DEC violations at these points
because the ADM-normal observer is not the worst-case observer. The
observer-robust margin strictly bounds the Eulerian margin from below
(`robust_margin ≤ eulerian_margin`), and the gap is where physical
violations hide.

## References

- Alcubierre, M. (1994). *The warp drive: hyper-fast travel within
  general relativity*. Class. Quantum Grav. 11, L73.
- Hawking, S. W., & Ellis, G. F. R. (1973). *The large scale structure
  of space-time*. Cambridge University Press. §4.3.
- Baumgarte, T. W., & Shapiro, S. L. (2010). *Numerical Relativity:
  Solving Einstein's Equations on the Computer*. Cambridge University
  Press. Chapter 2.
- `warpax_arxiv/main.tex` - the CQG-115130 submission (Sections 2–5).
