# Theory: ADM decomposition and energy conditions

warpax uses signature $(-+++)$ and geometric units $G=c=1$.
The [paper](https://arxiv.org/abs/2602.18023) develops the mathematical tests.

## ADM decomposition and curvature

An `ADMMetric` supplies lapse $\alpha$, shift $\beta^i$, and a positive-definite
spatial metric $\gamma_{ij}$. With $\beta_i=\gamma_{ij}\beta^j$,

$$
g_{ab}=\begin{pmatrix}
-\alpha^2+\beta_i\beta^i & \beta_j\\
\beta_i & \gamma_{ij}
\end{pmatrix},\qquad
n^a=\alpha^{-1}(1,-\beta^i).
$$

The unit normal $n^a$ remains timelike wherever the ADM decomposition is valid,
including superluminal bubble speeds. A coordinate-stationary vector
$\partial_t$ need not remain timelike.

`compute_curvature_chain` differentiates the metric automatically and computes

$$
g_{ab}\longrightarrow\Gamma^a{}_{bc}\longrightarrow R^a{}_{bcd}
\longrightarrow R_{ab}\longrightarrow G_{ab}=8\pi T_{ab}.
$$

The returned $T_{ab}=G_{ab}/8\pi$ includes any cosmological contribution as
effective stress. To test material stress with a separately specified
cosmological constant, use $T^{\rm matter}_{ab}=(G_{ab}+\Lambda g_{ab})/8\pi$.
Automatic differentiation avoids finite-difference truncation, not roundoff.

## Hawking-Ellis types

The mixed tensor $T^a{}_b$ has the following canonical structures:

| Type | Eigenstructure | Example or interpretation |
|---|---|---|
| I | Real diagonal form with a timelike eigenvector | Perfect or anisotropic fluid |
| II | Null Jordan block of size 2 | Null dust is a special case |
| III | Null Jordan block of size 3 | No timelike rest frame |
| IV | Complex-conjugate eigenvalue pair | No timelike rest frame |

For Type I, write the rest-frame tensor as $\mathrm{diag}(\rho,p_1,p_2,p_3)$.
NEC requires every $\rho+p_i\ge0$; WEC adds $\rho\ge0$; SEC requires the NEC
inequalities and $\rho+\sum_i p_i\ge0$; DEC requires $\rho\ge|p_i|$.
Along principal axis $i$, an observer at rest-frame rapidity $\zeta$ measures

$$
T(u,u)=\rho+(\rho+p_i)\sinh^2\zeta.
$$

If $\rho\ge0$ and $\rho+p_i<0$, the density becomes negative above
$\sinh^2\zeta=\rho/|\rho+p_i|$ on that axis. This is a directional threshold;
it does not identify a generic optimizer subject to an Eulerian rapidity cap.

## Conditions without classification

In an orthonormal tetrad, let $\eta=\mathrm{diag}(-1,1,1,1)$ and
$q(w)=\hat T_{00}+2\hat T_{0i}w^i+\hat T_{ij}w^iw^j$.
The NEC tests $q\ge0$ on $|w|=1$; WEC tests it on $|w|\le1$.
The corresponding S-lemma conditions are

$$
\hat T+\sigma\eta\succeq0,
$$

with free $\sigma$ for NEC and $\sigma\ge0$ for WEC. SEC applies the ball test
to $T-\tfrac12\mathrm{tr}_g(T)g$. DEC applies it to both $T$ and
$-Tg^{-1}T$ to enforce future-directed causal energy flux.

These Boolean equivalences are independent of algebraic type and observer
frame. The numerical LMI margin and normalized null deficit have magnitudes
that depend on the chosen tetrad. Type-I eigenvalue slacks, Eulerian contractions,
and capped observer-search minima must not be compared as a single scale.
Exact rational certificates, when found, certify the supplied tensor entries;
interval evaluation is needed to include uncertainty in the tensor itself.

## Warp-wall conclusions and limits

The sampled Alcubierre and Natário walls contain Type-IV regions, while the
ideal zero-momentum Rodal reduction and matched Garattini-Zatrimaylov
construction admit Type-I stress. Shift vorticity alone is neither a general
necessary nor sufficient condition for Type IV. Its observed association with
momentum-dominated wall regions depends on the metric and stress block.

For unit lapse, a time-independent Euclidean spatial metric, fixed profiles,
and vanishing Eulerian momentum density, pointwise quadratic speed scaling
can follow exactly from the ADM equations. Other fitted exponents are empirical.
Integrated negative Eulerian energy establishes WEC failure; a separate null
argument is needed for NEC. Finite null segments do not establish a
complete-geodesic averaged null energy condition.

## References

- Hawking and Ellis (1973), *The Large Scale Structure of Space-Time*, §4.3.
- Baumgarte and Shapiro (2010), *Numerical Relativity*, Chapter 2.
- Xia, Wang, and Sheu (2016), *S-lemma with equality and its applications*,
  [Math. Program. 156, 513–547](https://doi.org/10.1007/s10107-015-0907-0).
- [Observer-robust energy condition verification for warp drive spacetimes](https://arxiv.org/abs/2602.18023).
