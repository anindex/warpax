"""Hawking-Ellis classification of stress-energy tensors.

Classifies T^a_b by its eigenvalue structure:
- Type I:  One timelike eigenvector, three spacelike. Diagonalizable with real eigenvalues {-ρ, p₁, p₂, p₃}.
- Type II: One null eigenvector (non-diagonalizable), degenerate eigenvalue.
- Type III: One null eigenvector, even more degenerate (single eigenvalue with multiplicity 4).
- Type IV: No real eigenvalues (complex conjugate pairs).

Type I is by far the most common for physically reasonable matter (perfect fluids, EM fields, etc.).
The fast eigenvalue method applies only to Type I; others require the optimization fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class HawkingEllisType(Enum):
    TYPE_I = 1
    TYPE_II = 2
    TYPE_III = 3
    TYPE_IV = 4


@dataclass
class TypeClassification:
    """Result of Hawking-Ellis type classification at a point."""

    type: HawkingEllisType
    eigenvalues: NDArray  # (4,) complex in general
    eigenvectors: NDArray  # (4, 4) columns are eigenvectors
    rho: float | None = None  # Energy density (Type I only)
    pressures: NDArray | None = None  # (3,) principal pressures (Type I only)


def classify_point(
    T_mixed: NDArray, g_ab: NDArray, tol: float = 1e-10
) -> TypeClassification:
    """Classify the Hawking-Ellis type of T^a_b at a single point.

    Parameters
    ----------
    T_mixed : NDArray
        Mixed stress-energy tensor T^a_b, shape (4, 4).
    g_ab : NDArray
        Metric tensor g_{ab}, shape (4, 4).
    tol : float
        Tolerance for classifying eigenvalues as real/degenerate.

    Returns
    -------
    TypeClassification
        The classification result.
    """
    eigenvalues, eigenvectors = np.linalg.eig(T_mixed)

    # Check if all eigenvalues are real
    imag_parts = np.abs(eigenvalues.imag)
    all_real = np.all(imag_parts < tol)

    if not all_real:
        return TypeClassification(
            type=HawkingEllisType.TYPE_IV,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )

    # All eigenvalues real - check eigenvector causal characters
    evals_real = eigenvalues.real
    evecs = eigenvectors

    # Classify each eigenvector's causal character
    # v^a v_a = g_{ab} v^a v^b
    causal = np.array([
        evecs[:, i].real @ g_ab @ evecs[:, i].real for i in range(4)
    ])

    n_timelike = np.sum(causal < -tol)
    n_null = np.sum(np.abs(causal) <= tol)

    if n_timelike == 1 and n_null == 0:
        # Type I: one timelike eigenvector
        timelike_idx = np.argmin(causal)
        spacelike_idx = [i for i in range(4) if i != timelike_idx]

        # The eigenvalue of the timelike eigenvector is -ρ
        rho = -evals_real[timelike_idx]
        pressures = np.sort(evals_real[spacelike_idx])

        return TypeClassification(
            type=HawkingEllisType.TYPE_I,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            rho=rho,
            pressures=pressures,
        )
    elif n_null >= 1:
        # Check degeneracy to distinguish Type II vs III
        unique_evals = np.unique(np.round(evals_real / max(tol, np.max(np.abs(evals_real)) * tol)))
        if len(unique_evals) == 1:
            return TypeClassification(
                type=HawkingEllisType.TYPE_III,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
            )
        else:
            return TypeClassification(
                type=HawkingEllisType.TYPE_II,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
            )
    else:
        # Fallback: if no timelike eigenvector found but all real,
        # might be near-degenerate Type I. Treat as Type I if possible.
        # Find the most timelike eigenvector
        most_timelike_idx = np.argmin(causal)
        if causal[most_timelike_idx] < 0:
            rho = -evals_real[most_timelike_idx]
            spacelike_idx = [i for i in range(4) if i != most_timelike_idx]
            pressures = np.sort(evals_real[spacelike_idx])
            return TypeClassification(
                type=HawkingEllisType.TYPE_I,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                rho=rho,
                pressures=pressures,
            )

        return TypeClassification(
            type=HawkingEllisType.TYPE_II,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )


def classify_grid(
    T_mixed: NDArray, g_field: NDArray, tol: float = 1e-10
) -> list:
    """Classify Hawking-Ellis type at every grid point.

    Parameters
    ----------
    T_mixed : NDArray
        Mixed stress-energy, shape (*grid_shape, 4, 4).
    g_field : NDArray
        Metric, shape (*grid_shape, 4, 4).
    tol : float
        Classification tolerance.

    Returns
    -------
    list
        Flat list of TypeClassification results (row-major order).
    """
    grid_shape = T_mixed.shape[:-2]
    flat_T = T_mixed.reshape(-1, 4, 4)
    flat_g = g_field.reshape(-1, 4, 4)

    results = []
    for i in range(flat_T.shape[0]):
        results.append(classify_point(flat_T[i], flat_g[i], tol))

    return results
