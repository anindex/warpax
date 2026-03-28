"""Integration test: solver='generalized' stabilizes WarpShell v_s=0.5
classification across perturbation seeds near |λ| ~ 10^42, and the
Bauer-Fike cond_V diagnostic returns valid results.
"""
from __future__ import annotations

from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from warpax.energy_conditions.classification import classify_hawking_ellis
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.metrics import WarpShellMetric


# 10 fixed seeds for the perturbation ensemble.
N_SEEDS = 10
SEEDS = [20260418 + i for i in range(N_SEEDS)]


@pytest.mark.slow
class TestGeneralizedSolverStability:
    """Integration: WarpShell idx=8 10-seed stability + cond_V diagnostic."""

    @pytest.fixture(scope="class")
    def warpshell_idx_8_inputs(self):
        """Resolve the WarpShell v_s=0.5 idx=8 flat index.

        Builds the 50^3 WarpShell grid, classifies under solver='standard'
        (the v0.2.0 path), collects all Type-IV flat indices, samples 200 with
        seed=20260418, sorts, and picks the 9th element (0-indexed: sampled[8]).
        Returns T_ab, g_ab, T_mixed, and the flat index for that point.
        """
        metric = WarpShellMetric(v_s=0.5)
        grid = GridSpec(
            bounds=((-12.0, 12.0), (-6.0, 6.0), (-6.0, 6.0)),
            shape=(50, 50, 50),
        )
        chain = evaluate_curvature_grid(metric, grid)
        T_flat = np.asarray(chain.stress_energy.reshape(-1, 4, 4))
        g_flat = np.asarray(chain.metric.reshape(-1, 4, 4))
        g_inv_flat = np.asarray(chain.metric_inv.reshape(-1, 4, 4))
        T_mixed_flat = np.einsum('nab,nbc->nac', g_inv_flat, T_flat)

        classify_v = jax.vmap(classify_hawking_ellis, in_axes=(0, 0))
        he_types = np.asarray(
            classify_v(jnp.asarray(T_mixed_flat), jnp.asarray(g_flat)).he_type
        ).astype(int)
        type_iv_flat = np.where(he_types == 4)[0]
        rng = np.random.default_rng(seed=20260418)
        sampled = np.sort(
            rng.choice(
                type_iv_flat, size=min(200, len(type_iv_flat)), replace=False
            )
        )
        if len(sampled) < 9:
            pytest.skip(
                f"WarpShell idx=8 requires ≥9 Type-IV points; got {len(sampled)}"
            )
        idx_8_flat = int(sampled[8])
        return {
            'T_ab': T_flat[idx_8_flat],
            'g_ab': g_flat[idx_8_flat],
            'T_mixed': T_mixed_flat[idx_8_flat],
            'idx_8_flat': idx_8_flat,
        }

    def test_standard_solver_baseline(self, warpshell_idx_8_inputs):
        """Baseline: document solver='standard' instability via printed Counter.

        This is a characterisation test - NO hard assertion. It records the
        pre-fix behavior where perturbation causes Type-IV/II classification
        flip-counts.
        """
        T_ab = warpshell_idx_8_inputs['T_ab']
        g_ab = warpshell_idx_8_inputs['g_ab']
        T_mixed = warpshell_idx_8_inputs['T_mixed']

        he_types = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed=seed)
            eps = rng.standard_normal(T_mixed.shape) * np.finfo(np.float64).eps
            T_mixed_pert = jnp.asarray(T_mixed + eps)
            r = classify_hawking_ellis(T_mixed_pert, jnp.asarray(g_ab))
            he_types.append(int(r.he_type))
        dist = Counter(he_types)
        print(f"\n[BASELINE standard] idx=8 he_type across {N_SEEDS} seeds: {dist}")

    def test_generalized_solver_stable(self, warpshell_idx_8_inputs):
        """Contract: solver='generalized' stable across 10 perturbation seeds.

        The modal he_type count MUST equal N_SEEDS (i.e. all 10 seeds classify
        the same way).
        """
        T_ab = warpshell_idx_8_inputs['T_ab']
        g_ab = warpshell_idx_8_inputs['g_ab']
        T_mixed = warpshell_idx_8_inputs['T_mixed']

        he_types = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed=seed)
            eps = rng.standard_normal(T_mixed.shape) * np.finfo(np.float64).eps
            T_mixed_pert = jnp.asarray(T_mixed + eps)
            eps_ab = rng.standard_normal(T_ab.shape) * np.finfo(np.float64).eps
            T_ab_pert = jnp.asarray(T_ab + eps_ab)
            r = classify_hawking_ellis(
                T_mixed_pert, jnp.asarray(g_ab),
                solver='generalized', T_ab=T_ab_pert,
            )
            he_types.append(int(r.he_type))
        dist = Counter(he_types)
        print(f"\n[CONTRACT generalized] idx=8 he_type across {N_SEEDS} seeds: {dist}")

        total = sum(dist.values())
        modal_count = max(dist.values())
        n_deviant = total - modal_count
        assert n_deviant < 1, (
            f"solver='generalized' must be stable across {N_SEEDS} seeds; "
            f"got distribution {dict(dist)} with n_deviant={n_deviant}"
        )

    def test_cond_v_diagnostic_functional(self, warpshell_idx_8_inputs):
        """Bauer-Fike cond_V diagnostic returns valid finite results.

        The mpmath classifier must return a finite cond_V and a boolean
        uncertain flag for the WarpShell hard-boundary point.
        """
        from warpax.energy_conditions.classification_mpmath import (
            classify_hawking_ellis_mpmath,
        )

        T_mixed = np.asarray(warpshell_idx_8_inputs['T_mixed'])
        g_ab = np.asarray(warpshell_idx_8_inputs['g_ab'])

        result = classify_hawking_ellis_mpmath(T_mixed, g_ab)
        uncertain = bool(result['uncertain'])
        cond_V = float(result['cond_V'])
        print(
            f"\n[cond_V diagnostic] idx=8 uncertain={uncertain}, "
            f"cond_V={cond_V:.3e}"
        )
        assert np.isfinite(cond_V), f"cond_V must be finite; got {cond_V}"
        assert isinstance(uncertain, bool), (
            f"uncertain must be bool; got {type(uncertain)}"
        )
