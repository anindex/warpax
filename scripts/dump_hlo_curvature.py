"""Dump JIT-compiled HLO text for compute_curvature_chain.

Validates the paper's §2.1 'single-pass curvature chain' claim by emitting
the XLA HLO for a single pointwise invocation of ``compute_curvature_chain``
applied to an Alcubierre spacetime. The artifact is deterministic under a
fixed jaxlib version + CPU backend.

Usage
-----
Invoked via the shell wrapper::

    bash warpax/scripts/dump_hlo_curvature.sh

which pins ``JAX_PLATFORMS=cpu`` and writes the artifact to
``output/hlo/alcubierre_curvature_chain.hlo``.

The script is read-only with respect to any ``warpax/src/`` source file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.geometry.geometry import compute_curvature_chain


def dump_hlo(output_path: Path) -> tuple[int, int, int]:
    """Lower + compile + extract HLO text for the Alcubierre curvature chain.

    Uses the canonical JAX ``jit(fn).lower(*args).compile().as_text()``
    pipeline. Pointwise (not grid) is used to keep the artifact compact and
    focused on the algorithmic claim rather than grid vmap boilerplate.

    Returns
    -------
    (total_ops, fusions, kernels) : tuple[int, int, int]
        Rough fusion statistics - substring counts of ``%``, ``fusion``, and
        ``kernel`` in the HLO text.
    """
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
    coords = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)

    jit_fn = jax.jit(compute_curvature_chain)
    compiled = jit_fn.lower(metric, coords).compile()
    hlo_text = compiled.as_text()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(hlo_text)

    total_ops = hlo_text.count("%")
    fusions = hlo_text.count("fusion")
    kernels = hlo_text.count("kernel")
    return total_ops, fusions, kernels


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dump HLO text for compute_curvature_chain ."
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the HLO text artifact.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    total_ops, fusions, kernels = dump_hlo(args.output)
    print(f"HLO: total_ops={total_ops}, fusions={fusions}, kernels={kernels}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
