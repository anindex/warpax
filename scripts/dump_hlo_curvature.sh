#!/usr/bin/env bash
# Dump HLO text for the Alcubierre curvature chain.
#
# Pins JAX_PLATFORMS=cpu for reproducibility: the canonical HLO artifact
# used to validate the paper's Sec. 2.1 'single-pass curvature chain' claim is
# CPU-stable. GPU HLO drifts with CUDA driver version and is excluded from
# the committed reference artifact.
#
# Artifact: output/hlo/alcubierre_curvature_chain.hlo

set -euo pipefail

# Resolve repo root robustly regardless of invocation directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export JAX_PLATFORMS=cpu

mkdir -p output/hlo/

python warpax/scripts/dump_hlo_curvature.py \
    --output output/hlo/alcubierre_curvature_chain.hlo
