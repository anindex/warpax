"""Hand-synth EinFields Minkowski fixture generator.

Writes a minimal Orbax checkpoint whose state carries a single
``eta_metric`` key (the Minkowski 4-metric eta_{ab}). The
:func:`warpax.io.load_einfield` function picks up this key and returns
an :class:`InterpolatedADMMetric` of Minkowski spacetime.

Run this script once (with ``warpax[einfields]`` installed) to rebuild
the checkpoint at ``tests/fixtures/einfields/minkowski.ckpt``.

.. code-block:: bash

    pip install 'warpax[einfields]'
    python tests/fixtures/einfields/generate_minkowski_ckpt.py

The generated checkpoint (~28 KB) is committed, so this only needs re-running
if the Orbax checkpoint format drifts.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    try:
        import orbax.checkpoint as ocp  # type: ignore[import-not-found]
    except ImportError as err:
        raise SystemExit(
            "orbax-checkpoint is not installed. Install via "
            "`pip install 'warpax[einfields]'` and re-run."
        ) from err

    out_dir = Path(__file__).parent / "minkowski.ckpt"
    # Clean any prior content, including a stale orbax tmp dir from an
    # interrupted save.
    for stale in (out_dir, out_dir.with_suffix(".ckpt.orbax-checkpoint-tmp")):
        if stale.is_dir():
            shutil.rmtree(stale)

    eta = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float64)
    state = {"eta_metric": eta}

    # The save is asynchronous under the hood; wait for it to finalize
    # before the interpreter shuts down, or the checkpoint is left as an
    # unfinalized tmp dir.
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(out_dir.absolute(), state, force=True)
    checkpointer.wait_until_finished()
    checkpointer.close()
    print(f"Wrote Minkowski fixture to {out_dir}")


if __name__ == "__main__":
    main()
