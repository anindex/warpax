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

The committed ``.gitkeep`` directory stub keeps the path under version
control when the Orbax-populated checkpoint content is absent (CI where
``warpax[einfields]`` is not installed).
"""
from __future__ import annotations

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
    # Clean any prior content.
    if out_dir.exists() and out_dir.is_dir():
        for child in out_dir.glob("*"):
            if child.is_file():
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    eta = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float64)
    state = {"eta_metric": eta}

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(out_dir.absolute(), state, force=True)
    print(f"Wrote Minkowski fixture to {out_dir}")


if __name__ == "__main__":
    main()
