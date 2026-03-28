"""Generate the ET-compatible Minkowski slice fixture (D-03 hand-synth).

Run once to populate ``tests/fixtures/cactus/minkowski_slice.h5``:

.. code-block:: bash

    python tests/fixtures/cactus/generate_minkowski_slice.py

Orientation convention (ARCH-3 pin):

- Arrays are written in C-order with shape ``(nz, ny, nx)``.
- Grid axes: ``x`` = innermost (contiguous), ``z`` = outermost.
- This matches the ET ASC output conventions.
- :func:`warpax.io.load_cactus_slice` transposes to ``(nx, ny, nz)``
  for warpax consistency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def main(
    out_path: Path | None = None,
    nx: int = 8,
    ny: int = 8,
    nz: int = 8,
) -> Path:
    if out_path is None:
        out_path = Path(__file__).parent / "minkowski_slice.h5"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import h5py
    except ImportError as err:
        raise SystemExit(
            "generate_minkowski_slice.py requires h5py. "
            "Install via `pip install 'warpax[interop]'`."
        ) from err

    # Minkowski: alpha = 1, beta = 0, gamma = eye(3) at every point.
    # Shape convention (ARCH-3 pin): (nz, ny, nx).
    ones_zyx = np.ones((nz, ny, nx), dtype=np.float64)
    zeros_zyx = np.zeros((nz, ny, nx), dtype=np.float64)

    if out_path.exists():
        out_path.unlink()
    with h5py.File(out_path, "w") as f:
        grp = f.create_group("ITERATION=0").create_group("TIMELEVEL=0")
        grp.create_dataset("alp", data=ones_zyx)
        grp.create_dataset("betax", data=zeros_zyx)
        grp.create_dataset("betay", data=zeros_zyx)
        grp.create_dataset("betaz", data=zeros_zyx)
        grp.create_dataset("gxx", data=ones_zyx)
        grp.create_dataset("gxy", data=zeros_zyx)
        grp.create_dataset("gxz", data=zeros_zyx)
        grp.create_dataset("gyy", data=ones_zyx)
        grp.create_dataset("gyz", data=zeros_zyx)
        grp.create_dataset("gzz", data=ones_zyx)
        grp.attrs["time"] = np.float64(0.0)
        grp.attrs["x0"] = np.float64(-1.0)
        grp.attrs["y0"] = np.float64(-1.0)
        grp.attrs["z0"] = np.float64(-1.0)
        grp.attrs["dx"] = np.float64(2.0 / (nx - 1))
        grp.attrs["dy"] = np.float64(2.0 / (ny - 1))
        grp.attrs["dz"] = np.float64(2.0 / (nz - 1))

    return out_path


if __name__ == "__main__":
    out = main()
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
