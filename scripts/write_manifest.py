"""Write, or check, the integrity manifest for the cached ``results/*.npz`` grids.

``.gitignore`` excludes ``results/**/*.npz``, so roughly 800 MB of computed grids are
not under version control and reach a reader only through the archived release. This
manifest is the only record that lets that reader tell whether the files they have are
the files the paper was written from.

It had drifted: the committed copy was four months old, listed two files that no
longer exist, and gave a size for every large grid that disagreed with the file on
disk, which made all twenty-four of its SHA256 values wrong. A stale integrity record
is worse than none, so this script both writes it and checks it, and the check is
wired into ``check_paper_numbers.py`` so the two cannot drift apart again.

    python scripts/write_manifest.py            # rewrite results/MANIFEST.txt
    python scripts/write_manifest.py --check    # exit non-zero if it is stale
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
MANIFEST = RESULTS / "MANIFEST.txt"

_HEADER = """# warpax/results/ MANIFEST - SHA256 + size per cache file.
# Generated {stamp}
# Regen recipe: JAX_PLATFORMS=cpu bash reproduce_all.sh --stage core
# Per-file recipe: python scripts/run_analysis.py --metric <M> --v_s <V> --deterministic
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_body(results_dir: Path = RESULTS) -> list[str]:
    """One ``sha256  size  name`` line per cached grid, sorted by name."""
    return [
        f"{_sha256(p)}  {p.stat().st_size:>12}  {p.name}" for p in sorted(results_dir.glob("*.npz"))
    ]


def write(results_dir: Path = RESULTS) -> Path:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    body = manifest_body(results_dir)
    (results_dir / "MANIFEST.txt").write_text(
        _HEADER.format(stamp=stamp) + "\n" + "\n".join(body) + "\n"
    )
    return results_dir / "MANIFEST.txt"


def check(results_dir: Path = RESULTS) -> list[str]:
    """Return a list of discrepancies; empty means the manifest is current."""
    path = results_dir / "MANIFEST.txt"
    if not path.exists():
        return [f"{path} is missing"]

    listed: dict[str, tuple[str, int]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            return [f"malformed manifest line: {line!r}"]
        listed[parts[2]] = (parts[0], int(parts[1]))

    problems = []
    on_disk = {p.name: p for p in results_dir.glob("*.npz")}
    for name in sorted(set(listed) - set(on_disk)):
        problems.append(f"listed but absent: {name}")
    for name in sorted(set(on_disk) - set(listed)):
        problems.append(f"present but unlisted: {name}")
    for name in sorted(set(listed) & set(on_disk)):
        want_hash, want_size = listed[name]
        size = on_disk[name].stat().st_size
        if size != want_size:
            problems.append(f"{name}: size {size} != manifest {want_size}")
        elif _sha256(on_disk[name]) != want_hash:
            problems.append(f"{name}: sha256 differs from manifest")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true", help="verify instead of rewriting; non-zero exit if stale"
    )
    args = ap.parse_args()

    if args.check:
        problems = check()
        if problems:
            print("results/MANIFEST.txt is stale:")
            for p in problems:
                print(f"  {p}")
            return 1
        print(f"results/MANIFEST.txt is current ({len(manifest_body())} files)")
        return 0

    path = write()
    print(f"wrote {path} ({len(manifest_body())} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
