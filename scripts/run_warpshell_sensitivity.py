
"""WarpShell clamping sensitivity analysis.

Determines whether missed WEC/DEC violations in the WarpShell metric
are genuine physics or artifacts of hard clamping at shell boundaries.

Strategy:
  1. Load cached WarpShell results for all velocities.
  2. Reconstruct grid coordinates and compute radial distance.
  3. Create boundary masks: points within +/- delta of R_1, R_2.
  4. Recompute missed-violation fractions excluding boundary bands.
  5. Report whether fractions are stable or collapse to near-zero.

Usage:
    python scripts/run_warpshell_sensitivity.py
"""
from __future__ import annotations

import os

import numpy as np

# WarpShell parameters (must match run_analysis.py)
R_1 = 0.5    # Inner shell radius
R_2 = 1.0    # Outer shell radius
R_b = 1.0    # Buffer zone width

# Grid spec (must match run_analysis.py GRID_STANDARD)
BOUNDS = [(-5.0, 5.0)] * 3
SHAPE = (50, 50, 50)

V_S_VALUES = [0.1, 0.5, 0.9, 0.99]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Exclusion band widths (in coordinate units) to test
DELTA_VALUES = [0.05, 0.1, 0.2, 0.5]

CONDITIONS = ["nec", "wec", "sec", "dec"]


def make_grid_coords(bounds, shape):
    """Create 3D grid coordinate arrays matching GridSpec."""
    axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, shape)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return X, Y, Z


def compute_radial_distance(X, Y, Z, v_s, t=0.0):
    """Radial distance from WarpShell bubble center at time t."""
    x_rel = X - v_s * t
    return np.sqrt(x_rel**2 + Y**2 + Z**2)


def analyze_with_exclusion(data, r_grid, delta, conditions=CONDITIONS):
    """Recompute missed-violation stats excluding boundary bands.

    Excludes points where r is within +/- delta of R_1 or R_2.
    Also excludes the extended boundaries R_1-R_b and R_2+R_b.
    """
    # Boundary regions to exclude
    boundaries = [R_1, R_2, R_1 - R_b, R_2 + R_b]
    # Remove negative boundaries (R_1 - R_b = -0.5 is physically meaningless
    # since r >= 0, but keep for completeness)

    # Build exclusion mask: True = KEEP this point
    keep = np.ones(r_grid.shape, dtype=bool)
    for b in boundaries:
        keep &= np.abs(r_grid - b) > delta

    n_total = r_grid.size
    n_kept = np.sum(keep)

    results = {"n_total": n_total, "n_kept": int(n_kept), "delta": delta}

    for cond in conditions:
        eul_key = f"{cond}_eulerian"
        rob_key = f"{cond}_robust"
        if eul_key not in data or rob_key not in data:
            continue

        eul = data[eul_key].reshape(SHAPE)
        rob = data[rob_key].reshape(SHAPE)

        # Full-grid stats
        missed_full = (eul >= 0) & (rob < -1e-10)
        n_missed_full = np.sum(missed_full)

        # Filtered stats (excluding boundary bands)
        eul_filt = eul[keep]
        rob_filt = rob[keep]
        missed_filt = (eul_filt >= 0) & (rob_filt < -1e-10)
        n_missed_filt = np.sum(missed_filt)

        # Where do missed violations fall?
        missed_in_boundary = np.sum(missed_full & ~keep)
        missed_outside_boundary = np.sum(missed_full & keep)

        results[cond] = {
            "pct_missed_full": float(n_missed_full / n_total * 100),
            "pct_missed_filtered": float(n_missed_filt / n_kept * 100) if n_kept > 0 else 0.0,
            "n_missed_full": int(n_missed_full),
            "n_missed_in_boundary": int(missed_in_boundary),
            "n_missed_outside": int(missed_outside_boundary),
            "frac_missed_at_boundary": float(missed_in_boundary / n_missed_full) if n_missed_full > 0 else 0.0,
        }

    return results


def radial_profile(data, r_grid, cond="wec", n_bins=50):
    """Bin missed violations by radial distance for spatial analysis."""
    eul = data[f"{cond}_eulerian"].reshape(SHAPE)
    rob = data[f"{cond}_robust"].reshape(SHAPE)
    missed = (eul >= 0) & (rob < -1e-10)

    r_flat = r_grid.ravel()
    missed_flat = missed.ravel()

    r_max = np.max(r_flat)
    bin_edges = np.linspace(0, min(r_max, 5.0), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    missed_count = np.zeros(n_bins)
    total_count = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (r_flat >= bin_edges[i]) & (r_flat < bin_edges[i + 1])
        total_count[i] = np.sum(mask)
        missed_count[i] = np.sum(missed_flat[mask])

    missed_frac = np.where(total_count > 0, missed_count / total_count, 0.0)

    return bin_centers, missed_frac, missed_count, total_count


def main():
    X, Y, Z = make_grid_coords(BOUNDS, SHAPE)

    print("=" * 78)
    print("WarpShell Clamping Sensitivity Analysis")
    print(f"Shell boundaries: R_1={R_1}, R_2={R_2}, R_b={R_b}")
    print(f"Clamping points: R_1={R_1}, R_2={R_2}, R_1-R_b={R_1-R_b}, R_2+R_b={R_2+R_b}")
    print("=" * 78)

    for v_s in V_S_VALUES:
        cache_path = os.path.join(RESULTS_DIR, f"warpshell_vs{v_s}.npz")
        if not os.path.exists(cache_path):
            print(f"\n--- v_s={v_s}: SKIPPED (no cached data) ---")
            continue

        data = np.load(cache_path)
        r_grid = compute_radial_distance(X, Y, Z, v_s, t=0.0)

        print(f"\n{'='*78}")
        print(f"v_s = {v_s}")
        print(f"{'='*78}")

        # 1. Sensitivity across exclusion band widths
        print(f"\n  {'delta':>6s} | {'Cond':>4s} | {'Full%':>7s} | {'Filt%':>7s} | "
              f"{'#Missed':>8s} | {'#AtBdry':>8s} | {'#Outside':>8s} | {'%AtBdry':>8s}")
        print(f"  {'-'*6}-+-{'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

        for delta in DELTA_VALUES:
            res = analyze_with_exclusion(data, r_grid, delta)
            for cond in CONDITIONS:
                if cond not in res:
                    continue
                c = res[cond]
                if c["n_missed_full"] == 0:
                    continue
                print(
                    f"  {delta:6.2f} | {cond.upper():>4s} | "
                    f"{c['pct_missed_full']:7.3f} | {c['pct_missed_filtered']:7.3f} | "
                    f"{c['n_missed_full']:8d} | {c['n_missed_in_boundary']:8d} | "
                    f"{c['n_missed_outside']:8d} | {c['frac_missed_at_boundary']:8.3f}"
                )

        # 2. Radial profile for WEC
        print(f"\n  Radial profile of missed WEC violations:")
        print(f"  {'r_center':>8s} | {'#missed':>8s} | {'#total':>8s} | {'frac':>8s} | note")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+------")

        bin_centers, missed_frac, missed_count, total_count = radial_profile(
            data, r_grid, cond="wec", n_bins=40
        )
        for i, (rc, mf, mc, tc) in enumerate(
            zip(bin_centers, missed_frac, missed_count, total_count)
        ):
            if mc > 0 or (rc > R_1 - 0.3 and rc < R_2 + 0.3):
                note = ""
                if abs(rc - R_1) < 0.15:
                    note = "<-- R_1"
                elif abs(rc - R_2) < 0.15:
                    note = "<-- R_2"
                elif abs(rc - (R_1 - R_b)) < 0.15:
                    note = "<-- R_1-R_b"
                elif abs(rc - (R_2 + R_b)) < 0.15:
                    note = "<-- R_2+R_b"
                print(
                    f"  {rc:8.3f} | {mc:8.0f} | {tc:8.0f} | {mf:8.4f} | {note}"
                )

        # 3. DEC radial profile (DEC shows larger missed fractions)
        print(f"\n  Radial profile of missed DEC violations:")
        print(f"  {'r_center':>8s} | {'#missed':>8s} | {'#total':>8s} | {'frac':>8s} | note")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+------")

        bin_centers, missed_frac, missed_count, total_count = radial_profile(
            data, r_grid, cond="dec", n_bins=40
        )
        for i, (rc, mf, mc, tc) in enumerate(
            zip(bin_centers, missed_frac, missed_count, total_count)
        ):
            if mc > 0 or (rc > R_1 - 0.3 and rc < R_2 + 0.3):
                note = ""
                if abs(rc - R_1) < 0.15:
                    note = "<-- R_1"
                elif abs(rc - R_2) < 0.15:
                    note = "<-- R_2"
                elif abs(rc - (R_1 - R_b)) < 0.15:
                    note = "<-- R_1-R_b"
                elif abs(rc - (R_2 + R_b)) < 0.15:
                    note = "<-- R_2+R_b"
                print(
                    f"  {rc:8.3f} | {mc:8.0f} | {tc:8.0f} | {mf:8.4f} | {note}"
                )

    print("\n" + "=" * 78)
    print("INTERPRETATION GUIDE:")
    print("  - If %AtBdry is high (>50%): violations are boundary artifacts")
    print("  - If %AtBdry is low and Filt% ~ Full%: violations are genuine physics")
    print("  - Radial profile shows spatial distribution of missed violations")
    print("=" * 78)


if __name__ == "__main__":
    main()
