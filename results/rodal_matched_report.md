# Rodal Matched-Parameter Feasibility Report

**Date:** 2026-04-18T14:04:26Z

**Script:** `scripts/run_rodal_matched_resolution.py`

**Parameters:** v_s=0.5, R=1.0, sigma=8.0, domain=[-3,3]^3

## Verdict

**NOT FEASIBLE**: f_miss is not stable within +/-5% across resolutions N=30, 50, 70.

## Per-Resolution Results

| N | n_total | NEC_miss% | WEC_miss% | SEC_miss% | DEC_miss% | Type_I_pct | Time (s) |
|--:|--------:|----------:|----------:|----------:|----------:|-----------:|---------:|
| 30 | 27000 | 0.77 | 13.93 | 28.06 | 26.84 | 100.0 | 36.4 |
| 50 | 125000 | 0.88 | 14.41 | 27.80 | 27.32 | 100.0 | 109.9 |
| 70 | 343000 | 0.92 | 14.59 | 27.84 | 27.53 | 100.0 | 277.3 |

## Stability Analysis

| Condition | Stable | Mean f_miss | Max Deviation |
|-----------|--------|------------:|--------------:|
| NEC | No | 0.8554 | 0.0994 |
| WEC | Yes | 14.3100 | 0.0268 |
| SEC | Yes | 27.8992 | 0.0057 |
| DEC | Yes | 27.2309 | 0.0142 |

## Note for Paper

Rodal at matched parameters (R=1.0, sigma=8.0) exhibits unstable f_miss for NEC across resolutions. Report Rodal at native parameters (R=100, sigma=0.03) in the main comparison table with a comparability caveat.
