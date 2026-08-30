# Wall-Restricted Type-IV Analysis Report

**Date:** 2026-08-28T17:48:27Z
**Script:** `scripts/run_wall_restricted_analysis.py`
**Grid resolution:** 50^3 (per metric; bounds follow run_analysis.py)
**Wall region:** shape function in [0.1, 0.9]
**Velocity:** v_s = 0.5

## Overview

Full-domain Type-IV fractions are computed over the full grid, which for large-bubble metrics (Rodal, Lentz) is dominated by vacuum. Restricting to the active warp-wall region where the shape function lies in [0.1, 0.9] yields fractions that are directly physically meaningful. This report shows both quantities side-by-side so the scaling between full-grid (vacuum-dominated) and wall-restricted (wall-dominated) statistics is transparent.

## alcubierre

- Wall points (f in [0.1, 0.9]): 416
- Total grid points: 125000
- Type-IV fraction: full=15.49%, wall=100.00%
- Type I/II/III/IV wall breakdown: 0.00% / 0.00% / 0.00% / 100.00%
- Full-grid miss % (Eulerian satisfied, robust violated): NEC=0.00%, WEC=0.00%, SEC=7.19%, DEC=0.00%
- Wall-restricted conditional miss rate: NEC=0.00%, WEC=0.00%, SEC=15.38%, DEC=0.00%
- Elapsed: 58.3s

## rodal

- Wall points (f in [0.1, 0.9]): 5208
- Total grid points: 125000
- Type-IV fraction: full=0.00%, wall=0.00%
- Type I/II/III/IV wall breakdown: 100.00% / 0.00% / 0.00% / 0.00%
- Full-grid miss % (Eulerian satisfied, robust violated): NEC=1.58%, WEC=15.60%, SEC=28.01%, DEC=28.53%
- Wall-restricted conditional miss rate: NEC=10.13%, WEC=60.78%, SEC=11.60%, DEC=62.67%
- Elapsed: 2.2s

## vdb

- Wall points (f in [0.1, 0.9]): 416
- Total grid points: 125000
- Type-IV fraction: full=5.67%, wall=84.62%
- Type I/II/III/IV wall breakdown: 15.38% / 0.00% / 0.00% / 84.62%
- Full-grid miss % (Eulerian satisfied, robust violated): NEC=0.10%, WEC=0.36%, SEC=1.78%, DEC=0.31%
- Wall-restricted conditional miss rate: NEC=13.33%, WEC=53.33%, SEC=13.46%, DEC=33.33%
- Elapsed: 22.4s

## natario

- Wall points (f in [0.1, 0.9]): 416
- Total grid points: 125000
- Type-IV fraction: full=99.97%, wall=90.38%
- Type I/II/III/IV wall breakdown: 9.62% / 0.00% / 0.00% / 90.38%
- Full-grid miss % (Eulerian satisfied, robust violated): NEC=0.00%, WEC=0.00%, SEC=0.00%, DEC=0.00%
- Wall-restricted conditional miss rate: NEC=0.00%, WEC=0.00%, SEC=0.00%, DEC=0.00%
- Elapsed: 326.7s

## Summary Table

| Metric | Wall points | Full Type-IV | Wall Type-IV | Full DEC miss % | Wall DEC miss | Caveat |
|--------|-------------|--------------|--------------|------------------|----------------|--------|
| alcubierre | 416 | 15.49% | 100.00% | 0.00% | 0.00% |  |
| rodal | 5208 | 0.00% | 0.00% | 28.53% | 62.67% |  |
| vdb | 416 | 5.67% | 84.62% | 0.31% | 33.33% |  |
| natario | 416 | 99.97% | 90.38% | 0.00% | 0.00% |  |
