# Wall-Restricted Type-IV Analysis Report

**Date:** 2026-04-18T14:12:02Z
**Script:** `scripts/run_wall_restricted_analysis.py`
**Grid resolution:** 50^3 (per metric; bounds follow run_analysis.py)
**Wall region:** shape function in [0.1, 0.9]
**Velocity:** v_s = 0.5

## Overview

The paper's headline Type-IV fractions are computed over the full grid, which for large-bubble metrics (Rodal, Lentz) is dominated by vacuum. Restricting to the active warp-wall region where the shape function lies in [0.1, 0.9] yields conditional fractions that are directly physically meaningful. This report shows both quantities side-by-side so the scaling between full-grid (vacuum-dominated) and wall-restricted (wall-dominated) statistics is transparent.

## alcubierre

- Wall points (f in [0.1, 0.9]): 416
- Total grid points: 125000
- Type-IV fraction: full=2.05%, wall=98.08%
- Type I/II/III/IV wall breakdown: 1.92% / 0.00% / 0.00% / 98.08%
- Full-grid miss % (Eulerian satisfied, robust violated): NEC=0.00%, WEC=0.00%, SEC=0.32%, DEC=0.00%
- Wall-restricted conditional miss rate: NEC=0.00%, WEC=0.00%, SEC=23.08%, DEC=0.00%
- Elapsed: 70.1s

## rodal

- Wall points (f in [0.1, 0.9]): 5208
- Total grid points: 125000
- Type-IV fraction: full=0.00%, wall=0.00%
- Type I/II/III/IV wall breakdown: 100.00% / 0.00% / 0.00% / 0.00%
- Full-grid miss % (Eulerian satisfied, robust violated): NEC=1.58%, WEC=15.60%, SEC=28.01%, DEC=28.53%
- Wall-restricted conditional miss rate: NEC=10.13%, WEC=60.78%, SEC=11.60%, DEC=62.67%
- Elapsed: 191.3s

## vdb

- Wall points (f in [0.1, 0.9]): 416
- Total grid points: 125000
- Type-IV fraction: full=1.56%, wall=84.62%
- Type I/II/III/IV wall breakdown: 15.38% / 0.00% / 0.00% / 84.62%
- Full-grid miss % (Eulerian satisfied, robust violated): NEC=0.10%, WEC=0.36%, SEC=1.15%, DEC=0.31%
- Wall-restricted conditional miss rate: NEC=13.33%, WEC=53.33%, SEC=13.46%, DEC=33.33%
- Elapsed: 50.9s

## natario

- Wall points (f in [0.1, 0.9]): 416
- Total grid points: 125000
- Type-IV fraction: full=2.46%, wall=90.38%
- Type I/II/III/IV wall breakdown: 9.62% / 0.00% / 0.00% / 90.38%
- Full-grid miss % (Eulerian satisfied, robust violated): NEC=0.00%, WEC=0.00%, SEC=0.00%, DEC=0.00%
- Wall-restricted conditional miss rate: NEC=0.00%, WEC=0.00%, SEC=0.00%, DEC=0.00%
- Elapsed: 190.2s

## lentz

**Caveat:** unresolved lower-bound estimate (44x under-resolved wall at 50^3 over [-300, 300]^3; L1 feature width ~ 2/sigma at sigma = 8).

- Wall points (f in [0.1, 0.9]): 16
- Total grid points: 125000
- Type-IV fraction: full=0.07%, wall=100.00%
- Type I/II/III/IV wall breakdown: 0.00% / 0.00% / 0.00% / 100.00%
- Full-grid miss % (Eulerian satisfied, robust violated): NEC=0.00%, WEC=0.00%, SEC=0.05%, DEC=0.00%
- Wall-restricted conditional miss rate: NEC=0.00%, WEC=0.00%, SEC=0.00%, DEC=0.00%
- Elapsed: 14.1s

## warpshell

- Wall points (f in [0.1, 0.9]): 168
- Total grid points: 125000
- Type-IV fraction: full=0.07%, wall=12.50%
- Type I/II/III/IV wall breakdown: 53.57% / 33.93% / 0.00% / 12.50%
- Full-grid miss % (Eulerian satisfied, robust violated): NEC=0.00%, WEC=0.02%, SEC=0.03%, DEC=0.01%
- Wall-restricted conditional miss rate: NEC=12.90%, WEC=26.47%, SEC=25.81%, DEC=0.00%
- Elapsed: 15.7s

## Summary Table

| Metric | Wall points | Full Type-IV | Wall Type-IV | Full DEC miss % | Wall DEC miss | Caveat |
|--------|-------------|--------------|--------------|------------------|----------------|--------|
| alcubierre | 416 | 2.05% | 98.08% | 0.00% | 0.00% |  |
| rodal | 5208 | 0.00% | 0.00% | 28.53% | 62.67% |  |
| vdb | 416 | 1.56% | 84.62% | 0.31% | 33.33% |  |
| natario | 416 | 2.46% | 90.38% | 0.00% | 0.00% |  |
| lentz | 16 | 0.07% | 100.00% | 0.00% | 0.00% | unresolved_lower_bound |
| warpshell | 168 | 0.07% | 12.50% | 0.01% | 0.00% |  |
