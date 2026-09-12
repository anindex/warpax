# Rodal matched-parameter resolution

Date: 2026-09-12T09:35:27Z. Source: `rodal_matched_resolution.json`.

v_s=0.5, R=1.0, sigma=8.0, domain=[-3,3]^3. Miss percentages count grid points where the Eulerian test passes and the observer search finds a violation, divided by all grid points.

| N | Grid points | NEC miss % | WEC miss % | SEC miss % | DEC miss % | Type I % | Time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 27000 | 0.77 | 13.93 | 28.06 | 26.84 | 100.0 | 7.6 |
| 50 | 125000 | 0.88 | 14.41 | 27.80 | 27.32 | 100.0 | 7.1 |
| 70 | 343000 | 0.92 | 14.59 | 27.84 | 27.53 | 100.0 | 9.0 |

All conditions pass the numerical stability test: maximum deviation from the three-grid mean <= 0.5 percentage points (pp) **or** relative deviation <= 5%. Relative deviation is the absolute deviation divided by the mean. This is a resolution check, not a continuum error bound.

| Condition | Mean miss % | Max deviation (pp) | Relative deviation % | Stable |
|---|---:|---:|---:|---|
| NEC | 0.8554 | 0.0850 | 9.9376 | yes |
| WEC | 14.3100 | 0.3841 | 2.6839 | yes |
| SEC | 27.8992 | 0.1601 | 0.5737 | yes |
| DEC | 27.2309 | 0.3864 | 1.4190 | yes |
