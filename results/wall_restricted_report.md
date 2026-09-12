# Wall-restricted classification and missed violations

Date: 2026-09-12T09:36:26Z. Source: `wall_restricted_analysis.json`.

v_s=0.5; 50^3 grid points per metric, with bounds from `run_analysis.py`. The wall mask is 0.1 <= f <= 0.9. Type fractions use point counts within the stated region.

| Metric | Grid points | Wall points | Full Type IV % | Wall I / II / III / IV % | Time (s) |
|---|---:|---:|---:|---|---:|
| alcubierre | 125000 | 416 | 15.49 | 0.00 / 0.00 / 0.00 / 100.00 | 106.2 |
| rodal | 125000 | 5208 | 0.00 | 100.00 / 0.00 / 0.00 / 0.00 | 2.6 |
| vdb | 125000 | 416 | 5.67 | 15.38 / 0.00 / 0.00 / 84.62 | 39.7 |
| natario | 125000 | 416 | 99.49 | 17.31 / 0.00 / 0.00 / 82.69 | 522.4 |

A missed violation passes the Eulerian test but violates the condition for a searched observer. Full-grid percentages divide by all grid points; wall percentages divide by wall points with a detected violation of that condition. `N/A` means that denominator is zero. These denominators differ.

| Metric | Region | NEC miss % | WEC miss % | SEC miss % | DEC miss % |
|---|---|---:|---:|---:|---:|
| alcubierre | full grid | 0.00 | 0.00 | 7.19 | 0.00 |
| alcubierre | wall | 0.00 | 0.00 | 15.38 | 0.00 |
| rodal | full grid | 1.58 | 15.60 | 28.01 | 28.53 |
| rodal | wall | 10.13 | 60.78 | 11.60 | 62.67 |
| vdb | full grid | 0.10 | 0.36 | 1.78 | 0.31 |
| vdb | wall | 13.33 | 53.33 | 13.46 | 33.33 |
| natario | full grid | 0.00 | 0.00 | 3.21 | 0.00 |
| natario | wall | 0.00 | 0.00 | 0.00 | 0.00 |
