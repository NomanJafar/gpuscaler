| policy | GPU-hours | peak replicas | wait p50 | p95 | p99 | max | peak queue | served | unserved |
|---|---|---|---|---|---|---|---|---|---|
| fixed: peak-sized | 3.007 | 3 | 0.5 s | 0.9 s | 1.0 s | 1.0 s | 0 | 11777 | 0 |
| fixed: average-sized | 1.013 | 1 | 111.0 s | 251.0 s | 273.4 s | 283.9 s | 1508 | 11777 | 0 |
| reactive | 2.815 | 5 | 0.8 s | 40.3 s | 43.2 s | 46.0 s | 482 | 11777 | 0 |
| reactive + 1 warm standby | 2.655 | 5 | 0.6 s | 1.9 s | 2.8 s | 3.8 s | 31 | 11777 | 0 |
