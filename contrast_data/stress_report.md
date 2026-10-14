# Stress test report

63 files · recall **81%** (severity 1: 71%, 2: 89%, 3: 83%) · precision **66%** · false alarms on clean baselines: **0**

| Flaw | sev 1 | sev 2 | sev 3 | start err (median) | end err (median) | score s1 → s2 → s3 |
|---|---|---|---|---|---|---|
| dragging | 3/3 | 3/3 | 2/3 | 0.078 s | 0.273 s | 9.47 → 9.03 → 8.8 |
| long_pause | 3/3 | 3/3 | 5/6 | 0.032 s | 0.027 s | 9.37 → 9.13 → 8.37 |
| monotone | 5/6 | 3/3 | 6/6 | 0.335 s | 0.383 s | 9.5 → 8.97 → 8.73 |
| mumble | 3/3 | 3/3 | 3/3 | 0.119 s | 0.021 s | 8.93 → 8.27 → 7.17 |
| rushed | 3/6 | 3/3 | 6/6 | 0.051 s | 0.02 s | 9.77 → 9.33 → 9.1 |
| stutter | 0/3 | 1/3 | 3/6 | 0.045 s | 0.226 s | 9.87 → 9.9 → 9.6 |

Rubric falls monotonically with severity for 5/6 flaw types.
Composite files: {'baseline': 10.0, 'near_perfect': 9.0, 'botched': 5.33}
