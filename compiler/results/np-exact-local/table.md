# np-exact-local: local (single-step) order against exact diagonalisation

Local (single-step) order check, Part C of pfexact: ||G(d) - exp(-idH)|| ~ d^(p+1) for all eleven formulas on the full Heisenberg chain (J = 1, periodic), NumPy fields seed 1, n = 4 and 6, against exact diagonalisation. Added after np-exact's Part B (ii) failed for S8 (n = 4, 6) and M&O q=14 (n = 6), diagnosed as r-fold rounding accumulation and pre-asymptotic points in a fixed error window. np-exact is kept unchanged as the original result. Criteria fixed 2026-11-11 before this run.

Criteria (fixed before the run): usable if error >= 100 x 1e-15 x exponentials/step; fit the 4 smallest usable d; pass if |slope - (p+1)| <= 0.3.

| formula | p | n | exps/step | d used | error range | slope (expect p+1) | fitted order | result |
|---|---:|---:|---:|---|---|---:|---:|---|
| Lie-Trotter S1 | 1 | 4 | 16 | 0.000977, 0.00138, 0.00195, 0.00276 | 1.4e-05–1.1e-04 | 2.00 | 1.00 | PASS |
| Lie-Trotter S1 | 1 | 6 | 24 | 0.000977, 0.00138, 0.00195, 0.00276 | 1.5e-05–1.2e-04 | 2.00 | 1.00 | PASS |
| Suzuki S2 | 2 | 4 | 28 | 0.000977, 0.00138, 0.00195, 0.00276 | 2.2e-08–5.0e-07 | 3.00 | 2.00 | PASS |
| Suzuki S2 | 2 | 6 | 42 | 0.000977, 0.00138, 0.00195, 0.00276 | 2.2e-08–4.9e-07 | 3.00 | 2.00 | PASS |
| Suzuki S4 | 4 | 4 | 140 | 0.00391, 0.00552, 0.00781, 0.011 | 1.5e-11–2.8e-09 | 5.00 | 4.00 | PASS |
| Suzuki S4 | 4 | 6 | 210 | 0.00552, 0.00781, 0.011, 0.0156 | 1.0e-10–1.8e-08 | 5.00 | 4.00 | PASS |
| Suzuki S6 | 6 | 4 | 700 | 0.0442, 0.0625, 0.0884, 0.125 | 6.1e-10–8.6e-07 | 6.98 | 5.98 | PASS |
| Suzuki S6 | 6 | 6 | 1,050 | 0.0442, 0.0625, 0.0884, 0.125 | 1.1e-09–1.6e-06 | 6.97 | 5.97 | PASS |
| Suzuki S8 | 8 | 4 | 3,500 | 0.177, 0.25, 0.354, 0.5 | 3.9e-09–3.1e-05 | 8.64 | 7.64 | FAIL: |slope - (p+1)| = 0.36 > 0.3 |
| Suzuki S8 | 8 | 6 | 5,250 | 0.177, 0.25, 0.354, 0.5 | 1.1e-08–6.9e-05 | 8.40 | 7.40 | FAIL: |slope - (p+1)| = 0.60 > 0.3 |
| M&O q=6 (rec., Tab. 2) | 4 | 4 | 192 | 0.00781, 0.011, 0.0156, 0.0221 | 6.4e-11–1.2e-08 | 5.00 | 4.00 | PASS |
| M&O q=6 (rec., Tab. 2) | 4 | 6 | 288 | 0.00781, 0.011, 0.0156, 0.0221 | 5.1e-11–9.2e-09 | 5.00 | 4.00 | PASS |
| M&O q=14 (rec., Tab. 3) | 6 | 4 | 448 | 0.0442, 0.0625, 0.0884, 0.125 | 5.8e-11–8.0e-08 | 6.95 | 5.95 | PASS |
| M&O q=14 (rec., Tab. 3) | 6 | 6 | 672 | 0.0625, 0.0884, 0.125, 0.177 | 6.8e-10–8.8e-07 | 6.89 | 5.89 | PASS |
| Blanes & Moan q=10 | 6 | 4 | 320 | 0.0312, 0.0442, 0.0625, 0.0884 | 9.0e-11–1.3e-07 | 6.99 | 5.99 | PASS |
| Blanes & Moan q=10 | 6 | 6 | 480 | 0.0312, 0.0442, 0.0625, 0.0884 | 8.3e-11–1.2e-07 | 6.98 | 5.98 | PASS |
| M&O q=21 order 8 (exploratory) | 8 | 4 | 672 | 0.0884, 0.125, 0.177, 0.25 | 1.9e-10–2.0e-06 | 8.91 | 7.91 | PASS |
| M&O q=21 order 8 (exploratory) | 8 | 6 | 1,008 | 0.0884, 0.125, 0.177, 0.25 | 2.7e-10–2.8e-06 | 8.89 | 7.89 | PASS |
| Morales Y8m10 (21 steps) | 8 | 4 | 672 | 0.0884, 0.125, 0.177, 0.25 | 2.4e-10–2.7e-06 | 8.98 | 7.98 | PASS |
| Morales Y8m10 (21 steps) | 8 | 6 | 1,008 | 0.0884, 0.125, 0.177, 0.25 | 3.7e-10–4.1e-06 | 8.96 | 7.96 | PASS |
| Morales m=8 (17 steps) | 8 | 4 | 544 | 0.0625, 0.0884, 0.125, 0.177 | 6.9e-11–5.9e-07 | 8.71 | 7.71 | PASS |
| Morales m=8 (17 steps) | 8 | 6 | 816 | 0.0884, 0.125, 0.177, 0.25 | 1.7e-09–2.2e-05 | 9.10 | 8.10 | PASS |

## Negative controls (must FAIL)

| control | slope | expected slope | result |
|---|---:|---:|---|
| C1: Lie-Trotter claimed order 2, n=4 | 2.00 | 3 | FAIL (expected) |
| C2: S4, times x (1 + 1e-06), n=4 | 1.00 | 5 | FAIL (expected) |

Wall clock: 339 s.
