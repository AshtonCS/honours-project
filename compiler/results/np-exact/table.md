# np-exact: accuracy against exact unitaries

Validation independent of simcount: compiled circuits of all eleven formulas against exactly computed unitaries. Part A: commuting ZZ + Z chain (Jx = Jy = 0, XX/YY kept at coefficient 0), closed-form diagonal reference, every formula must be exact. Part B: full Heisenberg chain (J = 1, periodic) against exact diagonalisation: gate-level step = product of exact exponentials, and error ~ r^-p. NumPy fields seed 1, t = n. Criteria fixed before the run (2026-11-11).

Instance: NumPy fields, seed 1, t = n. Criteria (fixed before the run): tol = 1e-15 per exponential; slope window [1e-10, 0.01], |slope - p| <= 0.3, >= 3 points.

## Part A: commuting ZZ + Z, error vs closed form, r = 1

| formula | exps/step (n=8) | error n=4 | error n=6 | error n=8 | error/exp (n=8) | pass |
|---|---:|---:|---:|---:|---:|---|
| Lie-Trotter S1 | 32 | 3.7e-15 | 8.6e-15 | 1.7e-14 | 5.2e-16 | PASS |
| Suzuki S2 | 56 | 5.8e-15 | 1.1e-14 | 1.8e-14 | 3.3e-16 | PASS |
| Suzuki S4 | 280 | 3.0e-14 | 4.5e-14 | 6.2e-14 | 2.2e-16 | PASS |
| Suzuki S6 | 1,400 | 1.5e-13 | 2.1e-13 | 2.9e-13 | 2.1e-16 | PASS |
| Suzuki S8 | 7,000 | 7.1e-13 | 1.1e-12 | 1.4e-12 | 2.0e-16 | PASS |
| M&O q=6 (rec., Tab. 2) | 384 | 3.5e-14 | 5.4e-14 | 6.9e-14 | 1.8e-16 | PASS |
| M&O q=14 (rec., Tab. 3) | 896 | 8.0e-14 | 1.2e-13 | 1.6e-13 | 1.8e-16 | PASS |
| Blanes & Moan q=10 | 640 | 5.5e-14 | 8.6e-14 | 1.2e-13 | 1.9e-16 | PASS |
| M&O q=21 order 8 (exploratory) | 1,344 | 1.2e-13 | 1.8e-13 | 2.4e-13 | 1.8e-16 | PASS |
| Morales Y8m10 (21 steps) | 1,344 | 1.2e-13 | 1.8e-13 | 2.4e-13 | 1.8e-16 | PASS |
| Morales m=8 (17 steps) | 1,088 | 9.6e-14 | 1.5e-13 | 2.0e-13 | 1.8e-16 | PASS |

## Part A: commuting ZZ + Z, error vs closed form, r = 3

| formula | exps/step (n=8) | error n=4 | error n=6 | error n=8 | error/exp (n=8) | pass |
|---|---:|---:|---:|---:|---:|---|
| Lie-Trotter S1 | 32 | 8.8e-15 | 1.4e-14 | 2.4e-14 | 2.5e-16 | PASS |
| Suzuki S2 | 56 | 1.7e-14 | 2.5e-14 | 4.0e-14 | 2.4e-16 | PASS |
| Suzuki S4 | 280 | 8.7e-14 | 1.3e-13 | 1.7e-13 | 2.1e-16 | PASS |
| Suzuki S6 | 1,400 | 4.3e-13 | 6.3e-13 | 8.4e-13 | 2.0e-16 | PASS |
| Suzuki S8 | 7,000 | 2.2e-12 | 3.3e-12 | 4.5e-12 | 2.1e-16 | PASS |
| M&O q=6 (rec., Tab. 2) | 384 | 1.0e-13 | 1.6e-13 | 2.1e-13 | 1.8e-16 | PASS |
| M&O q=14 (rec., Tab. 3) | 896 | 2.3e-13 | 3.6e-13 | 4.8e-13 | 1.8e-16 | PASS |
| Blanes & Moan q=10 | 640 | 1.7e-13 | 2.6e-13 | 3.5e-13 | 1.8e-16 | PASS |
| M&O q=21 order 8 (exploratory) | 1,344 | 3.6e-13 | 5.5e-13 | 7.3e-13 | 1.8e-16 | PASS |
| Morales Y8m10 (21 steps) | 1,344 | 3.7e-13 | 5.5e-13 | 7.2e-13 | 1.8e-16 | PASS |
| Morales m=8 (17 steps) | 1,088 | 3.0e-13 | 4.4e-13 | 6.0e-13 | 1.8e-16 | PASS |

## Part B: full Heisenberg chain vs exact diagonalisation

| formula | p | n | max ‖G−F‖/step | compile | r in window | fitted order | order |
|---|---:|---:|---:|---|---|---:|---|
| Lie-Trotter S1 | 1 | 4 | 3.1e-15 | PASS | 2000, 4000, 8000, 16000, 32000, 64000 | 1.00 | PASS |
| Lie-Trotter S1 | 1 | 6 | 4.6e-15 | PASS | 8000, 16000, 32000, 64000 | 1.00 | PASS |
| Suzuki S2 | 2 | 4 | 6.1e-15 | PASS | 250, 500, 1000, 2000, 4000, 8000 | 2.00 | PASS |
| Suzuki S2 | 2 | 6 | 9.3e-15 | PASS | 500, 1000, 2000, 4000, 8000 | 2.00 | PASS |
| Suzuki S4 | 4 | 4 | 3.0e-14 | PASS | 24, 32, 48, 64, 96, 128, 192 | 3.90 | PASS |
| Suzuki S4 | 4 | 6 | 4.6e-14 | PASS | 48, 64, 96, 128, 192 | 3.93 | PASS |
| Suzuki S6 | 6 | 4 | 1.5e-13 | PASS | 12, 16, 24, 32, 48, 64, 96, 128 | 6.12 | PASS |
| Suzuki S6 | 6 | 6 | 2.2e-13 | PASS | 24, 32, 48, 64, 96, 128, 192 | 6.20 | PASS |
| Suzuki S8 | 8 | 4 | 7.2e-13 | PASS | 6, 8, 12, 16, 24, 32, 192 | 4.56 | FAIL: |slope - p| = 3.44 > 0.3 |
| Suzuki S8 | 8 | 6 | 1.1e-12 | PASS | 6, 8, 12, 16, 24, 32, 48, 96, 128, 192 | 5.91 | FAIL: |slope - p| = 2.09 > 0.3 |
| M&O q=6 (rec., Tab. 2) | 4 | 4 | 3.6e-14 | PASS | 16, 24, 32, 48, 64, 96, 128, 192 | 4.00 | PASS |
| M&O q=6 (rec., Tab. 2) | 4 | 6 | 5.4e-14 | PASS | 24, 32, 48, 64, 96, 128, 192 | 3.99 | PASS |
| M&O q=14 (rec., Tab. 3) | 6 | 4 | 8.3e-14 | PASS | 6, 8, 12, 16, 24, 32, 48, 64, 96 | 6.13 | PASS |
| M&O q=14 (rec., Tab. 3) | 6 | 6 | 1.2e-13 | PASS | 12, 16, 24, 32, 48, 64, 96, 128 | 6.39 | FAIL: |slope - p| = 0.39 > 0.3 |
| Blanes & Moan q=10 | 6 | 4 | 6.0e-14 | PASS | 12, 16, 24, 32, 48, 64, 96, 128, 192 | 5.96 | PASS |
| Blanes & Moan q=10 | 6 | 6 | 8.9e-14 | PASS | 24, 32, 48, 64, 96, 128, 192 | 5.97 | PASS |
| M&O q=21 order 8 (exploratory) | 8 | 4 | 1.2e-13 | PASS | 8, 12, 16, 24, 32, 48, 64 | 7.95 | PASS |
| M&O q=21 order 8 (exploratory) | 8 | 6 | 1.9e-13 | PASS | 12, 16, 24, 32, 48, 64, 96 | 8.17 | PASS |
| Morales Y8m10 (21 steps) | 8 | 4 | 1.2e-13 | PASS | 8, 12, 16, 24, 32, 48, 64 | 7.91 | PASS |
| Morales Y8m10 (21 steps) | 8 | 6 | 1.9e-13 | PASS | 12, 16, 24, 32, 48, 64, 96 | 8.02 | PASS |
| Morales m=8 (17 steps) | 8 | 4 | 1.0e-13 | PASS | 12, 16, 24, 32, 48, 64 | 8.16 | PASS |
| Morales m=8 (17 steps) | 8 | 6 | 1.5e-13 | PASS | 16, 24, 32, 48, 64, 96, 128 | 8.25 | PASS |

## Negative controls (must FAIL)

| control | measured | criterion | result |
|---|---|---|---|
| A1: S2, times x (1 + 0.001), n=4, r=1 | error 2.2e-02 | <= 2.8e-14 | FAIL (expected) |
| B1: Lie-Trotter claimed order 2, n=4 | fitted order 1.00 | |slope − 2| <= 0.3 | FAIL (expected) |

Wall clock: 690 s.
