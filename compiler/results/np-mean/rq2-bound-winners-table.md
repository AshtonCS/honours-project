# Cheapest Suzuki formula under each source of r (RQ2)

Merged CNOT total (cz, eps = 1e-3, t = n), seeds [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]. Each cell: the formula cheapest on every seed, or each winner with its number of seeds. Candidates: the Suzuki orders with a value under that source (tte and com: orders 1, 2, 4 only; measured, emp, min, ana: 1-8). Made at git_commit=e5d9ee696f892e28aced765585653760f8a427c4 git_dirty=False.

| r from | n = 4 | n = 5 | n = 6 | n = 7 | n = 8 | n = 9 | n = 10 | n = 11 | candidates |
|---|---|---|---|---|---|---|---|---|---|
| measured | S4 | S4 | S4 | S4 | S4 | S4 | S4 | S4 | S1, S2, S4, S6, S8 |
| tte | S4 | S4 | S4 | S4 | S4 | S4 | S4 | S4 | S1, S2, S4 |
| emp | S4 | S4 | S4 | S4 | S4 | S4 | S4 | S4 | S1, S2, S4, S6, S8 |
| com | S2 | S4 | S4 | S4 | S4 | S4 | S4 | S4 | S1, S2, S4 |
| min | S2 | S2 | S2 | S2 | S2 | S2 | S2 | S2 | S1, S2, S4, S6, S8 |
| ana | S2 | S2 | S2 | S2 | S2 | S2 | S2 | S2 | S1, S2, S4, S6, S8 |
