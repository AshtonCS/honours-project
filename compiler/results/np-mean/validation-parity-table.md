# Parity with simcount (Validation)

Every cached simcount fixture replayed with the comparison of tests/test_simcount_parity.py: same r, same total per-step gates, every aggregated gate count equal (rotations matched by angle label). Made at git_commit=e5d9ee696f892e28aced765585653760f8a427c4 git_dirty=False.

| formula | r from | gate set | cases | sizes n | simcount seeds | matched |
|---|---|---|---:|---|---|---:|
| Lie-Trotter S1 | ana | cz | 4 | 4–20 | 1 | 4/4 |
| Lie-Trotter S1 | com | cz | 5 | 2–20 | 1 | 5/5 |
| Lie-Trotter S1 | emp | cz | 4 | 4–20 | 1 | 4/4 |
| Lie-Trotter S1 | min | cz | 4 | 4–20 | 1 | 4/4 |
| Suzuki S2 | ana | cz | 4 | 4–20 | 1 | 4/4 |
| Suzuki S2 | com | ct | 2 | 4–6 | 1 | not checked (Clifford+T, needs Quipper) |
| Suzuki S2 | com | cz | 11 | 2–20 | 1–3 | 11/11 |
| Suzuki S2 | com | cz2ct | 2 | 4–6 | 1 | 2/2 |
| Suzuki S2 | emp | cz | 36 | 4–20 | 1–5 | 36/36 |
| Suzuki S2 | min | cz | 4 | 4–20 | 1 | 4/4 |
| Suzuki S4 | ana | cz | 4 | 4–20 | 1 | 4/4 |
| Suzuki S4 | com | cz | 6 | 4–20 | 1 | 6/6 |
| Suzuki S4 | emp | cz | 5 | 4–20 | 1 | 5/5 |
| Suzuki S4 | min | cz | 4 | 4–20 | 1 | 4/4 |
| Suzuki S6 | ana | cz | 4 | 4–20 | 1 | 4/4 |
| Suzuki S6 | emp | cz | 4 | 4–20 | 1 | 4/4 |
| Suzuki S6 | min | cz | 4 | 4–20 | 1 | 4/4 |
| Suzuki S8 | ana | cz | 4 | 4–20 | 1 | 4/4 |
| Suzuki S8 | emp | cz | 4 | 4–20 | 1 | 4/4 |
| Suzuki S8 | min | cz | 4 | 4–20 | 1 | 4/4 |

Total: 119 fixtures; 117 checked here, 117 matched exactly; 2 Clifford+T fixtures need the Quipper backend.
