# Fitted exponent b of r = a n^b (RQ3)

From `main-cnot-all-formulas.json`: mean measured r over seeds [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] fitted on n = 5-11 (least squares in log-log); per-seed range = the same fit on each seed alone. Theory 1 + 2/p: Childs et al. 2021 Sec. 4.1 with t = n (our substitution). Eq. 147: Childs et al. 2018 (simcount's constants), Suzuki only. Made at git_commit=e5d9ee696f892e28aced765585653760f8a427c4 git_dirty=False.

| formula | p | b (mean r) | per-seed b (min–max) | 1 + 2/p | Eq. 147 |
|---|---:|---:|---:|---:|---:|
| Lie-Trotter S1 | 1 | 2.182 | 1.636–3.322 | 3.000 | 1.964 |
| S2 | 2 | 1.899 | 1.747–2.006 | 2.000 | 1.883 |
| S4 | 4 | 1.599 | 1.545–1.669 | 1.500 | 1.555 |
| M&O q=6 | 4 | 1.323 | 1.255–1.402 | 1.500 | — |
| S6 | 6 | 1.353 | 1.299–1.429 | 1.333 | 1.311 |
| Blanes & Moan q=10 | 6 | 1.211 | 1.149–1.264 | 1.333 | — |
| M&O q=14 | 6 | 1.286 | 1.266–1.329 | 1.333 | — |
| S8 | 8 | 1.249 | 1.153–1.294 | 1.250 | 1.141 |
| M&O q=21 (order 8) | 8 | 1.308 | 1.247–1.379 | 1.250 | — |
| Morales m=8 | 8 | 1.171 | 1.137–1.219 | 1.250 | — |
| Morales Y8m10 | 8 | 1.254 | 1.186–1.316 | 1.250 | — |
