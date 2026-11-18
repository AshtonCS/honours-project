# Extrapolated ranking vs fit window

Written by `python -m pysimcount.pffitwindow`. Record `main-cnot-all-formulas.json` (its figures fit n = [5, 6, 7, 8, 9, 10, 11]); seeds [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; mean measured r over seeds, power law fitted per window, count = exact per-step count x ceil(a n^b) averaged over the per-seed fits (pfchilds's rule, point=seed-mean; the record's own window reproduces its values exactly). Rows ordered by the fit-window column `5-11`; bold = predicted cheapest.

## n = 100: count / cheapest count under the same fit window

| formula | fit n = 4-11 | fit n = 5-11 | fit n = 6-11 | fit n = 7-11 |
|---|---:|---:|---:|---:|
| M&O q=14 | 1.07 | **1.00** | **1.00** | **1.00** |
| Blanes & Moan q=10 | **1.00** | 1.02 | 1.65 | 1.68 |
| Morales m=8 | 1.58 | 1.11 | 1.64 | 1.50 |
| M&O q=6 | 1.40 | 1.46 | 2.59 | 2.63 |
| Morales Y8m10 | 1.42 | 1.53 | 2.13 | 1.91 |
| M&O q=21, order 8 | 1.45 | 1.61 | 1.63 | 1.59 |
| Suzuki S6 | 2.75 | 2.81 | 3.05 | 2.30 |
| Suzuki S4 | 3.05 | 4.05 | 5.97 | 5.22 |

## n = 500: count / cheapest count under the same fit window

| formula | fit n = 4-11 | fit n = 5-11 | fit n = 6-11 | fit n = 7-11 |
|---|---:|---:|---:|---:|
| Blanes & Moan q=10 | **1.00** | **1.00** | 2.02 | 2.10 |
| Morales m=8 | 1.89 | 1.01 | 1.77 | 1.53 |
| M&O q=14 | 1.27 | 1.10 | **1.00** | **1.00** |
| Morales Y8m10 | 1.48 | 1.60 | 2.56 | 2.13 |
| M&O q=6 | 1.67 | 1.71 | 4.13 | 4.41 |
| M&O q=21, order 8 | 1.60 | 1.84 | 1.72 | 1.64 |
| Suzuki S6 | 3.46 | 3.45 | 3.64 | 2.27 |
| Suzuki S4 | 4.79 | 7.42 | 13.07 | 10.46 |

## Fitted exponent b of r = a n^b

| formula | fit n = 4-11 | fit n = 5-11 | fit n = 6-11 | fit n = 7-11 |
|---|---:|---:|---:|---:|
| Suzuki S4 | 1.411 | 1.599 | 1.535 | 1.556 |
| Suzuki S6 | 1.273 | 1.353 | 1.164 | 1.128 |
| M&O q=6 | 1.231 | 1.323 | 1.339 | 1.409 |
| M&O q=14 | 1.239 | 1.286 | 1.064 | 1.146 |
| Blanes & Moan q=10 | 1.129 | 1.211 | 1.188 | 1.278 |
| M&O q=21, order 8 | 1.192 | 1.308 | 1.090 | 1.160 |
| Morales Y8m10 | 1.154 | 1.254 | 1.169 | 1.207 |
| Morales m=8 | 1.242 | 1.171 | 1.111 | 1.159 |
