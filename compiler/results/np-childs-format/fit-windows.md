# Extrapolated ranking vs fit window

Written by `python -m pysimcount.pffitwindow`. Record `main-cnot-all-formulas.json` (its figures fit n = [5, 6, 7, 8, 10]); seeds [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; mean measured r over seeds, power law fitted per window, count = exact per-step count x ceil(a n^b) (pfchilds's rule; the record's own window reproduces its values exactly). Rows ordered by the fit-window column `5-11`; bold = predicted cheapest.

## n = 100: count / cheapest count under the same fit window

| formula | fit n = 5-10 (no 9) | fit n = 5-11 | fit n = 4-11 | fit n = 6-11 | fit n = 7-11 |
|---|---:|---:|---:|---:|---:|
| M&O q=14 | 1.14 | **1.00** | 1.07 | **1.00** | **1.00** |
| Blanes & Moan q=10 | **1.00** | 1.02 | **1.00** | 1.65 | 1.68 |
| Morales m=8 | 1.07 | 1.11 | 1.58 | 1.63 | 1.51 |
| M&O q=6 | 1.41 | 1.45 | 1.39 | 2.57 | 2.50 |
| Morales Y8m10 | 1.51 | 1.52 | 1.42 | 2.11 | 1.90 |
| M&O q=21, order 8 | 1.85 | 1.60 | 1.45 | 1.62 | 1.58 |
| Suzuki S6 | 3.24 | 2.79 | 2.75 | 3.02 | 2.29 |
| Suzuki S4 | 4.24 | 4.03 | 3.05 | 5.90 | 5.11 |

## n = 500: count / cheapest count under the same fit window

| formula | fit n = 5-10 (no 9) | fit n = 5-11 | fit n = 4-11 | fit n = 6-11 | fit n = 7-11 |
|---|---:|---:|---:|---:|---:|
| Blanes & Moan q=10 | 1.00 | **1.00** | **1.00** | 2.01 | 2.08 |
| Morales m=8 | **1.00** | 1.02 | 1.89 | 1.76 | 1.54 |
| M&O q=14 | 1.40 | 1.10 | 1.27 | **1.00** | **1.00** |
| Morales Y8m10 | 1.64 | 1.60 | 1.48 | 2.50 | 2.10 |
| M&O q=6 | 1.68 | 1.71 | 1.64 | 4.00 | 3.83 |
| M&O q=21, order 8 | 2.39 | 1.84 | 1.60 | 1.69 | 1.62 |
| Suzuki S6 | 4.52 | 3.44 | 3.47 | 3.55 | 2.23 |
| Suzuki S4 | 8.29 | 7.40 | 4.80 | 12.61 | 9.91 |

## Fitted exponent b of r = a n^b

| formula | fit n = 5-10 (no 9) | fit n = 5-11 | fit n = 4-11 | fit n = 6-11 | fit n = 7-11 |
|---|---:|---:|---:|---:|---:|
| Suzuki S4 | 1.654 | 1.599 | 1.411 | 1.535 | 1.556 |
| Suzuki S6 | 1.445 | 1.353 | 1.273 | 1.164 | 1.128 |
| M&O q=6 | 1.349 | 1.323 | 1.231 | 1.339 | 1.409 |
| M&O q=14 | 1.369 | 1.286 | 1.239 | 1.064 | 1.146 |
| Blanes & Moan q=10 | 1.241 | 1.211 | 1.129 | 1.188 | 1.278 |
| M&O q=21, order 8 | 1.397 | 1.308 | 1.192 | 1.090 | 1.160 |
| Morales Y8m10 | 1.289 | 1.254 | 1.154 | 1.169 | 1.207 |
| Morales m=8 | 1.196 | 1.171 | 1.242 | 1.111 | 1.159 |
