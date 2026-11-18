# Optimiser saving split into merging and optimiser-only, n = 13

Written by `python -m pysimcount.pfmergesplit results/childs-fig13-split/config.json`. Per step: U unmerged (simcount template), M after `weighted.merge`, O Light reconstruction of U (marginal). Percentages of U. O_M (optimised merged step) equals O for every row (checked). 'total at r' includes the edge term.

## CNOT

| formula | n | r | U/n | M/n | O/n | merging | optimiser only | total (marginal) | ours, total at r | published, total at r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| First order | 13 | 372,491 | 6 | 6 | 6 | 0.0% | 0.0% | 0.0% | -- | -- |
| Second order | 13 | 4,937 | 12 | 10 | 8 | 16.7% | 16.7% | 33.3% | -- | -- |
| Fourth order | 13 | 218 | 60 | 42 | 40 | 30.0% | 3.3% | 33.3% | 33.32% | 33.32% |
| Sixth order | 13 | 52 | 300 | 202 | 200 | 32.7% | 0.7% | 33.3% | 33.32% | 33.32% |
| Eighth order | 13 | 22 | 1500 | 1002 | 1000 | 33.2% | 0.1% | 33.3% | -- | -- |

## Rz

| formula | n | r | U/n | M/n | O/n | merging | optimiser only | total (marginal) | ours, total at r | published, total at r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| First order | 13 | 372,491 | 4 | 4 | 4 | 0.0% | 0.0% | 0.0% | -- | -- |
| Second order | 13 | 4,937 | 7 | 6 | 5 | 14.3% | 14.3% | 28.6% | -- | -- |
| Fourth order | 13 | 218 | 35 | 26 | 25 | 25.7% | 2.9% | 28.6% | 28.56% | 28.56% |
| Sixth order | 13 | 52 | 175 | 126 | 125 | 28.0% | 0.6% | 28.6% | 28.56% | 28.56% |
| Eighth order | 13 | 22 | 875 | 626 | 625 | 28.5% | 0.1% | 28.6% | -- | -- |
