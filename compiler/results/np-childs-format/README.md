# np-childs-format

Gate count against system size n for n = 4-500, in the format of Childs et al.
(2018), Figs. 3 and 10 (arXiv:1711.10980; method in SI F.3, p. 38).

## Reproduce

```
cd compiler
python -m pysimcount.pfchilds steps   results/np-childs-format/config.json   # per-step counts (~26 min, 24 of them optimised; resumable)
python -m pysimcount.pfchilds figures results/np-childs-format/config.json --render
python -m pysimcount.figgallery results --render                            # searchable index
```

Inputs: the committed runs `np-main/seed*`, `np-grouping/seed*`,
`np-optimised-seeds/seed*` (NumPy seeds 1-10), read through their
`aggregate.json` run lists, and this directory's `config.json`.

## What each number is

| Plotted point | How it is obtained | Checked |
|---|---|---|
| measured n (4-10) | mean over seeds of the run's own total | per-step count identical on every seed |
| extrapolated r | ceil(a n^b), (a, b) fitted to the **mean** measured r over seeds at `fit_sizes` | -- (extrapolation) |
| error bars, extrapolated | min..max over seeds of each seed's own power-law fit | -- |
| per-step count, unoptimised and two-group | c(n) = alpha n + beta, exactly affine on n = 10, 20, 40 | equals the run rows at n = 10 and a direct build at n = 100 |
| per-step count, optimised | affine on n = 10, 13, 16 (statevector-verified) | **not** built or verified beyond n = 16 |

Wording used in figure headers:

- r: "r from power-law fit to mean measured r (n = 5--10)" (two-group: n = 6--10, even n only)
- optimised: "n > 16: optimised per-step count from linear fit (n = 10--16); circuit not built or verified"
  (legend: "Optimised count linear in n, fitted on n = 10--16")

## Deviations from Childs et al. (2018)

- 10 random instances (paper: 5); fields U[-1, 1] from NumPy `default_rng([seed, n])`.
- Fit on n = 5, 6, 7, 8, 10 (paper: 5-12). Two-group fits on n = 6, 8, 10 and plots
  even n only (14 in place of 13).
- r is the smallest r after which every r up to r + 10 passes (`fastempirical`);
  the paper used plain bisection.
- Counts: simcount template, merged steps, Clifford+Rz total excluding global-phase
  markers. Absolute totals are not comparable with the paper's figures without
  checking what each counts; compare shapes and orderings.

## Pruning

Each figure drops up to 3 formulas, worst first by the geometric mean of the
plotted values over every plotted n, but never one of the 2 cheapest at any
plotted n. `<id>-ranking.md` gives the ranking and the reason for every
formula; `<id>.json` holds every plotted value and both fits.

## Disagreement with Eq. 147 (reported, not adjusted)

Fitted exponents b of the mean measured r (n = 5-10), against Childs et al. (2018) Eq. 147 (n = 5-12):

| formula | b here | b, Eq. 147 |
|---|---:|---:|
| Lie-Trotter S1 | 2.285 | 1.964 |
| Suzuki S2 | 1.957 | 1.883 |
| Suzuki S4 | 1.654 | 1.555 |
| Suzuki S6 | 1.445 | 1.311 |
| Suzuki S8 | 1.325 | 1.141 |

At n = 6-10 the two agree on r itself to within ~5 % (np-main/aggregate/bounds.md, emp rows), so the
difference is in the slope over a short window. Not investigated: fit window (5-10 vs 5-12), mean of 10
vs 5 instances, window vs bisection definition of r.
