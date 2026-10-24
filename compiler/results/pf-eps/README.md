# pf-eps

Measured Trotter number r for every pf-thesis formula at three error targets,
eps_trotter ∈ {1e-2, 1e-3, 1e-4}. Settings: cz gate set (no synthesis),
simcount seed 1, n = 4–10, t = n. Everything else is as in `../pf-thesis`.

## Reproduce

From `compiler/`, `pennylane` environment, no Docker:

```
python -m plsimcount.pfcompare results/pf-eps/config.json | tee results/pf-eps/run.log   # ~36 min (shared machine)
python -m plsimcount.pfeps     results/pf-eps/results.json --render
```

The sweep is the config key `eps_trotter` (a list), added to `pfcompare`
for this run. It is accepted only for gate sets without synthesis (cz,
cz2ct), because for `ct` the Trotter/synthesis split is part of the gate
set's definition. It cannot be combined with `paper_instance`,
`extrapolate` or `bound_rows`. Configs without the key behave as before:
a test reruns part of pf-thesis's config and compares every field. As a
further check, **the eps = 1e-3 rows of this run equal pf-thesis's 132
measured cz rows field for field** (only wall-clock time excluded).

Outputs: `results.json`/`.csv` (rows carry `eps_trotter`), `slopes.md` /
`slopes.json`, `figures/cnot-order<p>-vs-eps.tex` + `all.pdf`,
`environment.txt`.

## What is tested

For an order-p formula, the Trotter error after r steps of size t/r behaves
asymptotically as C t^(p+1) / r^p. The smallest r with error ≤ eps
therefore scales as r ∝ eps^(−1/p): on a log-log plot, the slope of r
against eps tends to −1/p. `pfeps` fits that slope per (formula, n): least
squares over the three eps values, plus the two pairwise slopes. The
pairwise slopes show whether the slope approaches −1/p as eps falls.
Points with r ≤ 3 would be flagged as dominated by discreteness. **None
occurs** (smallest r is 5, S8 at n = 4).

## Results (`slopes.md` has every row)

fit slope / (−1/p), range over n = 4–10:

| formula | p | ratio | comment |
|---|---|---|---|
| Lie-Trotter | 1 | 1.00 (all n) | r scales exactly 10× per decade |
| Suzuki S2 | 2 | 1.00 (all n) | |
| Suzuki S4 | 4 | 1.02–1.06 | |
| M&O q=6 | 4 | 0.97–1.00 | |
| Blanes–Moan q=10 | 6 | 0.96–1.05 | |
| M&O q=14 | 6 | 0.82–1.01 | |
| **Suzuki S6** | 6 | **0.55–0.83** | shallower than −1/6 at every n |
| M&O q=21 (order 8) | 8 | 0.95–1.09 | |
| Morales Y8m10 | 8 | 0.97–1.10 | |
| Morales m=8 | 8 | 0.73–0.95 | |
| **Suzuki S8** | 8 | **1.02–1.70** | steeper than −1/8 at most n |

Reading:

- **Orders 1, 2 and 4 follow r ∝ eps^(−1/p) closely** (within 6%) at
  every n. This is a consistency check of the brute-force r: the measured r
  has the expected eps dependence.
- **At orders 6 and 8 the agreement is looser and partly
  discretisation-limited.** r there is 5–50, so one step is 2–20% of r, and
  across a decade of eps r changes by only a factor 10^(1/6) = 1.47 (order
  6) or 1.33 (order 8). Example: S8 at n = 4 has r = 9, 5, 5 at eps = 1e-4,
  1e-3, 1e-2, a pairwise slope of 0.000 then −0.255. With ±1 step on r ≈ 10,
  each end of the 2-decade fit moves the slope by about
  ±0.02 (ln(11/10) / ln 100), i.e. ±15% of −1/8 per endpoint.
- **Suzuki S6 is the exception that discreteness does not obviously
  explain.** Its slope is shallower than −1/6 at all six sizes (ratio
  0.55–0.83), systematically, while Blanes–Moan q=10 at the same order and
  similar r sits at ≈ 1. A shallow slope means r grows more slowly than the
  asymptotic law as eps falls. A possible reading is that S6 at these step
  sizes (d = t/r ≈ 0.2–0.45) is not yet in its asymptotic regime, i.e.
  higher-order error terms still matter. That is a hypothesis, not checked
  here. One way to check it: extend the sweep to eps = 1e-5, 1e-6 for S6
  and see whether the pairwise slope moves toward −1/6.
- Per-order CNOT figures (n = 8 dashed, n = 10 solid, with a dotted −1/p
  reference line): within each order, the ranking of formulas by CNOT count
  is the same at all three eps and all n for orders 4 (M&O q=6 < S4) and 6
  (M&O q=14 < Blanes–Moan q=10 < S6). At order 8, S8 is costliest at every
  eps and n, and the three non-Suzuki schemes are within a factor 1.36 of
  each other and swap places as eps changes (e.g. n = 10: M&O q=21 is
  cheapest at 1e-2, Morales m=8 at 1e-3 and 1e-4).

- **Non-monotone error (regenerated 2026-10-22 with the "keeps working"
  definition of r, `fastempirical.WINDOW` = 10).** Three rows had a dip that
  fails again just above it: M&O q=14 n = 6 at 1e-3 (14 → 16), and Suzuki
  S8 at 1e-4 for n = 6 (13 → 16) and n = 8 (19 → 21). Each row's
  `search_dips` lists them. The S8 changes are why its ratio range widened
  from 1.02–1.50 to 1.02–1.70.

## Assumptions

- `# ASSUMPTION` (in the config/README, unconfirmed): eps ∈ {1e-2, 1e-3,
  1e-4}, seed 1, n = 4–10, t = n, cz only, as specified for this run.
- The reference r ∝ eps^(−1/p) is the standard leading-order scaling. I
  have not checked a specific equation number in Childs et al., PRX 11,
  011020 (2021) for it.
