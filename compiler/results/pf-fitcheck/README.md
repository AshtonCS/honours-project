# pf-fitcheck

How far can the extrapolated Trotter numbers in `../pf-thesis` (n = 13, 20,
open markers, `r_source == "fit"`) be trusted? This directory tests the
extrapolation rule on sizes where r *was* measured. No new simulation: it
reads the measured rows of `../pf-thesis/results.json` (seed 1, t = n,
cz: eps_trotter = 1e-3, ct: 5e-4).

## Reproduce

From `compiler/`, `pennylane` environment, no Docker:

```
python -m plsimcount.pffit results/pf-fitcheck/config.json
```

Writes `results.json`, `results.csv`, `table.md` (all tables) and
`environment.txt`. The sha256 of the source results file is recorded in
`results.json` (`meta.source_results_sha256`).

## Method

The rule under test is `pfcompare`'s, unchanged: r(n) = ceil(a n^b), with
(a, b) from least squares on log r = log a + b log n over the fit sizes.

1. **Hold-out.** Fit on {5,6,7,8} and on {6,7,8}, predict n = 10; fit on
   {4,5,6,7}, predict n = 8. Relative error = (predicted − measured) /
   measured; positive = over-prediction.
2. **Fit-window sensitivity.** Predict n = 13 and 20 from six different
   windows of measured sizes (pf-thesis uses {6,7,8,10}; that column
   reproduces pf-thesis's fit rows exactly, checked). The max/min ratio says
   how much the extrapolated r depends on a choice the data do not fix. It is
   not an error bar.

Gate totals: for cz the per-step counts do not depend on r, so every cz total
inherits r's relative error. For ct the per-step T count also depends (weakly)
on r through eps_rot; not examined here.

## Results (seed 1 only)

Hold-out, all 11 formulas (full table in `table.md`):

| check | gate set | relative error, min to max | worst |
|---|---|---|---|
| {5,6,7,8} → 10 | cz | +0.0% to +18.9% | Suzuki S6 |
| {6,7,8} → 10 | cz | +0.0% to +65.3% (+22.2% without Lie-Trotter) | Lie-Trotter |
| {4,5,6,7} → 8 | cz | −17.9% to +3.7% | Lie-Trotter |
| {5,6,7,8} → 10 | ct | −1.0% to +13.7% | Lie-Trotter |
| {6,7,8} → 10 | ct | +0.0% to +65.3% | Lie-Trotter |
| {4,5,6,7} → 8 | ct | −17.9% to +7.1% | Lie-Trotter |

What the data show:

- **Predicting n = 10 from n ≤ 8 never under-predicts by more than 1%**:
  of 44 cases, 38 over-predict, 5 are exact and one is −1.0%; the
  over-prediction reaches ~19% (fit on {5,6,7,8}) to ~22%
  ({6,7,8}) for orders ≥ 2. **Predicting n = 8 from n ≤ 7 errs both ways**
  (of 22 cases: 11 under, 5 over, 6 exact). So the sign depends on the
  window; it is not a systematic bias that could be corrected.
- **The fitted exponent b is not stable.** For the same formula it moves by
  0.2–0.5 between windows (e.g. Suzuki S8, cz: 1.26 to 1.56; Lie-Trotter:
  1.57 to 3.03). The local slope of r(n) on one instance is noisy because
  the random fields change with n (each size is a different instance, not
  a larger copy of the same one).
- **Lie-Trotter is the outlier**: measured r is not monotone in n
  (r(5) = 84,473 > r(6) = 79,740), and its extrapolations are unreliable by
  factors, not percent.
- **Small r** (orders 6–8, r ≈ 15–40): a change of 1–3 steps is already
  5–20%, so these relative errors are partly ceiling/discreteness.

Fit-window sensitivity at n = 13 and 20 (max/min over six windows, cz):

| | cz, n = 13 | cz, n = 20 | ct, n = 13 | ct, n = 20 |
|---|---|---|---|---|
| orders 2–8 (10 formulas) | 1.08 – 1.36 | 1.14 – 1.78 | 1.07 – 1.20 | 1.14 – 1.38 |
| Lie-Trotter | 2.48 | 5.05 | 2.48 | 5.05 |

## How far to trust the n = 13 / 20 points

Stated only as far as the data support (one seed, one t, one eps):

- **n = 13** is a 1.3× step beyond the largest measured size, about the same
  as the tested 8 → 10 (1.25×). The hold-out errors there were up to ~±20%
  for orders ≥ 2, and the window sensitivity is up to 1.36×. Treat the
  n = 13 r (and every cz gate total) as uncertain at the **~20–35%** level for
  orders ≥ 2.
- **n = 20** is a 2× step; no hold-out of that length is possible with sizes
  4–10, so the data cannot validate it. The window sensitivity alone is up to
  1.8× for orders ≥ 2. The n = 20 points show the trend only; a ranking
  between two formulas at n = 20 that rests on a difference smaller than
  ~1.8× in r is not supported.
- **Lie-Trotter** at n = 13 and 20 should not be quoted as numbers.
- Comparisons of formulas *at measured sizes* (n ≤ 10) are unaffected.

This is not a confidence interval and no error model is fitted; with one
instance per size there is nothing to fit one to. The multi-seed run
(`../pf-seeds`) shows how much r moves between instances at fixed n.
