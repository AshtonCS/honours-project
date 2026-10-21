# pf-seeds

The pf-thesis measurement repeated on five simcount instances (seeds 1–5) to
see how much r and the gate counts move between instances, and whether any
formula ranking depends on the instance.

Settings per seed (`seed<k>/config.json`, identical except the seed): all
11 pf-thesis formulas, measured r at n = 4–10, extrapolated r at n = 13, 20
(power law fitted on n = 6, 7, 8, 10). Gate sets cz and ct, merged and
unmerged. t = n; eps as in pf-thesis (cz: 1e-3; ct: 5e-4 Trotter + 5e-4
synthesis). Synthesis is pygridsynth. No bound rows, no paper instance.

`# ASSUMPTION: seeds 1–5, n = 4–20, same t = n and eps as pf-thesis`
(unconfirmed; as specified for this run).

## Reproduce

Fixtures for seeds 2–5 were recorded with Docker (`simcount-new`), one per
(n, seed), via `python -m plsimcount compare -n N -a pf2emp -g cz -m SEED`.
They are committed, and every one matched simcount's own gate count. Then,
from `compiler/` (no Docker):

```
python -m plsimcount.pfcompare results/pf-seeds/seed<k>/config.json > results/pf-seeds/seed<k>/run.log   # k = 1..5
python -m plsimcount.pfseeds   results/pf-seeds/aggregate.json --render
```

The five runs took ~68 min each, run in parallel on 8 cores alongside other
jobs. `pfseeds` refuses runs made from uncommitted code or with configs that
differ in more than the seed.

Outputs in `aggregate/`: `spread.md`/`.csv` (every value per seed, median,
min, max, max/min), `rankings.md` (cost-order reversals relative to seed 1,
cheapest formula per seed), `reproduction.md`, `meta.json`, and
`figures/merged/` (15 figures: the A-measured and B families of
`pffigure`, median over seeds with min–max bars, measured r only).

## Check: seed 1 reproduces pf-thesis

All 352 seed-1 rows (measured and fitted r, both gate sets, merged and
unmerged) equal pf-thesis's simcount rows **field for field**, wall-clock
time excepted. That includes r, both errors at the boundary, the number
of search evaluations and every gate/T count (`reproduction.md`).

## Caution: the five seeds are correlated instances

Seeds 1–5 give different fields, but **not independent random ones**. For
every site j, consecutive seeds shift the field by an almost constant
amount, modulo the field range of width 2:

    h_j(seed + 1) - h_j(seed) ≈ Δ_j  (mod 2),   with deviations ~0.01

(e.g. site 0: −0.844, −0.653, −0.461, −0.269, ... steps of +0.192). Checked
at n = 6 and n = 10 for seeds 1–4. The five instances therefore lie close to
a line (on the torus) in field space. This looks like the known weakness of
seeding Haskell's `StdGen` with small consecutive integers. I have not
confirmed that against the `random` package source or the version simcount
uses. Consequence: the spread below is the spread over *these five*
instances. It may understate the variation over genuinely random fields.
More widely separated seeds (or a check of how simcount seeds its generator)
would be needed before calling it instance-to-instance variation in general.

## Spread across seeds (merged; `spread.md`)

Largest max/min of r over the five seeds, cz (gate totals in cz scale
exactly with r, so CNOT and total have the same ratios):

| | measured n = 4–10 | extrapolated n = 13, 20 |
|---|---|---|
| Lie-Trotter | 1.74 | 2.08 |
| Suzuki S2 | 1.31 | 1.49 |
| Suzuki S4 / S6 / S8 | 1.22 / 1.15 / 1.25 | 1.22 / 1.27 / 1.19 |
| M&O q=6 / q=14 | 1.14 / 1.14 | 1.22 / 1.18 |
| Blanes–Moan q=10 | 1.09 | 1.23 |
| order-8 non-Suzuki | 1.10–1.15 | 1.12–1.25 |

T counts (ct) behave the same (max/min up to 1.37 for S2 and 1.81 for
Lie-Trotter at measured n). Excluding Lie-Trotter, no formula's r varies by
more than 1.31× across seeds at a measured size. **Seed 1 is not a typical
instance**: its r is the largest of the five in 37 of 66 (formula, n) cases
and the smallest in 17. seed-1 r / median ranges 0.76–1.15, with a median
of 1.03.

## Rankings (`rankings.md`)

94 pairwise cost-order reversals relative to seed 1, counted over every
metric, size and seed. Almost all are between formulas whose costs differ
by little: the largest seed-1 gap that any other seed reverses is 18%.

- **The cheapest formula is the same on every seed, at every n and for
  every metric**: M&O q=14 (CNOT, total and T count).
- **M&O vs Suzuki at equal order never reverses.** M&O q=6 < S4 and
  M&O q=14 < S6 on every seed and size, and S8 is the costliest order-8
  formula on every seed.
- The 10 equal-order reversals are all *among the three non-Suzuki order-8
  schemes* (M&O q=21, Morales Y8m10, Morales m=8). Where they reverse, seed 1
  has them within 0.4–6.4% of each other, so their mutual order is instance-dependent and should
  not be reported as a ranking.
- Cross-order reversals (84; 53 at measured n, 31 at extrapolated n) mostly
  involve the order-8 non-Suzuki schemes against M&O q=6, Blanes–Moan q=10
  or S4. Those formulas are within ~18% of each other at every n, and
  every such pair reverses on at least one seed.
- **Suzuki S4 vs S6 reverses at extrapolated sizes**: n = 13 (T count,
  seeds 3, 4) and n = 20 (CNOT and total, seed 5). Which Suzuki order is
  cheapest at n = 13–20 is therefore not settled by these data. That fits
  `../pf-fitcheck` (extrapolated r uncertain at the 20–35% level at n = 13).

## Figures

Same families and styles as `pffigure` (colour by family, marker by order),
filled markers for measured r, with the median over seeds as the point and
the min–max range as the bar. Extrapolated n = 13, 20 are deliberately left
out of the figures: across seeds they combine instance variation with
fit-window variation (see `../pf-fitcheck`). They are in the tables.
