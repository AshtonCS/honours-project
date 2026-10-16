# pf-comparison

Suzuki product formulas (simcount's own) against the schemes of Maležič &
Ostmeyer, arXiv:2601.18756, on simcount's Heisenberg instances.

## Files

| File | What it is |
|---|---|
| `config.json` | Every setting of the run. The only input besides the code and the committed simcount fixtures. |
| `results.csv`, `results.json` | One row per (instance, n, scheme, gate set, merged). `results.json` also records the git commit, a dirty flag, a sha256 of every source file used, and the fitted power laws. |
| `report.md` | Tables made from `results.json` by `python -m plsimcount.pfreport`. |
| `run.log` | Console output of the run. |

## Reproduce

From `compiler/`, in the `pennylane` conda environment:

```
python -m plsimcount.pfcompare results/pf-comparison/config.json
python -m plsimcount.pfreport  results/pf-comparison/results.json
```

No Docker needed: the Hamiltonians come from `plsimcount/fixtures/simcount/`
(seed in the config). The scheme weights come from `compiler/effts`.

## How each number is made

- **r**: the smallest Trotter number whose *measured* spectral-norm error at
  time t = n is at most eps_trotter (`plsimcount/fastempirical.py`). The same
  search is used for every formula, Suzuki included. It is exact for each
  instance and **not a bound**. It reproduces `empirical.py` (tests, and
  r = 3,174 / 152 at n = 10, orders 2 / 4, from `bounds_n10_empirical.txt`).
- **Gates**: one step compiled with simcount's gate template; totals are
  per-step × r, as in simcount.
- **Merged / unmerged**: merging combines repeated exponentials inside each
  block of mutually commuting terms (`plsimcount/weighted.py::merge`). It
  never changes the unitary, so r is the same for both. It is applied
  identically to every formula. Unmerged is simcount's own convention.
  **Use the merged numbers for comparisons between families.** Unmerged is
  not like-for-like: simcount's second-order step already applies the fields
  once in the middle (a built-in partial merge), giving 7n rotations per
  cycle, while the ramp form as written in the paper (Eq. 7) gives 8n. After
  merging, a step of q cycles costs 5n·q + n rotations, i.e. 5n + n/q per
  cycle (at n = 10: 60 for S2, 52 for S4, 50.7 for q = 14). The extra n per
  step is the XX group at the step boundary, which merging within one step
  cannot remove; merging across steps would remove it and give exactly the
  paper's "cost ∝ q" model (5n per cycle). Totals here are per-step × r, so
  they do not merge across steps; this over-counts low-q formulas slightly
  (S2 by 16.7%, S4 by 3.8%, q = 14 by 1.4% at n = 10).
- **ct**: each rotation is synthesised (pygridsynth) to
  eps_rot = eps_synth / (rotations per step × r), as
  `synthesis.synthesise_step` does. pygridsynth gives ~4% more T gates than
  Quipper's gridsynth (CLAUDE.md); both formulas are affected equally.
- **eps**: `cz` gives all of 1e-3 to Trotter error. `ct` gives 5e-4 to Trotter
  and 5e-4 to synthesis, so r differs between the two gate sets.

## Choices made for this run

Agreed in conversation (2026-10-14):

1. Merging: both reported, merged as the headline.
2. Gate sets: `cz` and `ct`, with `ct` across sizes to show scaling.
3. Schemes: Suzuki S2/S4/S6/S8; the paper's recommended q = 6 (order 4) and
   q = 14 (order 6); Blanes & Moan q = 10 (order 6, best previously known);
   the highest-Eff order-8 scheme at q = 21 as an exploratory row (the paper
   recommends no order-8 scheme in its text).
4. Instance: simcount's fixtures, seed 1, t = n; plus the paper's own instance
   (fields ±0.1, upstream seed 42) at n = 6, t = 10, `cz` only.
5. Results recorded here, with provenance.

Added 2026-10-15, chosen before seeing any results for them:

6. Morales et al. (arXiv:2210.15817) unprocessed order-8 formulas, weights
   extracted from the arXiv LaTeX source (`effts/morales.py`):
   - Y8m10 (21 Leapfrog steps): their best unprocessed order-8 formula on
     spectral-norm error (their Table 8), matching simcount's error measure.
     The eigenvalue-optimised Y8m10b is not included.
   - m = 8 (17 steps): presumably the "Morales et al. (q=17)" scheme in
     Maležič & Ostmeyer's Figs. 5-6, to reproduce the paper's own comparison.
   Processed formulas (their Table 3) are out of scope: they need a
   P K^r P^-1 pipeline and a processor-cost convention.

Not agreed explicitly, chosen for this run:

- **Fit range.** The extrapolated rows (n = 13, 20) use r = ceil(a n^b) fitted on
  n = 6, 7, 8, 10. n = 4, 5 are left out of the fit because the smallest
  chains are least typical. This is a modelling choice: the fitted rows are
  labelled `fit` and are not measurements.
- **One instance per n.** Only seed 1 has fixtures for every size. One field
  pattern per size can be atypical; more seeds need Docker to generate.

## Limits

- Measured sizes are n = 4–8 and 10 (the committed fixtures; n = 9 has none).
- The Blanes & Moan weights are upstream's copy and have not been checked
  against Blanes & Moan (2002).
- The order-8 row is our pick of scheme, not the paper's recommendation.
