# pf-thesis

Gate counts for every product formula in `pfcompare.formula_registry()` on
simcount's Heisenberg instances (seed 1, t = n), and the pgfplots figures made
from them. Supersedes `../pf-comparison` for figures: same method, plus
Lie-Trotter, plus rows whose r comes from a Trotter bound, and run from a
clean commit.

## Reproduce

From `compiler/`, in the `pennylane` conda environment (no Docker):

```
python -m plsimcount.pfcompare results/pf-thesis/config.json | tee results/pf-thesis/run.log
python -m plsimcount.pfreport  results/pf-thesis/results.json
python -m plsimcount.pffigure  results/pf-thesis/results.json --render
```

`--render` needs `tectonic` (`brew install tectonic`); it only compiles the
preview `figures/merged/all.pdf`. Without it, the `.tex` figures are still
written.

## Adding a formula

1. Add it to `formula_registry()` in `plsimcount/pfcompare.py`.
2. Add its id to `"schemes"` in `config.json`.
3. Rerun the three commands above.

It then appears in every figure it belongs to (its order's `B-` figure,
`B-all`, and the `A-`/`C-` figures if it is a Suzuki formula). Its style
follows from its family (colour) and order (marker). If it would look identical
to another formula in some figure, `pffigure` raises and names the formula to
add to `COLOUR_OVERRIDE`.

## How each number is made

As in `../pf-comparison/README.md` (measured r by `fastempirical`, simcount's
gate template, per-step counts x r, merging, pygridsynth synthesis, eps split
by gate set), plus:

- **Measured r is "keeps working", not "first that works"** (decided
  2026-10-21). Before the r^-p regime the error need not fall monotonically
  in r, so r is the smallest value from which r, r + 1, ..., r + 10 all meet
  eps (`fastempirical.WINDOW`). Passing values below it that fail again
  above are recorded per row in `search_dips`. In this run's predecessor this
  changed one row: M&O q = 14, n = 6, `cz`, from 14 (a dip; r = 15 fails) to
  16.
- **Bound rows** (`r_source` = `ana`, `min`, `com`, `emp`, `tte`): r from
  `bounds.steps`, for Suzuki formulas only, because the bounds are
  Suzuki-specific. `ana`/`min`/`com`/`emp` are simcount's `PF.hs` bounds;
  `tte` is Childs et al. 2021 (Props. 15, 16, J.1/J.2). Pairs a bound does not
  define (`com`/`tte` above order 4) are skipped and listed in
  `results.json` under `meta.skipped`. `emp` ignores t and eps by
  construction, so it gives the same r for `cz` and `ct`. Consequence, visible
  in the T-count figures: `ct` halves the Trotter budget (5e-4), so measured r
  rises (x2 for Lie-Trotter, x1.4 for S2) but `emp` does not, and the `emp`
  T counts sit *below* measured. They do not meet the `ct` accuracy; this is
  simcount's convention, reported as is, not corrected.
  The `com` row for Suzuki S2 at n = 4 reproduces simcount's own run
  (r = 5,503; 1,122,612 gates unmerged), which is checked in the tests.
- **Synthesis precision**: the bound rows need eps_rot down to ~1e-17.
  Every synthesised word is checked against eps_rot in mpmath
  (`synthesis.word_error`), so no T count rests on an unverified word.

## Choices made for this run

Agreed in conversation (2026-10-20/21):

- Figures use **merged** counts; unmerged figures can be made with
  `--unmerged`.
- Sizes are the committed simcount fixtures for seed 1 (n = 4-20). Measured r
  stops at n = 10; n = 13 and 20 at measured r are power-law extrapolations
  (open markers). Bound rows are computed at every size, not extrapolated.
- One seed, no averaging. Childs et al. 2018 average their T counts over
  random instances; these figures do not.
- T counts use pygridsynth, which gives ~4% more T gates than the Quipper
  gridsynth used by simcount.
- CNOT and total counts use the `cz` gate set (eps = 1e-3 all to Trotter); T
  counts use `ct` (5e-4 Trotter + 5e-4 synthesis).
