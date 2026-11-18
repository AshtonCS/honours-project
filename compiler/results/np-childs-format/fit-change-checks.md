# Checks for the n = 9 / 11 runs used by the n = 5–11 fit (2026-11-08)

`np-main-n9` and `np-main-n11` were run at commit e42ff29 with uncommitted changes (`git_dirty`),
which `pfseeds.load_runs` refuses. The figures that fit n = 5–11 accept them through the
`dirty_exception` in `config.json` and `t-config.json`. This file records why.

## What was uncommitted

`git diff e42ff29 -- pysimcount/` (tracked files): one change, in `pfseeds.py` — `load_runs` gained a
`per_seed_keys` argument. `pfcompare` (which produced the runs) does not import `pfseeds`. Between
np-main's commit 01d3c27 and e42ff29 the only commit touching the simulation code is the
plsimcount → pysimcount rename.

## Recomputation from committed code

Clean `git worktree` at e42ff29 (`git_dirty=False`), `python -m pysimcount.pfcompare`, seed 1:

| size | formulas | gate sets | rows |
|---|---|---|---:|
| n = 9 | Suzuki S4, S6; M&O q=6, q=14, q=21 order 8; Blanes & Moan q=10; Morales Y8m10, m=8 | cz, ct | 32 |
| n = 11 | M&O q=14, Blanes & Moan q=10 | cz, ct | 8 |

Compared with the stored rows on r, CNOT, total gates, Rz, T, eps_rot, err_at_r and
err_at_r_minus_1, merged and unmerged: **40 of 40 rows identical, 0 mismatches.** The NumPy fields of
both instances are identical too. The recomputed results are kept beside this file as
`fit-change-spot9-seed1.json` and `fit-change-spot11-seed1.json`.

## Further check inside the T run

`pftchilds counts` (fit n = 5–11) recomputed every merged ct step at n = 11 (8 formulas × 10 seeds)
at the stored measured r: **80 of 80 T totals identical** to the n = 11 runs.

## Not covered

r at n = 9 for seeds 2–10 and at n = 11 for six formulas and seeds 2–10 were not recomputed (n = 11
took 108 CPU-hours in total). The checks above show the code path reproduces the stored values; they
do not re-derive every row.

# T-count cache made before the code commit (2026-11-09)

`t-counts-fit5-11.json` (720 synthesised steps behind `main-t-all-formulas` and its dot plots) was computed
2026-11-08/09 from uncommitted code at e42ff29 (cache provenance `git_dirty=True`); the code was committed
afterwards (f72498b) without changes to `pftchilds.py` or the synthesis path. Rather than re-running ~8 h, the
cache is accepted as a documented exception, checked as follows.

Recomputed at the clean commit e5d9ee6 (`git_dirty=False`) with `pftchilds.synthesised_step`, same template
(simcount) and backend (pygridsynth): every shown formula at n = 13 (seed 1) and n = 20 (seed 7), and M&O q=14,
Blanes & Moan, Morales m=8 at n = 50 (seed 3). **19 of 19 identical**: T total, rotations per step and eps_rot.
Log: `t-cache-spotcheck.log`. Together with the n = 11 check inside the run (80/80 steps reproduce the runs'
T totals), the cache is reproduced by committed code wherever it was tested; the n >= 70 steps were not
recomputed (12–29 min each at n = 500).
