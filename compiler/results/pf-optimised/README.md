# pf-optimised

Nam et al. "Light" optimiser (the reconstruction in `pysimcount/nam`, see its
README) applied to the **merged** cz step of five formulas — Suzuki S2, S4, S6,
Maležič–Ostmeyer q=6 (order 4) and q=14 (order 6) — on simcount seed 1,
n = 4–10, t = n, with r = the measured cz r of `../pf-thesis` (not
recomputed; the source file's sha256 is in `results.json`).

## Reproduce

From `compiler/`, `pennylane` environment, no Docker (~3 min):

```
python -m pysimcount.pfoptimised results/pf-optimised/config.json --render
```

Outputs: `results.json` / `results.csv` (one row per formula, n, metric),
`table.md` (CNOT, Rz, total), `figures/cnot-optimised.tex` (+ `all.pdf`),
`environment.txt`.

## Method

Per step, for CNOT, Rz and total gates:

- **unopt**: the step as compiled. Checked equal to pf-thesis's merged
  per-step row (test).
- **isolated**: Light on one step alone.
- **marginal**: Light on 3 consecutive steps minus Light on 2. This is the
  "marginal per block" of `nam/README.md`, the number it compares with the
  published optimised core `C` of `L · C^m · R`.
- **Totals** = marginal × r. This drops the one-off edge term
  opt(2 steps) − 2 × marginal, which is recorded per row (`edge`: 2n CNOT,
  n Rz, ≤ 54 gates here). Against totals of 4,000 or more CNOTs it is below
  0.2%.

**Soundness**: every optimised chain (1, 2 and 3 steps) was checked against
the unoptimised chain with `nam.verify.equivalent` (3 random states, up to
global phase). Worst error 5.4e-13. For n ≤ 6 the full unitaries were also
compared: worst 3.4e-13. So the optimiser preserved the unitary in every
case, not only the smallest.

**Caveat (from `nam/README.md`)**: this reconstruction recovers about 85% of
the published Light reduction on simcount's pf4 circuits (2.28× against the
paper's 2.68×, total gates). Optimised counts here are therefore expected to
sit *above* what the original optimiser would give. They are not a
reproduction of it.

## Results (seed 1, cz, merged steps)

| formula | CNOT factor | Rz factor | total-gate factor (n = 4 → 10) |
|---|---|---|---|
| Suzuki S2 | 1.25 | 1.20 | 2.29 → 2.16 |
| Suzuki S4 | 1.05 | 1.04 | 2.02 → 1.91 |
| M&O q=6 | 1.04 | 1.03 | 2.01 → 1.90 |
| Suzuki S6 | 1.01 | 1.01 | 1.97 → 1.86 |
| M&O q=14 | 1.02 | 1.01 | 1.98 → 1.87 |

(factor = unoptimised / optimised, marginal per step. The CNOT and Rz
factors are the same at every n.)

What this shows:

- **CNOT and Rz barely move.** The only CNOT/Rz saving is at step
  boundaries. The last half-step of one step and the first of the next
  apply the same bond terms, and merging them removes exactly 2n CNOTs and
  n Rz per step at every n and for every formula. Within a step, the merged
  schedule (`weighted.merge`) has already combined everything the optimiser
  can combine (isolated = unoptimised for CNOT and Rz in every row). The
  relative saving is 2n out of 10n for S2, and only 1–5% for the
  higher-order formulas, whose steps are longer.
- **Total gates fall about 2×**, almost entirely from single-qubit Clifford
  gates (H, S, S†) of simcount's basis-change template cancelling between
  adjacent exponentials.
- **Rankings by CNOT count do not change**. The optimiser removes the
  same 2n CNOTs per step from every formula, which is a slightly larger share
  for formulas with more steps (larger r), but nowhere enough to reorder
  them. M&O q=14 < M&O q=6 < S4 at every n here, before and after.
- The README's 2.28× was measured on *unmerged* simcount pf4 circuits. These
  rows start from merged steps, so the two factors are not directly
  comparable. Part of what the optimiser would find in an unmerged circuit
  has already been removed by merging.

## Assumptions

- `# ASSUMPTION: totals = marginal per step × r, dropping the constant edge
  term (recorded per row)` (unconfirmed; it is the README's per-block
  quantity scaled by r).
- r is pf-thesis's *unoptimised* measured r. Optimisation preserves the
  unitary, so r is unchanged by construction.
