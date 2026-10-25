# pf-grouping

The same Hamiltonian and the same Suzuki recursion (S2, S4), compiled four
ways. simcount seed 1, even n = 4–10, t = n, cz (eps_trotter = 1e-3, no
synthesis), r measured as in pf-thesis (smallest r with spectral error
≤ eps, `fastempirical.steps`).

| | grouping per step | compilation | r |
|---|---|---|---|
| (a) `xyz` | XX \| YY \| ZZ+Z (simcount's order; pf-thesis baseline) | each Pauli term separately | measured |
| (b) `evenodd` | even bonds \| odd bonds \| fields (`evenodd.even_odd`) | each term separately | measured |
| (c) `blocked` | as (b) | each bond's XX+YY+ZZ as one 3-CNOT block (`blocks.py`, Vatan–Williams) | = r(b): same unitary |
| (d) `twogroup-*-out` | E' = even bonds + fields \| O = odd bonds (`twogroup.py`) | every bond one exact block (E' blocks 3 CNOT + 7 Rz, O blocks 3 CNOT + 3 Rz) | measured, exact group exponentials |

(d) is run for both choices of which group is the outer one (applied
twice per S2).

## Reproduce

From `compiler/`, `pennylane` environment, no Docker (~7 min while other
jobs were running):

```
python -m pysimcount.pfgrouping results/pf-grouping/config.json --render
```

Outputs: `results.json`/`.csv`, `table.md` (every number), figures
`figures/{cnot,total}-S{2,4}-by-grouping.tex` and `figures/all.pdf`, and
`environment.txt`.

## Conventions

- **Counts are unmerged for all four variants**, so that no variant gets a
  merge the others do not (the reasoning in `blocks.py`). In S4 the blocked
  and two-group variants still emit two adjacent copies of some blocks,
  which could be merged as well. (a) and (b) are also given merged in
  `table.md`; the pf-thesis figures use merged counts.
- (a) reproduces pf-thesis's measured r exactly (S2: 699, 1,189, 2,102,
  3,174; S4: 45, 69, 111, 152), which checks that this run and pf-thesis use
  the same search.
- The search for (b) and (d) was checked against the slow reference searches
  (`empirical.steps`, `twogroup.steps_empirical`) at n = 4 in the tests.

## Results: where the differences come from

Total CNOT = r × CNOT per step, so each ratio against (a) splits into an
r factor (grouping changes the Trotter error) and a per-step factor
(compilation changes the gates). Ratios against (a) unmerged:

| | r factor, S2 (n = 6–10) | r factor, S4 (n = 4–10) | CNOT/step factor | CNOT total, S2 n = 10 | CNOT total, S4 n = 10 |
|---|---|---|---|---|---|
| (b) even/odd | 1.12–1.13 (1.01 at n = 4) | 0.93–0.97 | 1 | 1.12 | 0.97 |
| (c) blocked | same as (b) | same as (b) | 0.5 | 0.56 | 0.48 |
| (d) E' outer | 1.08–1.12 (1.03 at n = 4) | 0.87–0.91 | 0.375 | 0.41 | 0.34 |
| (d) O outer | 1.08–1.12 (1.03 at n = 4) | 0.89–0.95 | 0.375 | 0.41 | 0.36 |

Against (a) *merged* (the pf-thesis convention; merging removes the
duplicate ZZ half-steps of (a)), (c) is at 0.61–0.69 and (d) at 0.46–0.51
of the CNOT total for both orders and all n.

What the data say (seed 1 only):

- **Most of the CNOT saving comes from gates per step, not from r.** The
  per-step factor is exact and independent of n: 2 → 1 CNOT per bond term
  for blocks (6 → 3 per bond), and 3 CNOT per bond with fields folded in for
  two-group.
- **Grouping changes r only modestly, and in opposite directions for the
  two orders.** For S2 the even/odd ordering needs 12–13% *more* steps than
  X-Y-Z (n ≥ 6), and two-group 8–12% more. For S4 even/odd needs 3–7%
  *fewer*, and two-group 5–13% fewer. So "even/odd is better" is not true in
  general for this model: at second order it costs steps, and the CNOT
  saving of (c) and (d) more than pays for it.
- **The outer-group choice in (d)** gives the same CNOT count. O outer has
  fewer Rz and total gates (the 7-rotation E' blocks are applied once per
  S2 instead of twice), and E' outer has a slightly smaller r at S4.
- **Total gates** follow the same pattern, with smaller factors, because
  single-qubit gates do not shrink as much as CNOTs (the E' blocks carry 7
  rotations).

## Assumptions (all unconfirmed)

- `evenodd.even_odd`: even bonds outermost, odd next, fields innermost
  (pre-existing `# ASSUMPTION` in `evenodd.py`).
- (d) reported for both outer choices; neither is preferred.
- Unmerged counts as the like-for-like comparison (above).
- Results hold for one instance (seed 1) and t = n, eps = 1e-3. With the
  r differences at 3–13%, the seed-to-seed spread of r (`../pf-seeds`) is
  the relevant scale before generalising the r effects. The per-step
  factors are structural and do not depend on the instance.
