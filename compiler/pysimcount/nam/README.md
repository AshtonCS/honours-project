# `pysimcount.nam` — the Nam et al. circuit optimiser

A reimplementation of the optimiser in

> Y. Nam, N. J. Ross, Y. Su, A. M. Childs, D. Maslov.
> *Automated optimization of large quantum circuits with continuous parameters.*
> npj Quantum Information **4**, 23 (2018). arXiv:1710.07345.

The optimiser itself was never released — only the paper, and the
before/after circuits in `simcount/samples` and `njross/optimizer`. This module
reconstructs it from the publication and validates against those circuits.

## Status

| Subroutine | Status |
|---|---|
| 1. Hadamard gate reduction | implemented (`routines.routine1_hadamard_reduction`) |
| 2. Single-qubit cancellation + Rz merging | implemented |
| 3. Two-qubit (CNOT) cancellation | implemented |
| 4. Rotation merging via phase polynomials | implemented (`routine4_rotation_merging`) |
| 5. Floating Rz gates | not implementable (see below) |

`Light = 1,3,2,3,1,2,4,3,2` and `Heavy = 1,3,2,3,1,2,5`, each repeated to a
fixpoint (§4.5 of the paper).

**Routine 5 cannot be reproduced.** It depends on rewrite-rule library files
that were never published; Figures 6 and 7 show examples only. The paper
reports that for product-formula circuits Heavy gives no improvement over
Light, so nothing is lost.

**Routine 4 is implemented in simplified form.** The paper works with affine
functions — XORs that may carry a trailing `+ 1` — and spends a step
propagating NOT gates to remove those offsets. None of that is needed here:
these circuits carry no X gates, so every function is a plain XOR and the
offsets never arise. What remains is a conservative choice of *stretch*: the
circuit is split at every Hadamard, including ones on unrelated wires, where
the paper grows stretches per wire. That finds fewer merges, so the routine
falls short of the published figures by a known and documented amount.

## Where the Light optimiser gets to

Their published circuit is `L · C^m · R`, so **`C` assumes a neighbour on both
sides** and `L`/`R` absorb the ends. Optimising one isolated block therefore
overstates the per-block cost; the number to compare against `C` is the
*marginal* cost of one more block in a repeated chain. Both are given below,
because the isolated figure is what a naive comparison produces.

pf4com, n=13:

| | isolated block | marginal per block | their `C` |
|---|---|---|---|
| Routines 1–3 only | 1,674 | 1,640 (2.02×) | — |
| with Routine 4 | 1,524 | **1,455 (2.28×)** | 1,235 (2.68×) |

So the reconstruction recovers **85% of the published reduction**, and it does so
consistently:

| case | marginal | their `C` | ours | theirs | recovered |
|---|---|---|---|---|---|
| pf4ana n=13 | 1,455 | 1,235 | 2.28× | 2.68× | 85% |
| pf4min n=13 | 1,455 | 1,235 | 2.28× | 2.68× | 85% |
| pf4emp n=13 | 1,455 | 1,235 | 2.28× | 2.68× | 85% |
| pf4ana n=16 | 1,800 | 1,520 | 2.27× | 2.68× | 84% |
| pf4ana n=20 | 2,260 | 1,900 | 2.26× | 2.68× | 84% |
| pf6emp n=13 | 7,275 | 6,175 | 2.28× | 2.68× | 85% |
| pf6emp n=16 | 9,000 | 7,600 | 2.27× | 2.68× | 84% |

Stable across system size, formula order and coefficient regime — the pf4
variants differ in rotation angle by three orders of magnitude — so the
shortfall is a systematic missing mechanism rather than a per-case artefact.

## The residual 15%

Every mechanism recoverable from the publication was implemented and measured.
Only one helped; the rest are recorded here so the ruling-out is not repeated:

| mechanism | source | outcome |
|---|---|---|
| Per-wire stretch growth in Routine 4 | §4.4 | **the only gain**, 2.02× → 2.28× |
| Uniform `{NOT, H, Rz, CNOT}` gate set | §4.1 | a real inconsistency in our code; costs zero gates |
| Multi-gate commutation patterns | Fig 5 | the one recoverable pattern fires **0 times** |
| Generalised Fig 4 rule (several CNOTs sharing a target) | Fig 4 caption | verified correct, but blocked by the Rz our CNOT ladders always carry between them |
| `LCRoptimizer` | §4.6 | equivalent to the marginal measurement above |
| Commuting through the product of intervening gates | — | made results *worse* (1,455 → 1,835) |

The Light sequence also reaches its fixpoint in a single round, so the shortfall
is not insufficient iteration. The residual is attributable to implementation
detail not recoverable from the paper, which withheld its source.


## Validation

Two independent layers (`tests/test_nam.py`, 57 tests):

1. **Soundness.** Every rewrite preserves the unitary up to global phase. Full
   unitary comparison for n ≤ 6; random-state equivalence (`verify.equivalent`)
   at n = 13, where a 8192² unitary is out of reach but applying a circuit to a
   state vector is not.
2. **Fidelity.** `samples.compare(algo, n)` puts our factors beside theirs.
   50 cz files are on disk, but they are *not* 50 independent cases: the four
   pf4 variants at a given size are the same circuit, with identical gate counts
   and identical targets, differing only in rotation angle. So the structural
   coverage is **2 shapes × 10 sizes**, while the angle variants test numerical
   robustness across three orders of magnitude — which is what caught both
   tolerance bugs below. Sizes above n=20 are untested; the passes are O(g²).

Global phase is never tracked: two of the Figure 4 identities hold only up to
`exp(i pi/4)`, and the reference circuits carry no phase gates in cz mode.

## Two bugs worth remembering

Both were found by the soundness layer, and both are the same mistake — an
absolute tolerance where a relative one belongs:

1. `Gate.cancels_with` compared rotation angles with an absolute tolerance, so
   two small **same-sign** rotations tested as cancelling. Deleting them let
   the surrounding CNOTs become adjacent and cascade away; the pf4ana samples
   collapsed to **zero** CNOT gates, reported as a spectacular 24× reduction.
2. `np.allclose(..., atol=1e-12)` keeps numpy's **default `rtol=1e-05`**. Any
   rotation with |θ| ≲ 1e-5 was therefore declared to commute with gates it
   does not commute with. Angles here are `t/r` and reach 1e-7, so every
   circuit from a loose bound was silently corrupted. pf4com's smallest angle
   is 1.15e-5 — just above the threshold, so it passed by luck, while pf4ana's
   8e-7 broke. Gate counts alone looked entirely plausible (2.10× vs 2.68×).

Gate counts cannot detect either failure. Do not trust a reduction factor that
has not been paired with an equivalence check.

## Deviations from the paper

* **Commutation is exact, not pattern-based.** The paper uses a fixed list
  (Figure 5), described as sufficient but not necessary. `rules.commutes`
  instead tests whether the gate matrices commute on their shared support, a
  strict superset — so our counts may come out slightly *lower* than theirs.
  `rules.pattern_commutes` gives the restricted behaviour for a like-for-like
  comparison.
* **No explicit DAG.** Routines 2 and 3 walk the flat gate list rightwards,
  which is the same algorithm at the same O(g²) complexity. Routine 1 uses
  per-wire neighbour maps, rebuilt once per pass.
* **Figures 4 and 5 do not extract cleanly from the PDF.** The transcription in
  `rules.py` was reconstructed and then verified numerically; every identity is
  checked by `rules.verify_hadamard_rules`, which is also a test.

## Usage

```python
from pysimcount.nam import bridge, optimise, samples, verify

# our own circuits
circuit = bridge.from_tape(sim.step, n_qubits)
result  = optimise(circuit)                            # full Light sequence
print(result.before, result.after, result.factor("CNOT"))

# against the published reference
print(samples.compare("pf4com", 13).render())
ok, err = verify.equivalent(circuit, result.circuit)
```

## Files

| file | contents |
|---|---|
| `gates.py` | `Gate`, `Circuit`, matrices, conventions |
| `rules.py` | Figure 4 identities, commutation tests, numerical verification |
| `routines.py` | Routines 1–4 |
| `optimiser.py` | Light / Heavy sequences, fixpoint driver |
| `quipper.py` | reader for Quipper ASCII circuits |
| `samples.py` | load the reference pairs, compare, render |
| `verify.py` | unitary and random-state equivalence |
| `bridge.py` | conversion to and from PennyLane tapes |
