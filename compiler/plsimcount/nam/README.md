# `plsimcount.nam` — the Nam et al. circuit optimiser

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
| **4. Rotation merging via phase polynomials** | **stub — to be written by hand** |
| 5. Floating Rz gates | not implementable (see below) |

`Light = 1,3,2,3,1,2,4,3,2` and `Heavy = 1,3,2,3,1,2,5`, each repeated to a
fixpoint (§4.5 of the paper).

**Routine 5 cannot be reproduced.** It depends on rewrite-rule library files
that were never published; Figures 6 and 7 show examples only. The paper
reports that for product-formula circuits Heavy gives no improvement over
Light, so nothing is lost.

**Routine 4 is deliberately left unimplemented.** It is the subroutine with
the actual intellectual content. Its docstring sets out the five steps from
§4.4 (subcircuit identification, pruning, affine functions over GF(2),
leftmost-placement merging, NOT-propagation). `optimise(...)` raises unless
`skip_unimplemented=True`, so a run without it can never be mistaken for a
complete result.

## Where Routines 1–3 get to

Per core block, pf4com n=13 — their pre-optimisation circuit in, their
post-optimisation circuit as the target:

| gate | pre | ours (1–3) | theirs (Light) | ours | target |
|---|---|---|---|---|---|
| H | 1040 | 286 | 260 | 3.64× | 4.00× |
| S + Sdg | 1040 | 372 | 130 | 2.80× | 8.00× |
| CNOT | 780 | 772 | 520 | 1.01× | 1.50× |
| Rz | 455 | 451 | 325 | 1.01× | 1.40× |
| **total** | **3315** | **1881** | **1235** | **1.76×** | **2.68×** |

The split is informative: Hadamard reduction is nearly at target, while CNOT
and Rz have barely moved. That is exactly the expected shape — phase-polynomial
merging (Routine 4) is what reduces the rotation count, and the placement
freedom it creates is what then lets CNOT gates cancel. **Routine 4 has to
deliver CNOT 780 → 520 and Rz 455 → 325.**

Factors are near-identical across `pf4ana/min/com/emp` and `pf6emp` and across
n, because the circuit *structure* is the same for every bound — only `r` and
the rotation angles differ.

## Validation

Two independent layers (`tests/test_nam.py`, 43 tests):

1. **Soundness.** Every rewrite preserves the unitary up to global phase. Full
   unitary comparison for n ≤ 6; random-state equivalence (`verify.equivalent`)
   at n = 13, where a 8192² unitary is out of reach but applying a circuit to a
   state vector is not.
2. **Fidelity.** `samples.compare(algo, n)` puts our factors beside theirs.
   50 cz cases on disk: 5 algorithms × 10 sizes (13 … 100), plus ct variants.

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
from plsimcount.nam import bridge, optimise, samples, verify

# our own circuits
circuit = bridge.from_tape(sim.step, n_qubits)
result  = optimise(circuit, skip_unimplemented=True)   # until Routine 4 lands
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
| `routines.py` | Routines 1–3; Routine 4 stub with its specification |
| `optimiser.py` | Light / Heavy sequences, fixpoint driver |
| `quipper.py` | reader for Quipper ASCII circuits |
| `samples.py` | load the reference pairs, compare, render |
| `verify.py` | unitary and random-state equivalence |
| `bridge.py` | conversion to and from PennyLane tapes |
