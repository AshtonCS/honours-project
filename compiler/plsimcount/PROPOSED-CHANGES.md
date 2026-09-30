# Proposed changes (not implemented)

Findings from the efficiency investigation of 2026-09-23. Nothing here has been
applied. Measurements are on the `pennylane` conda env, n=10 seed-1 Heisenberg
unless stated.

## Conclusion first

PennyLane's data structures are **slower** than the specialised code in almost
every place they could substitute, because they carry validation, wire
bookkeeping and dispatch overhead. Measured, mine vs PennyLane:

| Candidate | mine | PennyLane | verdict |
|---|---|---|---|
| commutation test (`nam/rules.py`) | 0.20 s / 3000 | `qml.is_commuting` 0.27 s | 1.4x slower |
| state-vector application (`nam/verify.py`) | 0.11 s | `default.qubit` 0.22 s | 2x slower |
| step unitary (`nam/gates.py`) | 0.010 s | `qml.matrix` 0.033 s | 3x slower |
| Pauli commutators (`tte.py`) | 0.055 ms | `PauliSentence` 0.295 ms | 5.4x slower |
| gate construction | 0.01 s / 20k | `qml.Hadamard` 0.06 s | 4.7x slower |
| peephole passes (5100 gates) | `nam` 0.25 s | `qml.transforms` 1.34 s | 5x slower |

**Do not substitute PennyLane into `tte.py`, `nam/`, `exponentiate.py`, or the
counting path.** The one real PennyLane win is `qml.matrix(qml.PauliRot(...))`,
which computes `exp(-i theta P)` 138x faster than `scipy.expm` (and 2x faster
than a hand-rolled `cos(theta) I - i sin(theta) P`).

The dominant cost in the project is not PennyLane at all -- it is
`empirical.py`, and the fixes below are mathematical.

## P1. Rewrite `empirical.Evolution.step` with signed-permutation Pauli application

**Why.** One `error()` call at n=10 costs 21.6 s, of which 16.4 s is 71 calls to
`scipy.expm`. Every Hamiltonian term is a Pauli word, so `P**2 = I` and no
matrix exponential is needed. Better, `P` is a *signed permutation*, so

    exp(-i theta P) @ U  =  cos(theta) U  -  i sin(theta) (P @ U)

and `P @ U` is a row permutation of `U` with phases: **O(dim^2) instead of
O(dim^3)**.

**Sketch** (prototyped and verified). For a Pauli word on `n` qubits, with
wire 0 the most significant bit:

```python
x  = sum(1 << (n-1-q) for q, c in enumerate(word) if c in "XY")   # bit flips
z  = sum(1 << (n-1-q) for q, c in enumerate(word) if c in "ZY")   # sign bits
yc = sum(1 for c in word if c == "Y")                             # i per Y
idx   = np.arange(dim) ^ x                     # row permutation
parity = popcount(np.arange(dim) & z) & 1      # via xor-shift folding
phase  = ((-1.0) ** parity) * ((-1j) ** yc)
# then, per scheduled term:
U = np.cos(th) * U - 1j * np.sin(th) * (phase[:, None] * U[idx, :])
```

**Measured.** Building one step, n=10 order 2 (71 exponentials):
**18.9 s -> 0.50 s (38x)**. Bit-identical to the current implementation across
n in {4, 6, 8, 10} x order in {1, 2, 4}; largest deviation 1.4e-15.

**Keep the `expm` path** as a slow reference implementation and have the tests
assert the two agree -- that is what makes this change safe to land.

## P2. Cache `Evolution.exact(t)`

**Why.** `error()` calls `exact(t)` on every evaluation; it is a 0.92 s `expm`
on a 1024x1024 matrix, and `t` is constant throughout a `steps()` search.
Roughly 20 s wasted per search at n=10.

## P3. Replace the spectral norm with `svds(k=1)`

**Why.** `np.linalg.norm(M, 2)` runs a full SVD (1.97 s at dim=1024) when only
the largest singular value is wanted.

**Measured** on a dense 1024x1024: full SVD 1.050 s; `scipy.sparse.linalg.svds(M,
k=1, return_singular_vectors=False)` 0.362 s with relative error 4.9e-16.

**Rejected alternative:** power iteration is the same speed (0.327 s) but only
6.8e-4 accurate *and can under-estimate*, which near the epsilon threshold
would return an `r` that does not actually meet the bound. Use `svds`.

## Combined effect of P1-P3

Per `error()` evaluation at n=10: **21.6 s -> ~1.3 s**.
A `steps()` search makes ~22 evaluations: **~475 s -> ~30 s (16x)**.

This is what makes `python -m plsimcount bounds -n 10 --empirical` usable, and
the empirical column is the only ground truth for how loose each bound is.

## P4. Add PennyLane as a *test oracle* (not in the hot path)

Both of these are worth being slower in a test:

* `qml.is_commuting` cross-checks `rules.commutes`. It agreed on every case
  tried. A wrong commutation rule silently corrupts circuits -- that has already
  happened twice in this project -- so an independent oracle earns its cost.
* `qml.matrix(qml.PauliRot(...))` cross-checks the P1 signed-permutation code.

## P5. Leave alone

`tte.py`, `nam/` (all modules), `exponentiate.py`, `resources.py`,
`circuit.py`, `synthesis.py`. Each was measured; the specialised code wins.

## Scope note

P1-P3 touch only `empirical.py`, which nothing else imports. Each is verifiable
against the current implementation, so they can land independently.
