# pysimcount

A PennyLane re-implementation of the product-formula pipeline in
[njross/simcount](https://github.com/njross/simcount) (Childs, Maslov, Nam,
Ross, Su — *Toward the first quantum simulation with quantum speedup*),
built so that every stage can be swapped independently and every number can
be checked against the Haskell/Quipper original gate for gate.

Renamed from `plsimcount` on 2026-10-25; the code is otherwise unchanged.
Run records written before then (`results/*/results.json`, `config.json`)
still say `plsimcount` in their descriptions; reproduce them with
`python -m pysimcount ...`, or check out the commit they record.

```
python -m pysimcount compare -n 4 -a pf2com -g cz            # side by side with simcount
python -m pysimcount compare -n 4 -a pf2com -g ct            # ... through Clifford+T
python -m pysimcount compare -n 4 -a pf2com -g ct --synth quipper   # exact reference synthesis
python -m pysimcount compare -n 4 -a pf2com -g cz --optimise 2      # + opt-in optimisation
python -m pysimcount run -n 6 -a pf4com -g cz --fields 0.1 -0.3 0.5 0.2 -0.7 0.4
python -m pysimcount bounds -n 10 --empirical                # r across all bounds (+ brute force)
python -m pysimcount run -n 10 -a pf4tte -g ct               # tighter bound, same pipeline
python -m pytest pysimcount/tests                             # parity + physics checks
```

Run from `compiler/` with the `pennylane` conda environment
(`/opt/anaconda3/envs/pennylane/bin/python`). Requires `pygridsynth` for
`-g ct`; `compare` additionally needs the `simcount-new` Docker container
unless a fixture for the case already exists under `fixtures/simcount/`.

## Pipeline

| stage | module | simcount counterpart |
|---|---|---|
| Hamiltonian as ordered `(coeff, PauliWord)` list | `hamiltonian.py` | `Definitions.hs` `random_h`, `Hamiltonian` |
| step count `r` from a bound (ana / min / com / emp) | `bounds.py` | `PF.hs` `time_slice_*` |
| `tte` bound: nested-commutator bounds of Childs et al. 2021 | `tte.py` | — (post-dates simcount) |
| brute-force empirical `r` for small systems | `empirical.py` | "empirical" data in the paper |
| product formula as a schedule `[(term, time)]` | `formula.py` | `PF.hs` `simulate_1st`, `simulate_2kth` |
| Pauli exponential → `{H, S, S†, CNOT, RZ}` | `exponentiate.py` | `Definitions.hs` `exponentiate_Pauli` |
| one step as a PennyLane tape; counts = step × r | `circuit.py`, `resources.py` | `nbox "S" r`, `-f gatecount` |
| RZ → Clifford+T with ε_rot = ε_synth / zrots | `synthesis.py` | `-g ct`, Quipper `Approximate` |
| opt-in peephole optimisation over blocks of k steps | `optimise.py` | (Nam et al. optimiser, separate tool) |
| run simcount in Docker, parse, cache fixtures | `simcount.py` | — |
| side-by-side report | `compare.py` | — |

Defaults follow `Main.hs`: `t = n`, ε = 1e-3 (`cz`), 5e-4 (`cz2ct`),
5e-4 + 5e-4 (`ct`), periodic chain for n ≥ 3, `Jx = Jy = Jz = 1`.

## Design decisions

* **Step × r is the baseline, not an approximation.** simcount boxes the
  step as a subroutine and multiplies. The full circuit is only ever
  materialised when `--optimise k` asks for k consecutive steps.
* **Cliffords stay Cliffords.** Gates are emitted directly into
  `{H, S, S†, CNOT, RZ}`; nothing is compiled to `RX/RY/RZ`. That keeps the
  RZ count equal to the number of genuine rotations, which is what the
  synthesis budget is divided over.
* **Term order is explicit.** simcount's Heisenberg order is Pauli-major
  (all XX, all YY, all ZZ, then Z fields). `hamiltonian.from_pennylane`
  preserves PennyLane's order if you want that instead. The commutator bound
  constants (`com1/com2/com4`) are only valid for the Heisenberg chain.
* **Synthesis is verified.** Every gridsynth output is multiplied out and
  rejected if its error exceeds ε_rot. PennyLane 0.45's own
  `rs_decomposition` / `clifford_t_decomposition` silently return wrong
  sequences for ε ≲ 1e-8, which is the regime these circuits need
  (ε_rot ≈ 2e-9 already at n = 4). `pygridsynth` is the default backend;
  `--synth quipper` shells out to Ross's reference binary in the container
  for exact parity (≈ 20 s per distinct angle).
* **Random fields come from simcount.** Haskell's `StdGen` cannot be
  reproduced in Python, so `compare` reads the Hamiltonian simcount prints
  to stderr; `run` takes `--fields` or `--hamiltonian`.

## Verified parity (seed 1)

| case | r | total gates | status |
|---|---|---|---|
| n=4 pf1com cz | 640,137 | 66,574,248 | exact |
| n=4 pf2com cz | 5,503 | 1,122,612 | exact |
| n=4 pf2com cz2ct | 7,629 | 1,556,316 | exact |
| n=4 pf2com ct (`--synth quipper`) | 7,629 | 48,398,376 (18,416,406 T) | exact |
| n=4 pf2com ct (pygridsynth) | 7,629 | 51,205,848 (19,225,080 T) | +4.4 % T |
| n=4 pf4com cz | 1,349 | 1,375,980 | exact |

`tests/test_simcount_parity.py` replays every recorded fixture.

## The `tte` bound (Theory of Trotter Error with Commutator Scaling)

`tte.py` evaluates Propositions 15 (S₁), 16 (S₂) and J.1/J.2 (S₄ with two /
three commuting groups, Table 2) of Childs, Su, Tran, Wiebe & Zhu (PRX 2021)
for the actual Hamiltonian instance. simcount's PF2/PF4 step is exactly S₂/S₄
over the three commuting groups H₁=ΣXX, H₂=ΣYY, H₃=ΣZZ+Σh·Z (the paper's
X‑Y‑Z ordering), so the propositions apply directly and the result plugs in
as `pf{1,2,4}tte`. Nested commutators are computed in a bitmask Pauli
algebra; norms follow the paper's lattice recipe (innermost term fixed,
exact norm on its support, triangle inequality over terms). `exact=True`
takes the full spectral norm instead (tighter, small n only).

Trotter number r (t = n, ε = 1e‑3, seed 1) — same step, so gate counts scale
with r:

| n | order | simcount `com` | `tte` | improvement | brute-force minimum |
|---|---|---|---|---|---|
| 4 | 2 | 5,503 | 1,817 | 3.0× | 699 |
| 4 | 4 | 1,349 | 272 | 5.0× | 45 |
| 10 | 2 | 62,998 | 10,880 | 5.8× | 3,174 |
| 10 | 4 | 12,485 | 1,077 | 11.6× | 152 |
| 20 | 2 | 376,960 | 42,597 | 8.9× | — |
| 20 | 4 | 63,860 | 2,987 | 21.4× | — |

First order gains only ~1.3× (simcount's `com1` already captures the leading
commutator and the tail term is small there). At n=10 the S₄ bound is 7.1×
above the brute-force minimum, matching the 7.2× the paper reports for the
X‑Y‑Z ordering. `tests/test_tte.py` checks that the bound dominates the
measured step error at every step size and that the full evolution meets ε.
simcount's `emp` fit is not a guarantee: at n=6, PF2 it gives 1,152 while
the true minimum for that instance is 1,189.

Caveats: J.1/J.2 are heuristic (no tightness proof; validated numerically),
tabulated only for two or three groups, and the fourth-order evaluation
costs ~10–25 s at n=10–20 (support of the nested commutators is 10 qubits).
