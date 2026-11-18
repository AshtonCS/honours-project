# honours-project
Code and documentation used in the production of my honours thesis, "Fast Trotterisation Techniques for Simulating Quantum Systems".

## Abstract

Trotterisation is a standard technique in the field of quantum simulation of physical systems, with theoretical reductions in its cost helping to bridge the gap between the resource requirements of meaningful applications and the technical capabilities of modern-day quantum devices. This project investigates optimisation strategies for Trotterised Hamiltonian simulation by implementing established and recent strategies in product formulae, error bounds and term grouping within a single benchmark environment and evaluating estimated resource requirements at the gate level. We draw conclusions on the effect of each lever within the circuit compilation process and propose future research into the benchmarking and comparison of Trotterisation techniques.

## Branches

- `main`: the current project. `compiler/pysimcount/` (the Python reimplementation of simcount and the experiments built on it), `compiler/effts/`, the recorded results under `compiler/results/`, and the original simulators (`compiler/pennylane_simulator.py`, `compiler/pauli_simulator.py`).
- `plsimcount-baseline`: the development branch the work above was done on (merged into `main` on 2026-11-18); named after `pysimcount`'s earlier name, `plsimcount`.
- `accuracy_and_bounds`: **archived, kept for history only.** An August 2026 experiment adding error bounds to the original simulators; it was never merged and is superseded by the bounds in `compiler/pysimcount/`.
