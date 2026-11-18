# honours-project
Code and documentation used in the production of my honours thesis, "Fast Trotterization Techniques for Simulating Quantum Systems".

Presently, this repository contains a work-in-progress quantum compiler for applications structured around Heisenberg model hamiltonians. 

## Branches

- `main`: the current project. `compiler/pysimcount/` (the Python reimplementation of simcount and the experiments built on it), `compiler/effts/`, the recorded results under `compiler/results/`, and the original simulators (`compiler/pennylane_simulator.py`, `compiler/pauli_simulator.py`).
- `plsimcount-baseline`: the development branch the work above was done on (merged into `main` on 2026-11-18); named after `pysimcount`'s earlier name, `plsimcount`.
- `accuracy_and_bounds`: **archived, kept for history only.** An August 2026 experiment adding error bounds to the original simulators; it was never merged and is superseded by the bounds in `compiler/pysimcount/`.
