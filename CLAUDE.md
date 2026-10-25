# CLAUDE.md

## Context

This is an undergraduate honours research project on quantum circuit compilation
for Hamiltonian simulation. The implementation is a Python reimplementation of
the `njross/simcount` framework, used as a validated baseline for experiments on
recent Trotter and product-formula approaches.

Assume I understand: undergraduate quantum mechanics, linear algebra, the
circuit model, Pauli operators, and basic complexity notation.
Do not assume I already know: the internals of simcount, specific results from
recent product-formula literature, or Python library APIs beyond NumPy basics.

<!-- Update the "assume I understand" list as the project goes on. It is the main
     dial controlling how much Claude explains. -->

## How to explain

- Explain the reasoning before the code, not after. A short paragraph on the
  approach, then the implementation.
- When code encodes a physics or mathematics decision, say which one and why.
  Example: "this uses the commutator bound from Childs et al. rather than the
  norm bound, which gives a smaller step count for local Hamiltonians."
- Name the source when a formula comes from the literature. Give the paper and
  the equation or section, so I can check it. If you are unsure of the
  reference, say so rather than guessing at a citation.
- Prefer showing a small worked example with concrete numbers over an abstract
  description.
- When there is a standard approach and you deviate from it, flag the deviation
  explicitly.
- If I ask for something that rests on a misunderstanding, address the
  misunderstanding first. Do not quietly build what I asked for.
- Do not flatter the question or the code. Say plainly when an approach is
  wrong, fragile, or more complicated than it needs to be.

## Never assume

This is the most important rule in this file.

Before writing code that depends on an unstated choice, stop and ask. Do not
pick a plausible default and proceed. The unstated choices that matter most here:

- Hamiltonian conventions: sign, factors of 1/2 in Pauli coefficients, whether
  hbar = 1, open vs periodic boundary conditions, coupling parameter ranges.
- Term ordering in the product formula, and whether ordering is fixed,
  randomised, or grouped.
- Which error bound sets the Trotter step count, and whether the bound is
  analytic or empirical.
- Target precision, simulation time, and system sizes for any experiment.
- Gate set and cost metric: CNOT count, T-count, rotation count, depth, and the
  rotation-synthesis accuracy assumed.
- Random seeds and how many repeats, for any randomised method.

When you must proceed without an answer, state the assumption in a comment at
the point it is made, using the form `# ASSUMPTION: ... (unconfirmed)`, and list
the assumptions again at the end of your reply. Never bury an assumption in
code alone.

If a request is ambiguous, ask one specific question rather than several vague
ones, and say what you would do by default if I do not answer.

Do not invent: numerical constants, equation numbers, citations, API signatures,
or the behaviour of `simcount` that you have not read in the source. If you have
not checked the reference implementation for a behaviour you are matching, say
that you have not.

## Code style

- Python, NumPy for numerics. Ask before introducing any new dependency.
- Type hints on all function signatures.
- Docstrings state units and conventions, not just types. A coefficient argument
  should say what normalisation it expects.
- Variable names follow the mathematics where a standard symbol exists
  (`n_qubits`, `t`, `epsilon`, `r` for Trotter steps), with a comment linking the
  name to the symbol in the thesis.
- Prefer explicit loops over dense vectorised one-liners where the loop makes
  the algorithm's structure visible. This is research code that has to be read
  by an examiner.
- No silent fallbacks. If an input is outside the validated range, raise.

## Correctness and validation

- Every function implementing a formula from a paper gets a test comparing it
  against a small case that can be checked by hand or by exact diagonalisation.
- Any result that will appear in the thesis must be reproducible from a seed and
  a config, not from an interactive session.
- When output disagrees with a published number from simcount or elsewhere,
  report the disagreement. Do not adjust parameters until it matches.

## Academic integrity

This project has a declaration of AI usage. Keep it honest and easy to write:

- When you generate a non-trivial block of code, note in your reply what it does
  and what I should verify. I need to be able to describe it in my own words.
- Do not write prose for the thesis unless I explicitly ask. Comments,
  docstrings, and explanations to me are fine.
- If I ask you to do something I should be doing myself, say so.

## Commands

### Environment

Everything below needs the `pennylane` conda environment. The anaconda *base*
environment has an incompatible NumPy/SciPy pair and cannot import PennyLane.

```
conda activate pennylane          # or use /opt/anaconda3/envs/pennylane/bin/python
```

Non-stdlib dependencies: `pennylane` 0.45.1, `numpy`, `scipy`, `pygridsynth`
(Clifford+T rotation synthesis), `pytest`.

Docker is needed only for two things: a `compare` run with no cached fixture,
and `--synth quipper`. Both use the container `simcount-new`, which holds the
compiled reference implementation at `/tmp/simcount-main` and Ross's
`gridsynth` binary at `/root/.cabal/bin/gridsynth`.

### Run tests

```
cd compiler && python -m pytest pysimcount/tests -q
```

149 tests, ~30 s. No Docker needed: the 86 parity tests replay the 82 cached
simcount outputs in `compiler/pysimcount/fixtures/simcount/`. One test is
skipped by design (Clifford+T parity needs `--synth quipper`).

Run from `compiler/`, not the repo root — `pysimcount` is only importable with
`compiler/` on `sys.path`, and pytest from the root fails at collection.

Bare `pytest` from `compiler/` collects the same 149 tests. It does *not* pick
up the legacy `compiler/test.py` (the filename matches neither `test_*.py` nor
`*_test.py`), which is the suite for `pauli_simulator.py`. Run explicitly with
`python -m pytest test.py`; as of 2026-09-23 it reports 11 failed, 8 passed, in
committed code — `pauli_simulator.py` emits `RZ` angles as `-angle * 2` while
`test.py` asserts positive angles. A sign-convention disagreement, pre-existing
and unrelated to `pysimcount`. Deliberately left as-is and not maintained; do
not modify `pauli_simulator.py` or `test.py` to make it pass.

### Run an experiment

All three subcommands print to stdout; nothing is written to disk except
simcount fixtures.

```
cd compiler

# PennyLane vs simcount, side by side (the validation workhorse)
python -m pysimcount compare -n 4 -a pf2com -g cz
python -m pysimcount compare -n 4 -a pf2com -g ct --synth quipper   # exact Clifford+T
python -m pysimcount compare -n 4 -a pf2com -g cz --optimise 2      # + optimisation stage

# PennyLane only; needs the Hamiltonian supplied, or falls back to a fixture
python -m pysimcount run -n 6 -a pf4tte -g ct --fields 0.1 -0.3 0.5 0.2 -0.7 0.4

# Trotter step count r across all five bounds for one instance
python -m pysimcount bounds -n 10 --empirical
```

Key flags: `-n` size, `-a pf<order><bound>` with bound in
`ana|min|com|emp|tte`, `-g cz|cz2ct|ct` (which sets epsilon and its split),
`-m` seed, `--template`, `--synth`, `--optimise K`. `--help` on any subcommand.

Reproducing a result (the thesis-results rule above). Record the full parameter
set, including the defaults — several of them change the numbers silently:

- `compare`: `(n, algo, gatebase, seed, template, synth)`, plus `optimise K` and
  `up_to_phase` if used. `t = n` and epsilon are derived, not free: `cz` is
  eps=1e-3 all to Trotter, `cz2ct` 5e-4, `ct` 5e-4 Trotter + 5e-4 synthesis.
  The committed fixtures under `pysimcount/fixtures/simcount/` are part of the
  input — they supply the Hamiltonian.
- `run`: the Hamiltonian source (`--fields` or `--hamiltonian`, recorded
  verbatim), `(algo, gatebase, template, synth)`, plus `-t` / `-r` / `-z` where
  they override the derived values.

`-m <seed>` does not generate a Hamiltonian in Python — Haskell's `StdGen`
cannot be reproduced here, so the seed names which recorded simcount instance
to read. `--template` and `--synth` both change gate counts (`pygridsynth`
gives ~4% more T gates than `quipper`), so neither can be left unstated.

The original standalone scripts still run and are unchanged:
`python pennylane_simulator.py` (edit the config block at the top) and
`python pauli_simulator.py`.

### Lint/format

- `TODO` — nothing is configured. No `pyproject.toml`, `setup.cfg`, `ruff.toml`,
  or pre-commit config exists; `.gitignore` mentions ruff and mypy only because
  it is the stock GitHub Python template. Pick a formatter and linter, or record
  the decision not to.

### Other TODOs

- `TODO` — no mechanism for recording an experiment. Results go to stdout only;
  there is no config file format and no results directory. Decide how thesis
  numbers get captured (a config + committed output, or a small runner script).
