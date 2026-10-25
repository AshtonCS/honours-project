"""Opt-in circuit optimisation, kept separate from the baseline counts.

simcount reports *unoptimised* counts; the optimisation in Childs et al. was
a separate tool applied to the exported circuits. Here optimisation is an
explicit second stage so both numbers are always available.

``optimise(sim, k)`` optimises a block of ``k`` consecutive steps with
PennyLane's peephole passes and reports ``block x (r // k)`` plus the
optimised remainder. k = 1 is intra-step only; k = 2 captures the
cancellations at step boundaries; k = r is the fully expanded circuit.
"""

from __future__ import annotations

import pennylane as qml
from pennylane.transforms import cancel_inverses, commute_controlled, merge_rotations

from . import resources
from .circuit import Simulation

DEFAULT_PASSES = (commute_controlled, cancel_inverses, merge_rotations)


def optimise_tape(tape: qml.tape.QuantumScript, passes=DEFAULT_PASSES, num_passes: int = 2):
    for _ in range(num_passes):
        for p in passes:
            [tape], _ = p(tape)
    return tape


def repeat(tape: qml.tape.QuantumScript, k: int) -> qml.tape.QuantumScript:
    return qml.tape.QuantumScript(list(tape.operations) * k)


def optimise(sim: Simulation, k: int = 1, passes=DEFAULT_PASSES, num_passes: int = 2) -> dict[str, int]:
    """Aggregated counts after optimising blocks of ``k`` steps."""
    k = max(1, min(k, sim.r))
    blocks, rem = divmod(sim.r, k)
    block = optimise_tape(repeat(sim.step, k), passes, num_passes)
    counts = resources.scale(resources.count(block), blocks)
    if rem:
        tail = optimise_tape(repeat(sim.step, rem), passes, num_passes)
        for lbl, c in resources.count(tail).items():
            counts[lbl] = counts.get(lbl, 0) + c
    return counts
