"""Gate templates for exp(-i * time * coeff * P) with P a Pauli word.

Templates emit PennyLane operations directly in the Clifford + RZ gate set
{H, S, S†, CNOT, RZ}. Cliffords are kept as Cliffords on purpose: it keeps
the RZ count equal to the number of genuine non-Clifford rotations, which
is what the synthesis error budget is divided over.
"""

from __future__ import annotations

from collections.abc import Callable

import pennylane as qml

from .hamiltonian import Term

Template = Callable[[Term, float], list]


def simcount_template(term: Term, time: float) -> list:
    """``exponentiate_Pauli`` from simcount's Definitions.hs.

    Basis change (H for X; S† H S† for Y), CNOT ladder onto the first
    non-identity qubit, exp(-i Z t') = RZ(2 t') there, then uncompute in
    reverse. ``t' = time * coeff``.
    """
    tp = time * term.coeff
    xs = [q for q, p in enumerate(term.word) if p == "X"]
    ys = [q for q, p in enumerate(term.word) if p == "Y"]
    zs = term.support
    if not zs:
        return [qml.GlobalPhase(tp)]

    pairs = list(zip(zs, zs[1:]))[::-1]  # reverse $ consecutive_pairs zs

    compute = []
    compute += [qml.Hadamard(q) for q in xs]
    for q in ys:
        compute += [qml.adjoint(qml.S(q)), qml.Hadamard(q), qml.adjoint(qml.S(q))]
    compute += [qml.CNOT([q2, q1]) for q1, q2 in pairs]

    uncompute = [qml.CNOT([q2, q1]) for q1, q2 in reversed(pairs)]
    for q in reversed(ys):
        uncompute += [qml.S(q), qml.Hadamard(q), qml.S(q)]
    uncompute += [qml.Hadamard(q) for q in reversed(xs)]

    return compute + [qml.RZ(2 * tp, zs[0])] + uncompute


def pennylane_template(term: Term, time: float) -> list:
    """PennyLane's own ``PauliRot`` decomposition, expanded to the same
    gate set for comparison. Structurally different from simcount's."""
    tp = time * term.coeff
    zs = term.support
    if not zs:
        return [qml.GlobalPhase(tp)]
    word = "".join(term.word[q] for q in zs)
    op = qml.PauliRot(2 * tp, word, wires=zs)
    tape = qml.tape.QuantumScript([op])
    [tape], _ = qml.transforms.decompose(
        tape, gate_set={"Hadamard", "S", "Adjoint(S)", "CNOT", "RZ", "RX", "RY", "GlobalPhase"}
    )
    return list(tape.operations)


TEMPLATES: dict[str, Template] = {
    "simcount": simcount_template,
    "pennylane": pennylane_template,
}
