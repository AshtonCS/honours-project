"""Conversion between ``plsimcount`` PennyLane tapes and ``nam`` circuits.

The optimiser works on its own flat gate list (see ``gates.py``); the rest of
``plsimcount`` works on PennyLane ``QuantumScript`` tapes. These two functions
are the only place the representations meet.

Only the Clifford+Rz gate set produced by ``exponentiate.simcount_template``
is supported: H, S, S-adjoint, CNOT, X, Z, RZ. A tape containing anything else
raises rather than being silently dropped.
"""

from __future__ import annotations

import pennylane as qml

from .gates import Circuit, Gate

_FROM_PL = {
    "Hadamard": "H",
    "S": "S",
    "Adjoint(S)": "Sdg",
    "PauliX": "X",
    "PauliZ": "Z",
    "CNOT": "CNOT",
    "RZ": "Rz",
}

_TO_PL = {
    "H": lambda w, p: qml.Hadamard(w[0]),
    "S": lambda w, p: qml.S(w[0]),
    "Sdg": lambda w, p: qml.adjoint(qml.S(w[0])),
    "X": lambda w, p: qml.PauliX(w[0]),
    "Z": lambda w, p: qml.PauliZ(w[0]),
    "CNOT": lambda w, p: qml.CNOT(list(w)),
    "Rz": lambda w, p: qml.RZ(p, w[0]),
}


def from_tape(tape: qml.tape.QuantumScript, n_qubits: int) -> Circuit:
    """Convert a PennyLane tape to a :class:`~plsimcount.nam.gates.Circuit`.

    ``GlobalPhase`` operations are dropped: the optimiser does not track global
    phase, and two of the Hadamard-reduction identities only hold up to one.
    """
    gates: list[Gate] = []
    for op in tape.operations:
        if op.name == "GlobalPhase":
            continue
        if op.name not in _FROM_PL:
            raise ValueError(
                f"{op.name} is outside the optimiser's gate set "
                "{H, S, Adjoint(S), PauliX, PauliZ, CNOT, RZ}"
            )
        name = _FROM_PL[op.name]
        param = float(op.data[0]) if name == "Rz" else None
        gates.append(Gate(name, tuple(int(w) for w in op.wires), param))
    return Circuit(gates, n_qubits)


def to_tape(circuit: Circuit) -> qml.tape.QuantumScript:
    """Convert back to a PennyLane tape, for counting or further transforms."""
    return qml.tape.QuantumScript(
        [_TO_PL[g.name](g.wires, g.param) for g in circuit.gates]
    )
