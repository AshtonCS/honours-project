"""Equivalence checks for optimised circuits.

Building a full 2**n x 2**n unitary is only viable to about n=14, but the
reference circuits start at n=13 and go to n=100. Applying a circuit to a
*state vector* costs O(2**n) per gate instead of O(2**3n), which reaches n~25
comfortably, so equivalence is tested by acting on random states:

    if  ||C1|psi> - e^{i a} C2|psi>||  is at machine precision for several
    Haar-random |psi>, the two circuits agree up to global phase.

This is a randomised test, not a proof, but a structural error in a rewrite
rule fails it on the first state. Global phase is factored out because two of
the Hadamard-reduction identities only hold up to a phase.
"""

from __future__ import annotations

import numpy as np

from .gates import Circuit, Gate


def apply_gate(state: np.ndarray, gate: Gate, n_qubits: int) -> np.ndarray:
    """Apply one gate to a state of shape ``(2,) * n_qubits``."""
    if gate.name == "CNOT":
        control, target = gate.wires
        state = np.moveaxis(state, (control, target), (0, 1))
        out = state.copy()
        out[1, 0], out[1, 1] = state[1, 1], state[1, 0]
        return np.moveaxis(out, (0, 1), (control, target))

    (q,) = gate.wires
    state = np.moveaxis(state, q, 0)
    out = np.tensordot(gate.matrix(), state, axes=([1], [0]))
    return np.moveaxis(out, 0, q)


def run(circuit: Circuit, state: np.ndarray) -> np.ndarray:
    """Apply a whole circuit to a state of shape ``(2,) * n_qubits``."""
    for gate in circuit.gates:
        state = apply_gate(state, gate, circuit.n_qubits)
    return state


def random_state(n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    vec /= np.linalg.norm(vec)
    return vec.reshape((2,) * n_qubits)


def equivalent(
    a: Circuit,
    b: Circuit,
    trials: int = 3,
    atol: float = 1e-9,
    seed: int = 0,
) -> tuple[bool, float]:
    """Do ``a`` and ``b`` act identically, up to global phase?

    Returns ``(ok, worst_error)``. Raises if the two circuits disagree on the
    number of qubits.
    """
    if a.n_qubits != b.n_qubits:
        raise ValueError(f"qubit count differs: {a.n_qubits} vs {b.n_qubits}")
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        psi = random_state(a.n_qubits, rng)
        va = run(a, psi).ravel()
        vb = run(b, psi).ravel()
        overlap = np.vdot(vb, va)
        phase = overlap / abs(overlap) if abs(overlap) > 1e-30 else 1.0
        worst = max(worst, float(np.linalg.norm(va - phase * vb)))
    return worst <= atol, worst
