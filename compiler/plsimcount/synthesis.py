"""RZ -> Clifford+T synthesis via the Ross-Selinger algorithm (pygridsynth).

Follows Quipper's ``Approximate`` gate base as used by simcount's ``-g ct``:
each exp(-i Z t) becomes RZ(2t) synthesised to precision

    eps_rot = eps_synth / zrots

where ``zrots`` is the number of rotations in the whole circuit. Every
synthesised sequence is checked numerically against the target rotation and
rejected if it misses ``eps_rot`` -- PennyLane's built-in
``rs_decomposition`` silently returns wrong sequences below eps ~ 1e-8,
which is exactly the regime this project operates in.
"""

from __future__ import annotations

from functools import lru_cache

import mpmath
import numpy as np
import pennylane as qml

from .circuit import Simulation

_MATS = {
    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "S": np.diag([1, 1j]),
    "T": np.diag([1, np.exp(1j * np.pi / 4)]),
    "X": np.array([[0, 1], [1, 0]]),
    "W": np.exp(1j * np.pi / 4) * np.eye(2),
    "I": np.eye(2),
}


def _gate(ch: str, wire):
    return {
        "H": lambda: qml.Hadamard(wire),
        "S": lambda: qml.S(wire),
        "T": lambda: qml.T(wire),
        "X": lambda: qml.PauliX(wire),
        "W": lambda: qml.GlobalPhase(-np.pi / 4),  # omega = e^{i pi/4}
    }[ch]()


def rz_matrix(theta: float) -> np.ndarray:
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])


def distance(U: np.ndarray, V: np.ndarray) -> float:
    """Operator-norm distance ||U - V||."""
    return float(np.linalg.norm(U - V, 2))


def _pygridsynth(theta: float, eps: float, up_to_phase: bool) -> str:
    from pygridsynth import gridsynth_gates

    return gridsynth_gates(mpmath.mpf(theta), mpmath.mpf(eps), up_to_phase=up_to_phase)


def _quipper(theta: float, eps: float, up_to_phase: bool) -> str:
    """Ross's reference ``gridsynth`` binary, run inside the simcount
    container. Precision is passed in bits exactly as Quipper's
    ``Approximate`` gate base computes it (``-log10(eps) * digits``)."""
    import math
    import subprocess

    from .simcount import CONTAINER

    bits = (-math.log10(eps)) * math.log2(10)
    cmd = f"/root/.cabal/bin/gridsynth -b {bits!r} -r 1 {'-p ' if up_to_phase else ''}-- '{theta!r}'"
    proc = subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc", cmd],
                          capture_output=True, text=True, check=True)
    return proc.stdout.strip().splitlines()[-1]


BACKENDS = {"pygridsynth": _pygridsynth, "quipper": _quipper}


@lru_cache(maxsize=None)
def synthesise(theta: float, eps: float, up_to_phase: bool = False, backend: str = "pygridsynth") -> str:
    """Clifford+T word for RZ(theta) in matrix order (rightmost applied
    first), as printed by gridsynth. Cached: a product-formula step has only
    a handful of distinct angles."""
    word = BACKENDS[backend](theta, eps, up_to_phase)
    U = np.eye(2, dtype=complex)
    for ch in word:
        U = U @ _MATS[ch]
    if up_to_phase:
        tr = np.trace(U.conj().T @ rz_matrix(theta))
        U = U * np.exp(1j * np.angle(tr))
    err = distance(U, rz_matrix(theta))
    if err > eps:
        raise RuntimeError(f"gridsynth output for RZ({theta}) has error {err:.3e} > {eps:.3e}")
    return word


def word_to_ops(word: str, wire) -> list:
    """Circuit-order PennyLane ops for a gridsynth word."""
    return [_gate(ch, wire) for ch in reversed(word) if ch != "I"]


def synthesise_tape(tape: qml.tape.QuantumScript, eps_rot: float, up_to_phase: bool = False,
                    backend: str = "pygridsynth"):
    ops = []
    for op in tape.operations:
        if op.name == "RZ":
            word = synthesise(float(op.data[0]), eps_rot, up_to_phase, backend)
            ops += word_to_ops(word, op.wires[0])
        else:
            ops.append(op)
    return qml.tape.QuantumScript(ops)


def synthesise_step(sim: Simulation, eps_synth: float, zrots: int | None = None,
                    up_to_phase: bool = False, backend: str = "pygridsynth"):
    """Return ``(clifford_t_step, eps_rot)`` for a simulation, dividing
    ``eps_synth`` evenly over all ``zrots`` rotations of the full circuit."""
    zrots = sim.zrots if zrots is None else zrots
    eps_rot = eps_synth / zrots
    return synthesise_tape(sim.step, eps_rot, up_to_phase, backend), eps_rot
