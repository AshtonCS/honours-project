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


def _mp_mats() -> dict[str, mpmath.matrix]:
    """The gate matrices of ``_MATS`` at the current mpmath precision."""
    r = 1 / mpmath.sqrt(2)
    w = mpmath.expjpi(mpmath.mpf(1) / 4)  # e^{i pi/4}
    return {
        "H": mpmath.matrix([[r, r], [r, -r]]),
        "S": mpmath.matrix([[1, 0], [0, 1j]]),
        "T": mpmath.matrix([[1, 0], [0, w]]),
        "X": mpmath.matrix([[0, 1], [1, 0]]),
        "W": mpmath.matrix([[w, 0], [0, w]]),
        "I": mpmath.eye(2),
    }


def word_error(word: str, theta: float, up_to_phase: bool, eps: float) -> float:
    """||U(word) - RZ(theta)|| in the operator norm, evaluated in mpmath.

    Double precision cannot certify a word below eps ~ 1e-14: rounding over
    the ~150 matrix products of a long word is itself ~1e-14. The bound
    rows (``ana``/``min``/``com``) need eps_rot down to ~1e-17, so the check
    runs with 20 more significant digits than eps has. ``theta`` is the
    target exactly as given (a double); it is not rounded further.

    For a 2x2 M the operator norm is sqrt(lambda_max(M^dag M)), and for the
    Hermitian 2x2 A = M^dag M, lambda_max = (tr A + sqrt(tr^2 - 4 det A)) / 2.
    """
    dps = max(30, int(-mpmath.log10(eps)) + 20)
    with mpmath.workdps(dps):
        mats = _mp_mats()
        U = mpmath.eye(2)
        for ch in word:
            U = U * mats[ch]
        half = mpmath.mpf(theta) / 2
        target = mpmath.matrix([[mpmath.expj(-half), 0], [0, mpmath.expj(half)]])
        if up_to_phase:
            tr = (U.H * target)[0, 0] + (U.H * target)[1, 1]
            U = U * mpmath.expj(mpmath.arg(tr))
        M = U - target
        A = M.H * M
        tr = mpmath.re(A[0, 0] + A[1, 1])
        det = mpmath.re(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
        lam = (tr + mpmath.sqrt(max(tr * tr - 4 * det, 0))) / 2
        return float(mpmath.sqrt(max(lam, 0)))


@lru_cache(maxsize=None)
def synthesise(theta: float, eps: float, up_to_phase: bool = False, backend: str = "pygridsynth") -> str:
    """Clifford+T word for RZ(theta) in matrix order (rightmost applied
    first), as printed by gridsynth. Cached: a product-formula step has only
    a handful of distinct angles. Every word is checked against eps in
    high precision (``word_error``); a miss raises."""
    word = BACKENDS[backend](theta, eps, up_to_phase)
    err = word_error(word, theta, up_to_phase, eps)
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
