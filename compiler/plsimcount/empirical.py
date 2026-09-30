"""Empirical Trotter numbers by brute-force simulation (small systems).

Computes ||S(t/r)^r - exp(-iHt)|| exactly with dense matrices and searches
for the smallest r meeting eps, as in Section 5.2 of Childs et al. (2021)
and the 'empirical' data of Childs et al. (2018). Feasible up to n ~ 10-12.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from . import formula
from .hamiltonian import Hamiltonian

_P = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}


def _word_matrix(word: str) -> np.ndarray:
    m = np.array([[1.0]], dtype=complex)
    for p in word:
        m = np.kron(m, _P[p])
    return m


class Evolution:
    """Exact unitaries for one Hamiltonian, with the term matrices cached."""

    def __init__(self, H: Hamiltonian):
        self.H = H
        self.mats = [_word_matrix(t.word) for t in H.terms]
        self.Hmat = H.matrix()

    def exact(self, t: float) -> np.ndarray:
        return expm(-1j * t * self.Hmat)

    def step(self, order: int, d: float) -> np.ndarray:
        U = np.eye(2**self.H.n_qubits, dtype=complex)
        for j, time in formula.schedule(order, self.H, d):
            U = expm(-1j * time * self.H.terms[j].coeff * self.mats[j]) @ U
        return U

    def step_error(self, order: int, d: float) -> float:
        return float(np.linalg.norm(self.step(order, d) - self.exact(d), 2))

    def error(self, order: int, t: float, r: int) -> float:
        S = np.linalg.matrix_power(self.step(order, t / r), r)
        return float(np.linalg.norm(S - self.exact(t), 2))


def steps(order: int, H: Hamiltonian, t: float, eps: float) -> int:
    """Smallest r with measured error <= eps (exponential + binary search)."""
    ev = Evolution(H)
    lo, hi = 0, 1
    while ev.error(order, t, hi) > eps:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if ev.error(order, t, mid) <= eps:
            hi = mid
        else:
            lo = mid
    return hi
