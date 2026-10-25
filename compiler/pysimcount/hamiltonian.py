"""Hamiltonians as ordered lists of weighted Pauli words.

Mirrors ``Definitions.hs`` in simcount: a Hamiltonian is a list of
``(coefficient, PauliTensor)`` pairs, and *term order matters* because it
fixes the order of exponentials in every product formula.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Term:
    coeff: float
    word: str  # e.g. "XXII"; one character per qubit, from {I, X, Y, Z}

    @property
    def support(self) -> list[int]:
        return [q for q, p in enumerate(self.word) if p != "I"]


@dataclass
class Hamiltonian:
    terms: list[Term]
    # Number of leading terms treated as the non-commuting block by the
    # second-order formula (simcount's ``hamiltonian_split``). The remaining
    # terms are applied once, at the full time step, in the middle.
    split: int | None = None
    name: str = ""
    _n: int = field(init=False, repr=False)

    def __post_init__(self):
        lengths = {len(t.word) for t in self.terms}
        if len(lengths) != 1:
            raise ValueError("all Pauli words must have the same length")
        self._n = lengths.pop()
        if self.split is None:
            self.split = len(self.terms)

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    @property
    def max_coeff(self) -> float:
        return max(abs(t.coeff) for t in self.terms)

    def matrix(self) -> np.ndarray:
        """Dense matrix (for validation on small systems only)."""
        P = {
            "I": np.eye(2),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.diag([1.0, -1.0]).astype(complex),
        }
        H = np.zeros((2**self._n, 2**self._n), dtype=complex)
        for t in self.terms:
            op = np.array([[1.0]], dtype=complex)
            for p in t.word:
                op = np.kron(op, P[p])
            H += t.coeff * op
        return H


def _word(n: int, placements: dict[int, str]) -> str:
    return "".join(placements.get(q, "I") for q in range(n))


def heisenberg(n: int, fields, coupling=(1.0, 1.0, 1.0)) -> Hamiltonian:
    """simcount's ``random_h`` with explicit field values.

    H = sum_{<i,j>} (Jx XX + Jy YY + Jz ZZ)_{ij} + sum_i h_i Z_i

    Term order is Pauli-major (all XX bonds, all YY, all ZZ, then fields) and
    the chain is periodic for n >= 3, open otherwise -- exactly as in
    ``Definitions.hs``. ``fields`` must be the n values simcount drew for its
    seed (see :func:`parse_simcount`), since Haskell's StdGen cannot be
    reproduced from Python.
    """
    fields = list(map(float, fields))
    if len(fields) != n:
        raise ValueError(f"expected {n} field values, got {len(fields)}")
    if n == 1:
        return Hamiltonian([Term(fields[0], "Z")], split=0, name="heisenberg")

    bonds = [(i, i + 1) for i in range(n - 1)]
    if n >= 3:
        bonds.append((0, n - 1))

    terms = []
    for J, p in zip(coupling, "XYZ"):
        terms += [Term(float(J), _word(n, {i: p, j: p})) for i, j in bonds]
    terms += [Term(h, _word(n, {i: "Z"})) for i, h in enumerate(fields)]

    split = 3 * len(bonds)  # 3n for n>=3, 3 for n==2
    return Hamiltonian(terms, split=split, name="heisenberg")


_TERM_RE = re.compile(r"\(\s*(-?[0-9.eE+-]+)\s*,\s*\[([IXYZ,\s]+)\]\s*\)")


def parse_simcount(text: str) -> Hamiltonian:
    """Parse the ``Hamiltonian: [(1.0,[X,X,I,I]),...]`` line simcount prints."""
    if "Hamiltonian:" in text:
        text = text.split("Hamiltonian:", 1)[1].splitlines()[0]
    terms = [
        Term(float(c), "".join(w.replace(" ", "").split(",")))
        for c, w in _TERM_RE.findall(text)
    ]
    if not terms:
        raise ValueError("no Hamiltonian terms found")
    n = len(terms[0].word)
    # simcount's split rule (hamiltonian_split in PF.hs)
    split = {0: 0, 1: 0, 2: 3}.get(n, 3 * n)
    return Hamiltonian(terms, split=min(split, len(terms)), name="simcount")


def from_pennylane(H, split: int | None = None) -> Hamiltonian:
    """Convert a PennyLane operator (e.g. ``qml.spin.heisenberg``) to a
    :class:`Hamiltonian`, preserving PennyLane's term order."""
    import pennylane as qml

    coeffs, ops = H.terms()
    wires = sorted(H.wires.tolist())
    n = max(wires) + 1
    terms = []
    for c, op in zip(coeffs, ops):
        pw = qml.pauli.pauli_sentence(op)
        if len(pw) != 1:
            raise ValueError(f"{op} is not a single Pauli word")
        (word, k), = pw.items()
        terms.append(Term(float(c) * float(k), _word(n, dict(word.items()))))
    return Hamiltonian(terms, split=split, name="pennylane")
