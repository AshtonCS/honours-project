"""Gate and circuit representation for the Nam et al. optimiser.

Deliberately separate from ``pysimcount.circuit`` (which holds PennyLane
tapes): the optimiser needs cheap structural edits and per-wire adjacency,
which a flat list of small dataclasses gives more directly than a tape.

Conventions
-----------
* Circuit order: ``gates[0]`` is applied **first**. The unitary is therefore
  ``U = M(gates[-1]) @ ... @ M(gates[0])``.
* Wire 0 is the **most significant** tensor factor, matching
  ``pysimcount.hamiltonian.Hamiltonian.matrix``.
* ``Rz(theta)`` is ``diag(exp(-i theta/2), exp(+i theta/2))`` -- PennyLane's
  convention. Quipper's ``exp(-i%Z)`` parameter ``t`` is ``theta/2``; the
  conversion lives in ``quipper.py``, not here.
* Global phase is **not** tracked. Two of the Hadamard-reduction identities
  (Figure 4 of the paper) hold only up to a phase of ``exp(i pi/4)``, and the
  reference circuits in ``simcount/samples`` carry no phase gates in ``cz``
  mode, so equality is always checked up to global phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

_SQRT1_2 = 1.0 / np.sqrt(2.0)

# One-qubit gate matrices, keyed by gate name.
ONE_QUBIT: dict[str, np.ndarray] = {
    "H": np.array([[_SQRT1_2, _SQRT1_2], [_SQRT1_2, -_SQRT1_2]], dtype=complex),
    "S": np.diag([1.0, 1.0j]).astype(complex),
    "Sdg": np.diag([1.0, -1.0j]).astype(complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}

SELF_INVERSE: frozenset[str] = frozenset({"H", "X", "Z", "CNOT"})

# name -> name of the inverse gate, for the non-self-inverse fixed gates.
INVERSE_OF: dict[str, str] = {"S": "Sdg", "Sdg": "S"}


@dataclass(frozen=True, slots=True)
class Gate:
    """A single gate.

    Args:
        name: one of ``H``, ``S``, ``Sdg``, ``X``, ``Z``, ``CNOT``, ``Rz``.
        wires: qubit indices. For ``CNOT`` this is ``(control, target)``.
        param: rotation angle in radians for ``Rz``, else ``None``.
    """

    name: str
    wires: tuple[int, ...]
    param: float | None = None

    def __post_init__(self) -> None:
        if self.name == "Rz":
            if self.param is None:
                raise ValueError("Rz requires an angle")
            if len(self.wires) != 1:
                raise ValueError(f"Rz acts on one wire, got {self.wires}")
        elif self.name == "CNOT":
            if self.param is not None:
                raise ValueError("CNOT takes no parameter")
            if len(self.wires) != 2 or self.wires[0] == self.wires[1]:
                raise ValueError(f"CNOT needs two distinct wires, got {self.wires}")
        elif self.name in ONE_QUBIT:
            if self.param is not None:
                raise ValueError(f"{self.name} takes no parameter")
            if len(self.wires) != 1:
                raise ValueError(f"{self.name} acts on one wire, got {self.wires}")
        else:
            raise ValueError(f"unknown gate {self.name!r}")

    @property
    def is_rotation(self) -> bool:
        return self.name == "Rz"

    def matrix(self) -> np.ndarray:
        """Matrix on this gate's own wires, in ``self.wires`` order."""
        if self.name == "Rz":
            half = self.param / 2.0
            return np.diag([np.exp(-1j * half), np.exp(1j * half)])
        if self.name == "CNOT":
            return np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
            )
        return ONE_QUBIT[self.name]

    def inverse(self) -> Gate:
        if self.name == "Rz":
            return Gate("Rz", self.wires, -self.param)
        if self.name in SELF_INVERSE:
            return self
        return Gate(INVERSE_OF[self.name], self.wires)

    def cancels_with(self, other: Gate, rtol: float = 1e-12) -> bool:
        """True if applying ``self`` then ``other`` is the identity.

        Rz pairs cancel only when the angles sum to zero *relative to their own
        magnitude*. The tolerance must be relative: with an absolute one, two
        small same-sign rotations (which do not cancel) test as cancelling,
        and once they are deleted the CNOTs either side become adjacent and
        cascade away -- on the pf4ana samples this collapsed the circuit to
        zero CNOT gates. Rotation angles here scale as t/r and reach 1e-7 at
        n=32, with merged angles drifting smaller still.

        Use :func:`merged_rotation` for the general (non-cancelling) merge.
        """
        if self.wires != other.wires:
            return False
        if self.name != other.name:
            return {self.name, other.name} == {"S", "Sdg"}
        if self.name == "Rz":
            scale = abs(self.param) + abs(other.param)
            return abs(self.param + other.param) <= rtol * scale
        return self.name in SELF_INVERSE


def merged_rotation(a: Gate, b: Gate) -> Gate:
    """Combine two Rz gates on the same wire into one (Routine 2's variant)."""
    if not (a.is_rotation and b.is_rotation and a.wires == b.wires):
        raise ValueError("merged_rotation expects two Rz gates on the same wire")
    return Gate("Rz", a.wires, a.param + b.param)


@dataclass
class Circuit:
    """An ordered gate list on ``n_qubits`` wires."""

    gates: list[Gate]
    n_qubits: int

    def __len__(self) -> int:
        return len(self.gates)

    def copy(self) -> Circuit:
        return Circuit(list(self.gates), self.n_qubits)

    def counts(self) -> dict[str, int]:
        """Gate counts keyed by name, with ``Rz`` aggregated over angles."""
        out: dict[str, int] = {}
        for g in self.gates:
            out[g.name] = out.get(g.name, 0) + 1
        return out

    def matrix(self) -> np.ndarray:
        """Full unitary. Only for small ``n_qubits`` -- cost is 2**n_qubits."""
        dim = 2**self.n_qubits
        if self.n_qubits > 14:
            raise ValueError(
                f"refusing to build a {dim}x{dim} matrix for {self.n_qubits} qubits"
            )
        u = np.eye(dim, dtype=complex)
        for g in self.gates:
            u = embed(g, self.n_qubits) @ u
        return u


def embed(gate: Gate, n_qubits: int) -> np.ndarray:
    """Embed a gate's matrix into the full ``2**n_qubits`` space."""
    return _embed_cached(gate.name, gate.wires, gate.param, n_qubits)


@lru_cache(maxsize=4096)
def _embed_cached(
    name: str, wires: tuple[int, ...], param: float | None, n_qubits: int
) -> np.ndarray:
    gate = Gate(name, wires, param)
    if len(wires) == 1:
        (q,) = wires
        factors = [np.eye(2, dtype=complex) for _ in range(n_qubits)]
        factors[q] = gate.matrix()
        out = np.array([[1.0]], dtype=complex)
        for f in factors:
            out = np.kron(out, f)
        return out

    # Two-qubit gate: build by acting on basis states, which avoids
    # constructing permutation matrices for non-adjacent wires.
    control, target = wires
    dim = 2**n_qubits
    out = np.zeros((dim, dim), dtype=complex)
    c_bit = n_qubits - 1 - control  # wire 0 is most significant
    t_bit = n_qubits - 1 - target
    for col in range(dim):
        row = col
        if (col >> c_bit) & 1:
            row = col ^ (1 << t_bit)
        out[row, col] = 1.0
    return out


def local_matrix(gates: list[Gate], wires: tuple[int, ...]) -> np.ndarray:
    """Unitary of ``gates`` restricted to ``wires`` (in that order).

    Every gate must act only within ``wires``. Used by the commutation test,
    where the support is one or two qubits, so this stays cheap.
    """
    index = {w: i for i, w in enumerate(wires)}
    k = len(wires)
    sub = Circuit(
        [Gate(g.name, tuple(index[w] for w in g.wires), g.param) for g in gates], k
    )
    return sub.matrix()
