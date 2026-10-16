"""Pauli strings acting on L qubits, applied without building dense matrices.

Basis convention (upstream ``embed_op2``): site 0 is the leftmost tensor
factor, i.e. bit (L - 1 - s) of the basis index belongs to site s.

A Pauli string P maps basis state |x> to phase(x) |x XOR flip>.  So for any
matrix M,  (P M)[y, :] = P[y, y ^ flip] * M[y ^ flip, :],  and

    exp(-i theta P) M = cos(theta) M - i sin(theta) (P M)            (P^2 = I)

costs O(N^2) instead of the O(N^3) of a dense product.  Single-qubit factors:
    X: flips the bit, P[y, x] = 1.
    Y: flips the bit, P[y, x] = +i if y_bit = 1 else -i   (Y = [[0, -i], [i, 0]]).
    Z: no flip,       P[y, y] = +1 if y_bit = 0 else -1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PauliString:
    """A tensor product of single-qubit Paulis; ``ops`` maps site -> 'X'|'Y'|'Z'."""

    L: int
    ops: tuple[tuple[int, str], ...]
    _flip: int = field(init=False, repr=False, compare=False)
    _phase: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sites = [s for s, _ in self.ops]
        if len(set(sites)) != len(sites):
            raise ValueError(f"repeated site in {self.ops}")
        N = 2 ** self.L
        y = np.arange(N)
        flip = 0
        phase = np.ones(N, dtype=complex)
        for s, p in self.ops:
            if not 0 <= s < self.L:
                raise ValueError(f"site {s} outside 0..{self.L - 1}")
            bit = 1 << (self.L - 1 - s)
            y_bit = (y & bit) != 0
            if p == "X":
                flip |= bit
            elif p == "Y":
                flip |= bit
                phase *= np.where(y_bit, 1j, -1j)
            elif p == "Z":
                phase *= np.where(y_bit, -1.0, 1.0)
            else:
                raise ValueError(f"unknown Pauli {p!r}")
        object.__setattr__(self, "_flip", flip)
        object.__setattr__(self, "_phase", phase)

    @staticmethod
    def one(L: int, site: int, p: str) -> "PauliString":
        return PauliString(L, ((site, p),))

    @staticmethod
    def two(L: int, s0: int, s1: int, p: str) -> "PauliString":
        """p on s0 and on s1 (a bond term such as X_i X_{i+1})."""
        return PauliString(L, ((s0, p), (s1, p)))

    def apply(self, M: np.ndarray) -> np.ndarray:
        """P M, for M of shape (2^L, ...)."""
        perm = np.arange(2 ** self.L) ^ self._flip
        return self._phase.reshape(-1, *([1] * (M.ndim - 1))) * M[perm]

    def exp_apply(self, theta: float, M: np.ndarray) -> np.ndarray:
        """exp(-i theta P) M = cos(theta) M - i sin(theta) P M."""
        return np.cos(theta) * M - 1j * np.sin(theta) * self.apply(M)

    def dense(self) -> np.ndarray:
        """P as a dense 2^L x 2^L matrix (for building H and for tests)."""
        return self.apply(np.eye(2 ** self.L, dtype=complex))
