"""Heisenberg-chain Hamiltonians of paper Eq. (24), with upstream's term layout.

    H = sum_{i=0}^{L-1} ( Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1} + h_i Z_i )

Conventions (all read from upstream ``Code/Time_Evolution/src/Model.cpp`` and
``include/Operators.h``, and consistent with the paper):

* Pauli matrices, no factors of 1/2, hbar = 1.
* Periodic by default: bond L-1 couples sites L-1 and 0.  ``periodic=False``
  sets that bond's couplings to 0 but keeps the field h_{L-1} -- exactly what
  upstream's ``bound = true`` does.  (Upstream's naming is inverted relative
  to the word: ``bound = false`` means periodic.)
* Site 0 is the leftmost tensor factor (most significant bit of the basis
  index), as in upstream ``embed_op2``.
* Models: "XXZ" -> Jx = Jy = Jz = J (upstream ``ModelXXZ``; the paper calls
  J^x = J^y = J^z = 1 with random fields the "XXZ model");  "XZ" -> Jy = 0,
  Jx = Jz = J (``ModelXZ``);  "Heisenberg" -> separate (Jx, Jy, Jz).

Term layout.  Each bond i carries up to three *local stages*, keyed by
direction, which the product formula treats as units (paper Eq. 26):

    H_i^x = Jx X_i X_{i+1},  H_i^y = Jy Y_i Y_{i+1},
    H_i^z = Jz Z_i Z_{i+1} + h_i Z_i.

The field rides with the ZZ term of the same bond.  Inside H_i^z the two
Paulis commute and are exponentiated separately, ZZ first, as upstream does.
Upstream skips the Y stages entirely for ``ModelXZ`` (no Y terms are added),
and so does this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .paulis import PauliString


@dataclass(frozen=True)
class LocalStage:
    """One H_i^alpha: direction alpha in 'XYZ', bond index i, and its terms.

    terms: (PauliString, coefficient) pairs, coefficients in units of energy
    (H itself, no 1/2 factors); exponentiated in the listed order.
    """

    direction: str
    bond: int
    terms: tuple[tuple[PauliString, float], ...]


@dataclass(frozen=True)
class Chain:
    """A Heisenberg chain: its parameters and its local stages."""

    L: int
    model: str
    J: tuple[float, float, float]
    fields: tuple[float, ...]
    periodic: bool
    stages: dict[str, tuple[LocalStage, ...]]  # 'X' | 'Y' | 'Z' -> bonds 0..L-1

    def hamiltonian(self) -> np.ndarray:
        """Dense H, 2^L x 2^L, summed in upstream's order (all X, all Y, all Z)."""
        N = 2 ** self.L
        H = np.zeros((N, N), dtype=complex)
        for direction in "XYZ":
            for stage in self.stages.get(direction, ()):
                for p, coef in stage.terms:
                    H += coef * p.dense()
        return H


def chain(L: int, model: str, fields: list[float] | np.ndarray, J: float | tuple[float, float, float] = 1.0,
          periodic: bool = True) -> Chain:
    """Build the chain.  ``fields`` is h_0..h_{L-1}, supplied explicitly.

    Upstream's ``hmag = random`` draws h_i ~ U[-0.1, 0.1] from
    ``std::mt19937(42)``; that stream is implementation-defined in C++, so this
    module never generates fields itself.  See ``reference.py`` for the
    recorded upstream values.
    """
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")
    # L == 2 and periodic: bonds (0,1) and (1,0) are both added, so the single
    # physical bond is counted twice (H has 2J per direction).  Upstream does
    # the same, and paper Fig. 6 starts at L = 2, so this is kept for parity.
    fields = tuple(float(h) for h in fields)
    if len(fields) != L:
        raise ValueError(f"need {L} fields, got {len(fields)}")
    if model == "XXZ":
        if not np.isscalar(J):
            raise ValueError("XXZ takes a single J")
        Jx = Jy = Jz = float(J)
    elif model == "XZ":
        if not np.isscalar(J):
            raise ValueError("XZ takes a single J")
        Jx, Jy, Jz = float(J), 0.0, float(J)
    elif model == "Heisenberg":
        Jx, Jy, Jz = (float(j) for j in J)
    else:
        raise ValueError(f"model must be XXZ, XZ or Heisenberg, got {model!r}")

    stages: dict[str, list[LocalStage]] = {"X": [], "Y": [], "Z": []}
    for i in range(L):
        j = (i + 1) % L
        open_bond = (not periodic) and i == L - 1
        cx, cy, cz = (0.0, 0.0, 0.0) if open_bond else (Jx, Jy, Jz)
        stages["X"].append(LocalStage("X", i, ((PauliString.two(L, i, j, "X"), cx),)))
        stages["Y"].append(LocalStage("Y", i, ((PauliString.two(L, i, j, "Y"), cy),)))
        stages["Z"].append(LocalStage("Z", i, ((PauliString.two(L, i, j, "Z"), cz),
                                                (PauliString.one(L, i, "Z"), fields[i]))))
    if model == "XZ":
        del stages["Y"]  # upstream ModelXZ adds no Y terms at all
    return Chain(L, model, (Jx, Jy, Jz), fields, periodic,
                 {k: tuple(v) for k, v in stages.items()})
