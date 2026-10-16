"""Two-group product formula for the periodic Heisenberg chain -- the "full" option.

Groups:  E' = even bonds with the fields folded in,   O = odd bonds.

Each even bond takes the fields of its two qubits (for even n every qubit is
on exactly one even bond), so E' is a sum of two-qubit blocks on disjoint
pairs and O likewise. Blocks in one group commute (no shared qubit); terms
*inside* an E' block do not (h Z_j clashes with X_j X_k and Y_j Y_k), so
each block is compiled as one exact two-qubit gate.

E' block, for Jx = Jy = J (required -- raises otherwise):

    exp(-i tau [ J (XX+YY) + Jz ZZ + h0 Z0 + h1 Z1 ])

Write h0 Z0 + h1 Z1 = hbar (Z0+Z1) + delta (Z0-Z1). ZZ and Z0+Z1 commute
with everything in the block, so they factor out. What remains,
J(XX+YY) + delta(Z0-Z1), conserves the number of 1s: it is zero on
{|00>, |11>} and on {|01>, |10>} equals 2 (J sx' + delta sz') with
sx' = (XX+YY)/2, sz' = (Z0-Z1)/2 acting as Pauli matrices there. A ZXZ Euler
decomposition of that 2x2 rotation then maps back to gates:

    Rz'(g) = RZ(g/2) x RZ(-g/2)         (local, identity on |00>,|11>)
    Rx'(b) = exp(-i b (XX+YY)/4)        (canonical block, a = b_ = b/4)

so the block is  RZ x RZ  .  canonical(b/4, b/4, tau Jz)  .  RZ x RZ,
with hbar folded into the outer RZs: 3 CNOT and 7 rotations. Exactness is
checked against expm in ``tests/test_twogroup.py``.

O block: ``blocks.canonical_block`` (3 CNOT, 3 rotations).

    python -m plsimcount.twogroup -n 4 6 8 --orders 2 4 --method empirical
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pennylane as qml
from scipy.linalg import expm

from .blocks import canonical_block
from .evenodd import bond_index
from .hamiltonian import Hamiltonian, Term

# ------------------------------------------------------------------ grouping

@dataclass(frozen=True)
class Bond:
    q0: int
    q1: int
    Jx: float
    Jy: float
    Jz: float
    h0: float = 0.0  # field on q0 folded into this bond (E' only)
    h1: float = 0.0

    def terms(self, n: int) -> list[Term]:
        def w(p0: str, p1: str) -> str:
            return "".join(p0 if q == self.q0 else p1 if q == self.q1 else "I" for q in range(n))
        out = [Term(self.Jx, w("X", "X")), Term(self.Jy, w("Y", "Y")), Term(self.Jz, w("Z", "Z"))]
        out += [Term(self.h0, w("Z", "I")), Term(self.h1, w("I", "Z"))] if (self.h0 or self.h1) else []
        return out


def split_groups(H: Hamiltonian) -> tuple[list[Bond], list[Bond]]:
    """(E', O): even bonds carrying their qubits' fields, and odd bonds.

    Requires even n >= 4, a periodic nearest-neighbour chain, and exactly one
    XX, YY, ZZ per bond and one Z per site; anything else raises."""
    n = H.n_qubits
    if n < 4 or n % 2:
        raise ValueError(f"two-group splitting needs even n >= 4 on a periodic chain, got n={n}")
    coeffs: dict[int, dict[str, float]] = {}
    pairs: dict[int, tuple[int, int]] = {}
    fields: dict[int, float] = {}
    for t in H.terms:
        k = bond_index(t, n)
        if k is None:
            [q] = t.support
            if t.word[q] != "Z" or q in fields:
                raise ValueError(f"unexpected single-site term {t.word!r}")
            fields[q] = t.coeff
            continue
        letters = {t.word[q] for q in t.support}
        if len(letters) != 1:
            raise ValueError(f"unexpected two-site term {t.word!r}")
        p = letters.pop()
        if p in coeffs.setdefault(k, {}):
            raise ValueError(f"duplicate {p}{p} term on bond {k}")
        coeffs[k][p] = t.coeff
        pairs[k] = tuple(t.support)
    if sorted(coeffs) != list(range(n)) or any(sorted(c) != ["X", "Y", "Z"] for c in coeffs.values()):
        raise ValueError("expected XX, YY, ZZ on each of the n bonds of a periodic chain")
    if sorted(fields) != list(range(n)):
        raise ValueError("expected one Z field per site")

    even, odd = [], []
    for k in range(n):
        q0, q1 = pairs[k]
        c = coeffs[k]
        if k % 2 == 0:
            even.append(Bond(q0, q1, c["X"], c["Y"], c["Z"], fields[q0], fields[q1]))
        else:
            odd.append(Bond(q0, q1, c["X"], c["Y"], c["Z"]))
    return even, odd


# ------------------------------------------------------------------ blocks

def _zxz(W: np.ndarray) -> tuple[float, float, float]:
    """(a, b, g) with W = Rz(a) Rx(b) Rz(g) exactly, for W in SU(2).
    Rz(t) = exp(-i t Z/2), Rx likewise. Via ZYZ and Ry(b) = Rz(pi/2) Rx(b) Rz(-pi/2)."""
    b = 2 * math.atan2(abs(W[1, 0]), abs(W[0, 0]))
    if abs(W[1, 0]) < 1e-14:          # diagonal: only a + g is defined
        a, g = 2 * float(np.angle(W[1, 1])), 0.0
    elif abs(W[0, 0]) < 1e-14:        # anti-diagonal: only a - g is defined
        a, g = 2 * float(np.angle(W[1, 0])), 0.0
    else:
        a = float(np.angle(W[1, 0]) - np.angle(W[0, 0]))
        g = float(np.angle(W[1, 1]) - np.angle(W[1, 0]))

    def rz(t):
        return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])

    ry = np.array([[math.cos(b / 2), -math.sin(b / 2)], [math.sin(b / 2), math.cos(b / 2)]])
    if not np.allclose(rz(a) @ ry @ rz(g), W, atol=1e-12):
        a += 2 * math.pi              # Rz(a + 2pi) = -Rz(a): fixes the SU(2) sign
    if not np.allclose(rz(a) @ ry @ rz(g), W, atol=1e-12):
        raise ArithmeticError("ZYZ decomposition failed")
    return a + math.pi / 2, b, g - math.pi / 2


def field_block(bond: Bond, tau: float) -> list:
    """exp(-i tau [Jx XX + Jy YY + Jz ZZ + h0 Z_q0 + h1 Z_q1]) up to global
    phase: 3 CNOT, 7 RZ. ``tau`` is the formula's time for this group."""
    if not math.isclose(bond.Jx, bond.Jy, rel_tol=0, abs_tol=1e-15):
        raise ValueError("field_block needs Jx == Jy (magnetisation-conserving bond)")
    J, hbar, delta = bond.Jx, (bond.h0 + bond.h1) / 2, (bond.h0 - bond.h1) / 2
    # 2x2 action on {|01>, |10>}: exp(-i 2 tau (J sx + delta sz))
    W = expm(-2j * tau * np.array([[delta, J], [J, -delta]]))
    a, b, g = _zxz(W)
    q0, q1 = bond.q0, bond.q1
    ops = [qml.RZ(g / 2, q0), qml.RZ(-g / 2, q1)]                       # Rz'(g)
    ops += canonical_block(q0, q1, b / 4, b / 4, tau * bond.Jz)         # Rx'(b) and the ZZ part
    ops += [qml.RZ(a / 2 + 2 * tau * hbar, q0), qml.RZ(-a / 2 + 2 * tau * hbar, q1)]  # Rz'(a), hbar
    return ops


def bond_ops(bond: Bond, tau: float) -> list:
    if bond.h0 or bond.h1:
        return field_block(bond, tau)
    return canonical_block(bond.q0, bond.q1, tau * bond.Jx, tau * bond.Jy, tau * bond.Jz)


# ------------------------------------------------------------------ formula

def group_schedule(order: int, d: float) -> list[tuple[int, float]]:
    """Two-group schedule [(group, time)], group 0 outermost. Mirrors
    ``formula.first_order`` / ``second_order`` / ``suzuki`` (same p_k)."""
    if order == 1:
        return [(0, d), (1, d)]
    if order % 2 or order < 2:
        raise ValueError("orders are 1 or even >= 2")
    if order == 2:
        return [(0, d / 2), (1, d), (0, d / 2)]
    k = order // 2
    pk = (4 - (4 ** (1 / (2 * k - 1)))) ** (-1)
    a = group_schedule(order - 2, pk * d)
    b = group_schedule(order - 2, (1 - 4 * pk) * d)
    return a + a + b + a + a


OUTER = ("even", "odd")  # which group is applied twice per S2


def ordered(H: Hamiltonian, outer: str) -> list[list[Bond]]:
    even, odd = split_groups(H)
    if outer not in OUTER:
        raise ValueError(f"outer must be one of {OUTER}")
    return [even, odd] if outer == "even" else [odd, even]


def build_step(H: Hamiltonian, order: int, d: float, outer: str) -> qml.tape.QuantumScript:
    """One step as a tape; each group application emits a block per bond."""
    groups = ordered(H, outer)
    ops: list = []
    for g, time in group_schedule(order, d):
        for bond in groups[g]:
            ops += bond_ops(bond, time)
    return qml.tape.QuantumScript(ops)


# ------------------------------------------------------------------ step counts

def _group_matrices(H: Hamiltonian, outer: str) -> list[np.ndarray]:
    n = H.n_qubits
    return [Hamiltonian([t for b in g for t in b.terms(n)]).matrix() for g in ordered(H, outer)]


def step_unitary(mats: list[np.ndarray], order: int, d: float) -> np.ndarray:
    U = np.eye(mats[0].shape[0], dtype=complex)
    for g, time in group_schedule(order, d):
        U = expm(-1j * time * mats[g]) @ U
    return U


def steps_empirical(H: Hamiltonian, order: int, t: float, eps: float, outer: str) -> int:
    """Smallest r with ||S(t/r)^r - exp(-iHt)|| <= eps; same search as
    ``empirical.steps``."""
    mats = _group_matrices(H, outer)
    exact = expm(-1j * t * H.matrix())

    def err(r: int) -> float:
        return float(np.linalg.norm(np.linalg.matrix_power(step_unitary(mats, order, t / r), r) - exact, 2))

    lo, hi = 0, 1
    while err(hi) > eps:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if err(mid) <= eps else (mid, hi)
    return hi


def steps_tte(H: Hamiltonian, order: int, t: float, eps: float, outer: str) -> int:
    """Childs et al. (2021) bound over the two groups (Props 15/16, and J.1 for
    order 4), using ``tte``'s machinery on the grouped sums."""
    from . import tte

    n = H.n_qubits
    Hs = [tte._sentence([t_ for b in g for t_ in b.terms(n)]) for g in ordered(H, outer)]
    alpha = tte._BOUNDS[order](Hs, 1.0)
    return max(1, math.ceil((alpha * t ** (order + 1) / eps) ** (1 / order)))


# ------------------------------------------------------------------ comparison

def compare(n: int, seed: int, order: int, method: str = "empirical") -> dict:
    """xyz / middle option rows from ``blocks.compare`` plus both full variants."""
    from . import blocks, simcount
    from . import compare as cmp

    base = blocks.compare(n, seed, order, method)
    H = simcount.run(n, "pf2com", "cz", seed).hamiltonian
    t, eps = float(n), cmp.GATEBASES["cz"][0]
    out = {"xyz": (base.r["xyz"], base.raw["xyz"], base.nam["xyz"]),
           "middle": (base.r["blocked"], base.raw["blocked"], base.nam["blocked"])}
    for outer in OUTER:
        r = (steps_empirical if method == "empirical" else steps_tte)(H, order, t, eps, outer)
        tape = build_step(H, order, t / r, outer)
        # ASSUMPTION: Nam Light on one isolated step, as in blocks.compare. (unconfirmed)
        out[f"full/{outer}-out"] = (r, blocks._counts(tape), blocks._counts(blocks._nam(tape, n)))
    return out


def render(rows: list[tuple[int, int, str, dict]]) -> str:
    lines = ["=== full two-group option vs X-Y-Z and middle option  (cz, t = n, seed 1) ===",
             "per step: CNOT / rotations / total.   totals = r x Nam-optimised per step.", "",
             f"{'n':>3} {'ord':>3} {'variant':>15} {'r':>7} {'raw/step':>16} {'Nam/step':>16} "
             f"{'CNOT total':>12} {'rot total':>12} {'CNOT vs xyz':>12} {'rot vs xyz':>11}"]
    for n, order, method, res in rows:
        rx, _, nx = res["xyz"]
        for v, (r, raw, nam) in res.items():
            lines.append(
                f"{n:>3} {order:>3} {v:>15} {r:>7,} {f'{raw.cnot}/{raw.rot}/{raw.total}':>16} "
                f"{f'{nam.cnot}/{nam.rot}/{nam.total}':>16} {nam.cnot * r:>12,} {nam.rot * r:>12,} "
                f"{nam.cnot * r / (nx.cnot * rx):>12.3f} {nam.rot * r / (nx.rot * rx):>11.3f}")
        lines.append("")
    lines.append(f"r from: {rows[0][2]}.  Nam = Light on one isolated step.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="plsimcount.twogroup", description=__doc__.split("\n\n")[0])
    ap.add_argument("-n", "--sizes", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("-m", "--seed", type=int, default=1)
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--method", default="empirical", choices=["tte", "empirical"])
    args = ap.parse_args(argv)
    rows = [(n, o, args.method, compare(n, args.seed, o, args.method))
            for n in args.sizes for o in args.orders]
    print(render(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
