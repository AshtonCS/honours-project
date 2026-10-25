"""Product formulas with arbitrary weights, and one merge rule for every formula.

simcount only knows Suzuki's recursion (``formula.py``). This module adds the
schemes of Maležič & Ostmeyer (arXiv:2601.18756), or any other symmetric
two-operator scheme, as ordinary pysimcount *schedules*: lists of
``(term_index, time)`` meaning "apply exp(-i * time * coeff * P)". Everything
downstream (gate templates, resource counts, synthesis, brute-force r) only
consumes schedules, so nothing else has to change.

From weights to a schedule (paper Eqs. 7 and 9)
-----------------------------------------------
A scheme is given in ramp form: q forward ramps with weights c_i and q
backward ramps with weights d_i (``effts.schemes.Scheme.ramps``). The
Hamiltonian's terms are split into *stages*: maximal consecutive runs of
mutually commuting terms in simcount's term order. For the Heisenberg chain
that is exactly three stages, XX bonds | YY bonds | ZZ bonds + fields, the
paper's global grouping S^(3) (Eq. 27). One step of size d is then

    for i = 1..q:  stages 1..Λ at time c_i * d,  then stages Λ..1 at d_i * d

with the term order inside a stage also reversed on the backward ramp, so the
whole schedule is a palindrome.

Merging (decision 1, both reported)
-----------------------------------
``merge`` walks a schedule and collects runs of mutually commuting terms into
*blocks*; inside a block, repeated terms are combined by adding their times.
Because the terms in a block commute, this is exact: the unitary is unchanged
and only the gate count drops. It is applied identically to every formula,
Suzuki included, so neither side gets an advantage the other doesn't. Note
that merging can also reorder terms inside a block (first-appearance order),
which is harmless for the same reason.

Not a simcount feature: simcount's own output is the *unmerged* count.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import pennylane as qml

from . import formula
from .exponentiate import TEMPLATES, Template
from .hamiltonian import Hamiltonian, Term

Schedule = list[tuple[int, float]]


# ------------------------------------------------------------------ Pauli algebra

def commute(a: Term, b: Term) -> bool:
    """Pauli words commute iff they differ (both non-identity) on an even number of qubits."""
    clashes = 0
    for p, q in zip(a.word, b.word):
        if p != "I" and q != "I" and p != q:
            clashes += 1
    return clashes % 2 == 0


def stage_groups(H: Hamiltonian) -> list[list[int]]:
    """Split H's terms, in order, into maximal runs of mutually commuting terms.

    For simcount's Heisenberg Hamiltonian (Pauli-major order) this gives
    [XX bonds], [YY bonds], [ZZ bonds + fields]: Λ = 3.
    """
    groups: list[list[int]] = []
    for j, term in enumerate(H.terms):
        if groups and all(commute(term, H.terms[k]) for k in groups[-1]):
            groups[-1].append(j)
        else:
            groups.append([j])
    return groups


# ------------------------------------------------------------------ schedules

def ramp_schedule(H: Hamiltonian, ramps: tuple, d: float) -> Schedule:
    """Eq. (7) with stages from ``stage_groups``: forward ramps at c_i * d,
    backward ramps (stage order and term order reversed) at d_i * d."""
    c, dd = ramps
    groups = stage_groups(H)
    out: Schedule = []
    for i in range(len(c)):
        for g in groups:
            for j in g:
                out.append((j, c[i] * d))
        for g in reversed(groups):
            for j in reversed(g):
                out.append((j, dd[i] * d))
    return out


def commuting_blocks(H: Hamiltonian, schedule: Schedule) -> list[dict[int, float]]:
    """Group a schedule into consecutive blocks of mutually commuting terms,
    summing the times of repeated terms. Dicts keep first-appearance order."""
    blocks: list[dict[int, float]] = []
    for j, time in schedule:
        if blocks and all(commute(H.terms[j], H.terms[k]) for k in blocks[-1]):
            blocks[-1][j] = blocks[-1].get(j, 0.0) + time
        else:
            blocks.append({j: time})
    return blocks


def merge(H: Hamiltonian, schedule: Schedule) -> Schedule:
    """The merge rule: same unitary, one exponential per term per commuting block."""
    out: Schedule = []
    for block in commuting_blocks(H, schedule):
        out.extend(block.items())
    return out


# ------------------------------------------------------------------ formulas

@dataclass(frozen=True)
class FormulaSpec:
    """A product formula as used by the comparison.

    name: label in results tables.
    family: 'suzuki' (simcount's own), 'paper' (Maležič & Ostmeyer),
        or 'historical' (an earlier non-Suzuki scheme).
    order: the formula's order (paper's n, simcount's p).
    q: cycles per step (Suzuki S_2k has 5^(k-1)).
    schedule: d -> unmerged schedule for one step of size d.
    """

    name: str
    family: str
    order: int
    q: int
    schedule: Callable[[Hamiltonian, float], Schedule]


def suzuki_formula(order: int) -> FormulaSpec:
    """simcount's own Suzuki S_order, exactly as ``formula.schedule`` builds it."""
    if order % 2 or order < 2:
        raise ValueError("Suzuki formulas here are the even orders 2, 4, 6, 8")
    q = 5 ** (order // 2 - 1)
    return FormulaSpec(f"Suzuki S{order}", "suzuki", order, q,
                       lambda H, d: formula.schedule(order, H, d))


def weighted_formula(scheme, name: str, family: str) -> FormulaSpec:
    """A symmetric two-operator scheme (``effts.schemes.Scheme``) via its ramps."""
    ramps = scheme.ramps()
    return FormulaSpec(name, family, scheme.order, scheme.q,
                       lambda H, d: ramp_schedule(H, ramps, d))


def suzuki_scheme(order: int):
    """Suzuki S_order as a two-operator stage-form Scheme (A = first stages,
    B = rest), built by the same recursion as ``formula.suzuki``. Used only to
    check that ``ramp_schedule`` and ``formula.suzuki`` agree."""
    from effts.schemes import Scheme

    def rec(o: int, x: float) -> list[tuple[str, float]]:
        if o == 2:
            return [("A", x / 2), ("B", x), ("A", x / 2)]
        k = o // 2
        p = (4 - 4 ** (1 / (2 * k - 1))) ** (-1)  # same expression as formula.suzuki
        a, b = rec(o - 2, p * x), rec(o - 2, (1 - 4 * p) * x)
        return a + a + b + a + a

    merged: list[tuple[str, float]] = []
    for op, w in rec(order, 1.0):
        if merged and merged[-1][0] == op:
            merged[-1] = (op, merged[-1][1] + w)
        else:
            merged.append((op, w))
    a = tuple(w for op, w in merged if op == "A")
    b = tuple(w for op, w in merged if op == "B")
    return Scheme(a, b, order, f"Suzuki S{order} (stage form)")


# ------------------------------------------------------------------ circuits

def build_step_from_schedule(H: Hamiltonian, schedule: Schedule,
                             template: str | Template = "simcount") -> qml.tape.QuantumScript:
    """``circuit.build_step`` for an explicit schedule (same templates)."""
    fn = TEMPLATES[template] if isinstance(template, str) else template
    ops = []
    for j, time in schedule:
        ops += fn(H.terms[j], time)
    return qml.tape.QuantumScript(ops)
