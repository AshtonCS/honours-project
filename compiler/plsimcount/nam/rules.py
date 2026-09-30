"""Rewrite and commutation rules for the Nam et al. optimiser.

Source: Y. Nam, N. J. Ross, Y. Su, A. M. Childs, D. Maslov, *Automated
optimization of large quantum circuits with continuous parameters*, npj
Quantum Information 4, 23 (2018); arXiv:1710.07345.

* :data:`HADAMARD_RULES` transcribes the five identities of Figure 4.
* :func:`commutes` implements the commutation test used by Routines 2 and 3.

Both are validated numerically -- see the DEVIATION note on :func:`commutes`,
and :func:`verify_hadamard_rules`, which multiplies out every identity and is
also run as a test. Figures 4 and 5 are circuit diagrams that do not extract
cleanly from the PDF, so the transcription below was reconstructed and then
checked against the matrices rather than trusted as read.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .gates import Gate, local_matrix

# Placeholder wire labels used inside rule templates.
C, T = 0, 1  # control, target (or simply "the wire" when only T is used)


@dataclass(frozen=True)
class Rule:
    """A local rewrite ``pattern -> replacement`` on one or two wires.

    Gate wires inside a rule are the placeholders :data:`C` / :data:`T`, bound
    to real wires when the rule is matched. ``per_wire`` lists, for each wire,
    the sequence of pattern positions that must be *consecutive on that wire*
    -- which is what makes the rule matchable without a full DAG walk.
    """

    name: str
    pattern: tuple[Gate, ...]
    replacement: tuple[Gate, ...]
    note: str = ""

    @property
    def wires(self) -> tuple[int, ...]:
        seen: list[int] = []
        for g in self.pattern:
            for w in g.wires:
                if w not in seen:
                    seen.append(w)
        return tuple(sorted(seen))

    def h_reduction(self) -> int:
        """How many Hadamard gates the rule removes."""
        count = lambda gs: sum(1 for g in gs if g.name == "H")  # noqa: E731
        return count(self.pattern) - count(self.replacement)


# --- Figure 4: Hadamard gate reductions ---------------------------------------
#
# The first two hold only up to a global phase of exp(i pi/4); the rest are
# exact. Global phase is not tracked (see gates.py), and the reference circuits
# carry no phase gates in cz mode.

HADAMARD_RULES: tuple[Rule, ...] = (
    Rule(
        "H-S-H",
        (Gate("H", (T,)), Gate("S", (T,)), Gate("H", (T,))),
        (Gate("Sdg", (T,)), Gate("H", (T,)), Gate("Sdg", (T,))),
        "up to global phase",
    ),
    Rule(
        "H-Sdg-H",
        (Gate("H", (T,)), Gate("Sdg", (T,)), Gate("H", (T,))),
        (Gate("S", (T,)), Gate("H", (T,)), Gate("S", (T,))),
        "up to global phase",
    ),
    Rule(
        "HH-CNOT-HH",
        (
            Gate("H", (C,)),
            Gate("H", (T,)),
            Gate("CNOT", (C, T)),
            Gate("H", (C,)),
            Gate("H", (T,)),
        ),
        (Gate("CNOT", (T, C)),),
        "reverses control and target; removes 4 H",
    ),
    Rule(
        "H-S-CNOT-Sdg-H",
        (
            Gate("H", (T,)),
            Gate("S", (T,)),
            Gate("CNOT", (C, T)),
            Gate("Sdg", (T,)),
            Gate("H", (T,)),
        ),
        (Gate("Sdg", (T,)), Gate("CNOT", (C, T)), Gate("S", (T,))),
        "removes 2 H",
    ),
    Rule(
        "H-Sdg-CNOT-S-H",
        (
            Gate("H", (T,)),
            Gate("Sdg", (T,)),
            Gate("CNOT", (C, T)),
            Gate("S", (T,)),
            Gate("H", (T,)),
        ),
        (Gate("S", (T,)), Gate("CNOT", (C, T)), Gate("Sdg", (T,))),
        "removes 2 H",
    ),
)


def verify_hadamard_rules(atol: float = 1e-12) -> list[tuple[str, bool, bool]]:
    """Multiply out each rule. Returns ``(name, exact, up_to_phase)`` rows."""
    rows: list[tuple[str, bool, bool]] = []
    for rule in HADAMARD_RULES:
        wires = rule.wires
        lhs = local_matrix(list(rule.pattern), wires)
        rhs = local_matrix(list(rule.replacement), wires)
        trace = np.trace(rhs.conj().T @ lhs)
        phase = np.exp(1j * np.angle(trace)) if abs(trace) > atol else 1.0
        rows.append(
            (
                rule.name,
                bool(np.allclose(lhs, rhs, atol=atol, rtol=0.0)),
                bool(np.allclose(lhs, phase * rhs, atol=atol, rtol=0.0)),
            )
        )
    return rows


# --- Figure 5: commutation ----------------------------------------------------


def commutes(a: Gate, b: Gate, atol: float = 1e-12) -> bool:
    """True if ``a`` and ``b`` may be exchanged without changing the circuit.

    DEVIATION from the paper. Nam et al. use a fixed list of commutation
    patterns (Figure 5), described as "sufficient (but not necessary)
    conditions for commutation". Here the condition is exact instead: gates
    commute if their supports are disjoint, or if their matrices commute on
    the union of their wires. That is a strict superset of any fixed pattern
    list, so this implementation may find cancellations theirs does not, and
    the optimised gate counts can come out slightly *lower* than the
    published ones. Keep this in mind when comparing against
    ``simcount/samples``; :func:`pattern_commutes` gives the restricted
    behaviour if a like-for-like comparison is wanted.

    Cost is at most a 4x4 matrix product, memoised on the gate signatures.
    """
    if not (set(a.wires) & set(b.wires)):
        return True
    return _commutes_cached(
        (a.name, a.wires, a.param), (b.name, b.wires, b.param), atol
    )


@lru_cache(maxsize=65536)
def _commutes_cached(
    sig_a: tuple[str, tuple[int, ...], float | None],
    sig_b: tuple[str, tuple[int, ...], float | None],
    atol: float,
) -> bool:
    a, b = Gate(*sig_a), Gate(*sig_b)
    wires = tuple(sorted(set(a.wires) | set(b.wires)))
    ab = local_matrix([a, b], wires)
    ba = local_matrix([b, a], wires)
    # rtol MUST be 0. numpy's default rtol=1e-05 would declare any rotation
    # with |theta| <~ 1e-5 to commute with gates it does not commute with --
    # rotation angles here are t/r and reach 1e-7, so the default silently
    # corrupted every circuit built from a loose step-count bound.
    return bool(np.allclose(ab, ba, atol=atol, rtol=0.0))


def pattern_commutes(a: Gate, b: Gate) -> bool:
    """The paper's restricted commutation patterns (Figure 5).

    Provided for a like-for-like comparison against the published counts.
    The patterns implemented here are the ones that could be recovered from
    the figure and then verified numerically:

    * disjoint supports;
    * ``Rz`` on the control of a ``CNOT``;
    * two ``Rz`` gates on the same wire;
    * ``CNOT`` gates sharing a control, or sharing a target;
    * ``X`` on the control of a ``CNOT`` (anti-commutes on the target, so
      excluded), ``X`` on the target of a ``CNOT``.
    """
    if not (set(a.wires) & set(b.wires)):
        return True
    if a.is_rotation and b.is_rotation and a.wires == b.wires:
        return True
    for u, v in ((a, b), (b, a)):
        if u.is_rotation and v.name == "CNOT" and u.wires[0] == v.wires[0]:
            return True  # Rz on the control
        if u.name == "X" and v.name == "CNOT" and u.wires[0] == v.wires[1]:
            return True  # X on the target
    if a.name == "CNOT" and b.name == "CNOT":
        if a.wires[0] == b.wires[0] or a.wires[1] == b.wires[1]:
            return True  # shared control, or shared target
    return False
