"""Optimisation subroutines 1-4 of Nam et al. (arXiv:1710.07345, section 4.4).

Implemented here: Routine 1 (Hadamard gate reduction), Routine 2 (single-qubit
gate cancellation and rotation merging), Routine 3 (two-qubit gate
cancellation). Routine 4 (rotation merging via phase polynomials) is declared
but **not implemented** -- see :func:`routine4_rotation_merging`.

Routine 5 (floating Rz gates) is not implemented and cannot be: it depends on
external rewrite-rule library files that were never published. The paper
reports that for product-formula circuits the Heavy optimiser (the only one
using Routine 5) gives no improvement over Light, so nothing is lost here.

DEVIATION from the paper: Nam et al. describe these routines over an explicit
DAG representation. Routines 2 and 3 are implemented here as a rightward walk
along the flat gate list, which is the same algorithm -- "repeatedly check
whether U commutes through a set of consecutive gates" -- with the same O(g^2)
complexity, and is easier to read. Routine 1 uses per-wire neighbour maps,
rebuilt once per pass, since its patterns are defined by adjacency *on a wire*
rather than in the list.
"""

from __future__ import annotations

import math

from .gates import Circuit, Gate, merged_rotation
from .rules import HADAMARD_RULES, Rule, commutes

# Signature of a routine: it mutates nothing and returns a new Circuit.
WireNeighbours = tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]


def wire_neighbours(gates: list[Gate]) -> WireNeighbours:
    """Per-wire predecessor and successor maps.

    Returns ``(prev, next)`` where ``next[(i, w)]`` is the index of the next
    gate acting on wire ``w`` after gate ``i``, or ``-1`` if there is none.
    """
    last_on: dict[int, int] = {}
    prev: dict[tuple[int, int], int] = {}
    nxt: dict[tuple[int, int], int] = {}
    for i, g in enumerate(gates):
        for w in g.wires:
            j = last_on.get(w, -1)
            prev[(i, w)] = j
            if j >= 0:
                nxt[(j, w)] = i
            last_on[w] = i
    for i, g in enumerate(gates):
        for w in g.wires:
            nxt.setdefault((i, w), -1)
    return prev, nxt


# --- Routine 1 ----------------------------------------------------------------


def _match_rule(
    rule: Rule, gates: list[Gate], anchor: int, nbrs: WireNeighbours
) -> dict[int, int] | None:
    """Try to match ``rule`` with its CNOT (or middle gate) at ``anchor``.

    Returns a map from pattern position to gate index, or ``None``.
    Single-wire rules anchor on their middle gate; two-wire rules on the CNOT.
    """
    prev, nxt = nbrs
    pattern = rule.pattern
    mid = next(
        (k for k, g in enumerate(pattern) if g.name == "CNOT"),
        len(pattern) // 2,
    )
    anchor_gate = gates[anchor]
    if anchor_gate.name != pattern[mid].name:
        return None

    # Bind placeholder wires to real wires using the anchor gate.
    binding: dict[int, int] = {}
    for placeholder, real in zip(pattern[mid].wires, anchor_gate.wires):
        binding[placeholder] = real
    if len(set(binding.values())) != len(binding):
        return None

    found: dict[int, int] = {mid: anchor}
    # Walk outwards along each wire, matching pattern positions in order.
    for direction, positions in (
        (-1, range(mid - 1, -1, -1)),
        (+1, range(mid + 1, len(pattern))),
    ):
        cursor: dict[int, int] = {w: anchor for w in anchor_gate.wires}
        for pos in positions:
            want = pattern[pos]
            if len(want.wires) != 1:
                return None
            placeholder = want.wires[0]
            if placeholder not in binding:
                return None
            wire = binding[placeholder]
            here = cursor.get(wire)
            if here is None:
                return None
            step = prev[(here, wire)] if direction < 0 else nxt[(here, wire)]
            if step < 0:
                return None
            got = gates[step]
            if got.name != want.name or got.wires != (wire,):
                return None
            found[pos] = step
            cursor[wire] = step
    return found


def routine1_hadamard_reduction(circuit: Circuit) -> Circuit:
    """Routine 1: apply the Figure 4 identities to reduce the Hadamard count.

    Hadamard gates do not participate in phase-polynomial optimisation and
    obstruct commutation, so they are reduced first.
    """
    gates = list(circuit.gates)
    nbrs = wire_neighbours(gates)
    claimed: set[int] = set()
    edits: list[tuple[dict[int, int], Rule]] = []

    for anchor in range(len(gates)):
        if anchor in claimed:
            continue
        for rule in HADAMARD_RULES:
            if rule.h_reduction() <= 0:
                continue
            found = _match_rule(rule, gates, anchor, nbrs)
            if found is None or set(found.values()) & claimed:
                continue
            edits.append((found, rule))
            claimed |= set(found.values())
            break

    if not edits:
        return circuit.copy()

    # Apply: drop every matched gate, then splice the replacement in at the
    # position of the earliest matched gate.
    replacement_at: dict[int, list[Gate]] = {}
    drop: set[int] = set()
    for found, rule in edits:
        binding: dict[int, int] = {}
        for pos, idx in found.items():
            for placeholder, real in zip(rule.pattern[pos].wires, gates[idx].wires):
                binding[placeholder] = real
        drop |= set(found.values())
        replacement_at[min(found.values())] = [
            Gate(g.name, tuple(binding[w] for w in g.wires), g.param)
            for g in rule.replacement
        ]

    out: list[Gate] = []
    for i, g in enumerate(gates):
        if i in replacement_at:
            out.extend(replacement_at[i])
        if i not in drop:
            out.append(g)
    return Circuit(out, circuit.n_qubits)


# --- Routines 2 and 3 ---------------------------------------------------------


def _cancel_pass(circuit: Circuit, two_qubit: bool, merge_rotations: bool) -> Circuit:
    """Shared engine for Routines 2 and 3.

    For each gate U, walk rightwards. A gate that shares no wire with U is
    skipped. If the gate is U-inverse, both are removed; if it is an Rz on the
    same wire as an Rz U and ``merge_rotations``, they are combined. Otherwise
    the walk continues only while U commutes past the gate.
    """
    gates: list[Gate | None] = list(circuit.gates)

    for i, u in enumerate(gates):
        if u is None:
            continue
        if two_qubit and u.name != "CNOT":
            continue
        if not two_qubit and u.name == "CNOT":
            continue

        u_wires = set(u.wires)
        for j in range(i + 1, len(gates)):
            v = gates[j]
            if v is None:
                continue
            if not (u_wires & set(v.wires)):
                continue
            if u.cancels_with(v):
                gates[i] = None
                gates[j] = None
                break
            if merge_rotations and u.is_rotation and v.is_rotation and u.wires == v.wires:
                gates[i] = None
                gates[j] = merged_rotation(u, v)
                break
            if not commutes(u, v):
                break

    return Circuit([g for g in gates if g is not None], circuit.n_qubits)


def routine2_single_qubit_cancellation(circuit: Circuit) -> Circuit:
    """Routine 2: cancel a single-qubit gate against its inverse, and merge
    adjacent-after-commutation Rz rotations into one."""
    return _cancel_pass(circuit, two_qubit=False, merge_rotations=True)


def routine3_two_qubit_cancellation(circuit: Circuit) -> Circuit:
    """Routine 3: as Routine 2, but for CNOT gates (the only two-qubit gate
    in these circuits)."""
    return _cancel_pass(circuit, two_qubit=True, merge_rotations=False)


# --- Routine 4 ----------------------------------------------------------------
#
# Routine 4 of Nam et al. (arXiv:1710.07345 section 4.4); the phase polynomial
# representation it relies on is section 4.2, equations 3-6.
#
# Two simplifications relative to the paper, both justified by section 4.1,
# where the gate set is {NOT, H, Rz, CNOT} and "the phase gate p and the t gate
# can be obtained from rz(theta) ... as rz(pi/2) and rz(pi/4)":
#
#   * S and S-dagger are Z-rotations, so they take part in merging like any
#     other rotation. A merged angle landing on a multiple of pi/2 is mapped
#     back to a Clifford gate on the way out -- leaving it as an Rz would hand a
#     free gate to the synthesis stage to approximate, inflating the T-count.
#   * The paper's affine functions carry a trailing "+ 1", and its step 5
#     propagates NOT gates to remove those offsets. Neither is needed here: the
#     circuits this runs on contain no X gates, so every function is a plain
#     XOR. Reinstate both if X gates are ever introduced upstream.

HALF_PI = math.pi / 2
TWO_PI = 2.0 * math.pi

#: How close to a multiple of pi/2, measured in quarter turns, counts as exact.
#: Absolute rather than relative: this compares an angle against a fixed lattice
#: of Clifford angles, not against its own magnitude.
QUARTER_TURN_ATOL = 1e-9

#: Gates that may appear inside a stretch admitting a phase polynomial.
PHASE_GATE_NAMES: frozenset[str] = frozenset({"CNOT", "Rz", "S", "Sdg", "Z", "X"})

#: Z-rotation angle of each fixed Clifford that is one.
CLIFFORD_ANGLE: dict[str, float] = {"S": HALF_PI, "Sdg": -HALF_PI, "Z": math.pi}


def as_rotation(gate: Gate) -> float | None:
    """The angle this gate rotates by about Z, or None if it is not a Z-rotation.

    Global phase is ignored, so ``S`` reads as ``Rz(pi/2)`` exactly.
    """
    if gate.name == "Rz":
        return gate.param
    return CLIFFORD_ANGLE.get(gate.name)


def from_rotation(angle: float, wires: tuple[int, ...]) -> Gate | None:
    """Cheapest gate implementing a Z-rotation by ``angle``; None for identity.

    A rotation by a multiple of pi/2 is Clifford and free to implement exactly,
    so it is emitted as ``S``, ``Z`` or ``Sdg`` rather than ``Rz``. Anything else
    stays an ``Rz``, with its angle folded into (-pi, pi].
    """
    quarter_turns = (angle / HALF_PI) % 4.0
    nearest = round(quarter_turns)
    if abs(quarter_turns - nearest) < QUARTER_TURN_ATOL:
        name = {0: None, 1: "S", 2: "Z", 3: "Sdg"}[nearest % 4]
        return None if name is None else Gate(name, wires)
    return Gate("Rz", wires, ((angle + math.pi) % TWO_PI) - math.pi)


def phase_stretches(circuit: Circuit) -> list[list[int]]:
    """Maximal stretches of gates admitting a phase polynomial, as index lists.

    A phase polynomial exists only where every gate is a NOT, CNOT or rotation
    (section 4.2). A Hadamard breaks it -- but only on *its own* wire, which is
    what the paper exploits by growing a stretch outwards one wire at a time
    rather than ending it wholesale.

    Implemented by tracking which wires are still usable. A wire is live from
    the start of a stretch until a non-phase gate touches it; a gate may join
    only while all of its wires are live. That is what makes each stretch valid:
    every gate on a given wire precedes that wire's first Hadamard, so no wire
    can leave the stretch and re-enter it. The paper needs a separate pruning
    step for exactly that case; here it cannot arise by construction.

    Stretches carrying fewer than two rotations are dropped -- nothing can merge.
    """
    stretches: list[list[int]] = []
    current: list[int] = []
    live: set[int] = set(range(circuit.n_qubits))

    def close() -> None:
        nonlocal current, live
        rotations = sum(1 for i in current if as_rotation(circuit.gates[i]) is not None)
        if rotations >= 2:
            stretches.append(current)
        current = []
        live = set(range(circuit.n_qubits))

    for i, gate in enumerate(circuit.gates):
        if gate.name not in PHASE_GATE_NAMES:
            live -= set(gate.wires)          # retire this wire; the others go on
            continue
        if all(wire in live for wire in gate.wires):
            current.append(i)
            continue
        close()                              # cannot join, so begin a new stretch
        current.append(i)

    close()
    return stretches


def merge_stretch(circuit: Circuit, indices: list[int]) -> dict[int, Gate | None]:
    """Merge rotations within one stretch. Returns a replacement per gate index,
    where ``None`` means the gate is removed.

    CNOT gates never create superposition; they only XOR wires together. So at
    every point in a stretch each wire carries some XOR of the wires the stretch
    began with, tracked here as a set of wire labels: ``{2}`` means the wire
    still carries its own input, ``{1, 2}`` that it carries their XOR. A rotation
    applies a phase determined by whatever its wire currently carries, so two
    rotations whose wires carry the *same set* are one rotation split in two --
    however far apart, and however many CNOT gates lie between -- and their
    angles add.

    The combined rotation is placed at the leftmost rotation on that function,
    which is the paper's choice. Any location carrying the same function would
    be valid, and exploiting that freedom to expose CNOT cancellations is what
    the paper's Routine 5 generalises.

    Two details that are easy to get wrong: the surviving gate keeps its own
    wire, since distinct wires can carry the same function simultaneously; and a
    wire can come to carry the empty set, meaning constant zero, where a
    rotation contributes only a global phase.
    """
    holds: dict[int, frozenset[int]] = {}
    for i in indices:
        for wire in circuit.gates[i].wires:
            holds.setdefault(wire, frozenset([wire]))

    angles: dict[int, float] = {}
    first_on: dict[frozenset[int], int] = {}
    replacements: dict[int, Gate | None] = {}

    for i in indices:
        gate = circuit.gates[i]
        if gate.name == "CNOT":
            control, target = gate.wires
            holds[target] = holds[control] ^ holds[target]
            continue
        if gate.name == "X":
            continue                          # no X gates arise in these circuits
        angle = as_rotation(gate)
        if angle is None:
            raise ValueError(f"{gate.name} cannot appear in a phase stretch")
        function = holds[gate.wires[0]]
        if function not in first_on:
            first_on[function] = i
            angles[i] = angle
        else:
            angles[first_on[function]] += angle
            replacements[i] = None

    for i, total in angles.items():
        replacements[i] = from_rotation(total, circuit.gates[i].wires)
    return replacements


def routine4_rotation_merging(circuit: Circuit) -> Circuit:
    """Routine 4: merge rotations that apply a phase to the same function.

    Rotations separated by a network of CNOT gates can be applying a phase to
    the same XOR of the inputs, in which case they are one rotation split in two
    and their angles add. Sliding gates past one another never reveals this --
    it is visible only in the phase polynomial representation. This is the
    routine that reduces the rotation count, and the freedom it creates in where
    to place a merged rotation is what subsequently lets Routine 3 remove CNOT
    gates.
    """
    replacements: dict[int, Gate | None] = {}
    for indices in phase_stretches(circuit):
        replacements.update(merge_stretch(circuit, indices))

    if not replacements:
        return circuit.copy()

    out: list[Gate] = []
    for i, gate in enumerate(circuit.gates):
        if i not in replacements:
            out.append(gate)
        elif replacements[i] is not None:
            out.append(replacements[i])
    return Circuit(out, circuit.n_qubits)


ROUTINES = {
    1: routine1_hadamard_reduction,
    2: routine2_single_qubit_cancellation,
    3: routine3_two_qubit_cancellation,
    4: routine4_rotation_merging,
}
