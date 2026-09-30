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


def routine4_rotation_merging(circuit: Circuit) -> Circuit:
    """Routine 4: rotation merging using phase polynomials. NOT IMPLEMENTED.

    Left deliberately unimplemented -- this is the subroutine with the actual
    intellectual content, and it is being written by hand as part of the
    thesis. What it has to do, from section 4.4 of the paper:

    1. Identify a maximal subcircuit of ``{X, CNOT, Rz}`` gates, built one
       wire at a time outwards from a designated CNOT. Stop exploring a wire
       at a Hadamard gate or at the circuit boundary (a "termination point");
       where a CNOT reaches a not-yet-visited wire, record an "anchor point"
       and recurse from there.
    2. Prune it so a phase polynomial representation exists: a wire that
       leaves and re-enters the subcircuit with an intervening Hadamard breaks
       the representation and must be excluded.
    3. For each Rz in the subcircuit, compute the affine function
       ``x_{i1} + ... + x_{ik} (+ 1)`` over GF(2) whose phase it applies, and
       where in the subcircuit that function appears.
    4. Merge all Rz gates sharing an affine function, placing the combined
       rotation at the **first (leftmost)** location computing it.
    5. Reduce affine functions to linear ones by NOT-propagation, to expose
       further collisions:
         ``X(a) CNOT(a,b) -> CNOT(a,b) X(a) X(b)``
         ``X(b) CNOT(a,b) -> CNOT(a,b) X(b)``
         ``X(a) Rz(a)     -> Rz(a)^dagger X(a)``

    Complexity given in the paper: O(g^2 log g), dominated by sorting the
    affine functions.
    """
    raise NotImplementedError(
        "Routine 4 (phase-polynomial rotation merging) is not implemented yet. "
        "Run the optimiser with skip_unimplemented=True to use Routines 1-3 only."
    )


ROUTINES = {
    1: routine1_hadamard_reduction,
    2: routine2_single_qubit_cancellation,
    3: routine3_two_qubit_cancellation,
    4: routine4_rotation_merging,
}
