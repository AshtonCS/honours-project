"""Gate counting with simcount-compatible labels.

Counts are plain ``{label: count}`` dictionaries using the labels Quipper
prints in ``-f gatecount`` output, so the PennyLane and simcount numbers can
be laid side by side without translation:

    H, S, S*, not, Rot(exp(-i%Z),<t>), T, T*, X, omega

For a rotation, ``<t>`` is the parameter of exp(-i Z t), i.e. RZ angle / 2.
"""

from __future__ import annotations

import re
from collections import Counter

import pennylane as qml

ROT = "Rot(exp(-i%Z),"
_ROT_RE = re.compile(r"Rot\(exp\(-i%Z\),(-?[0-9.eE+-]+)\)")

_NAMES = {
    "Hadamard": "H",
    "S": "S",
    "Adjoint(S)": "S*",
    "CNOT": "not",
    "T": "T",
    "Adjoint(T)": "T*",
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
    "GlobalPhase": "omega",
    "Identity": "I",
}


def label(op) -> str:
    if op.name == "RZ":
        return f"{ROT}{float(op.data[0]) / 2!r})"
    if op.name in _NAMES:
        return _NAMES[op.name]
    if op.name in ("RX", "RY"):
        return f"{op.name}({float(op.data[0])!r})"
    return op.name


def rot_value(lbl: str) -> float | None:
    m = _ROT_RE.fullmatch(lbl)
    return float(m.group(1)) if m else None


def is_rot(lbl: str) -> bool:
    return lbl.startswith(ROT)


def count(tape: qml.tape.QuantumScript) -> dict[str, int]:
    return dict(Counter(label(op) for op in tape.operations))


def scale(counts: dict[str, int], k: int) -> dict[str, int]:
    return {lbl: c * k for lbl, c in counts.items()}


def total(counts: dict[str, int]) -> int:
    return sum(counts.values())


def rotations(counts: dict[str, int]) -> int:
    return sum(c for lbl, c in counts.items() if is_rot(lbl))


def summary(counts: dict[str, int]) -> dict[str, int]:
    """Coarse view: Clifford / T / rotation / CNOT / total."""
    t = counts.get("T", 0) + counts.get("T*", 0)
    rot = rotations(counts)
    omega = counts.get("omega", 0)
    tot = total(counts)
    return {
        "CNOT": counts.get("not", 0),
        "T": t,
        "Rz": rot,
        "Clifford (excl. CNOT)": tot - t - rot - omega - counts.get("not", 0),
        "omega (global phase)": omega,
        "Total": tot,
        "Total excl. omega": tot - omega,
    }


def _sort_key(lbl: str):
    # Quipper sorts labels lexicographically; rotations sort among themselves
    # by their printed value, which is what a reader expects.
    v = rot_value(lbl)
    return (1, v, lbl) if v is not None else (0, 0.0, lbl)


def format_gatecount(counts: dict[str, int], title: str = "Aggregated gate count:") -> str:
    """Render in Quipper's ``-f gatecount`` layout."""
    width = max((len(str(c)) for c in counts.values()), default=1)
    lines = [title]
    for lbl in sorted(counts, key=_sort_key):
        arity = ", controls 1" if lbl == "not" else ""
        lines.append(f'{counts[lbl]:>{width}}: "{lbl}, arity 1"{arity}')
    lines.append(f"Total gates: {total(counts)}")
    return "\n".join(lines)
