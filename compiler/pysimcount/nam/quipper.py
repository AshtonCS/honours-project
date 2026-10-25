"""Reader for Quipper's ASCII circuit format.

Used to load the reference circuits in ``simcount/samples``, which are the
before/after-optimisation circuits published with Childs et al. (2017) and
optimised with the (unreleased) Nam et al. optimiser.

A file is a top-level call list plus a table of named subroutines. Subroutines
may call one another, so the whole circuit is only materialised on request --
expanding ``pf4com_preoptim_cz_13`` in full would be 77 million gates, whereas
its repeated core block is about a thousand.

Gate mapping (``cz`` gate set):

    QGate["H"](q)                          -> H(q)
    QGate["S"](q) / QGate["S"]*(q)         -> S(q) / Sdg(q)
    QGate["not"](t) with controls=[+c]     -> CNOT(c, t)
    QGate["not"](t)      (no controls)     -> X(t)
    QRot["exp(-i%Z)", t](q)                -> Rz(2t)

``exp(-i Z t)`` equals ``Rz(2t)`` in PennyLane's convention, so the parameter
is doubled on the way in and halved on the way out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .gates import Circuit, Gate

_INPUTS = re.compile(r"^Inputs:\s*(.*)$")
_SUB_HEADER = re.compile(r'^Subroutine:\s*"(.+)"')
_CALL_TOP = re.compile(r'^Subroutine\(x(\d+)\)\["(.+?)",')
_CALL_IN = re.compile(r'^Subroutine\["(.+?)",')
_QGATE = re.compile(r'^QGate\["(.+?)"\](\*?)\((\d+)\)(.*)$')
_QROT = re.compile(r'^QRot\["(.+?)",(-?[0-9.eE+-]+)\]\((\d+)\)')
_CONTROLS = re.compile(r"controls=\[([^\]]*)\]")


@dataclass
class QuipperFile:
    """Parsed Quipper ASCII circuit."""

    n_qubits: int
    toplevel: list[tuple[str, int]]  # (subroutine name, repeat count)
    subroutines: dict[str, list] = field(default_factory=dict)
    path: Path | None = None

    def expand(self, name: str) -> Circuit:
        """Materialise one subroutine, expanding any nested calls."""
        return Circuit(self._expand_gates(name, set()), self.n_qubits)

    def _expand_gates(self, name: str, stack: set[str]) -> list[Gate]:
        if name in stack:
            raise ValueError(f"recursive subroutine {name!r}")
        out: list[Gate] = []
        for item in self.subroutines[name]:
            if isinstance(item, Gate):
                out.append(item)
            else:  # a nested call: (callee, repeats)
                callee, repeats = item
                body = self._expand_gates(callee, stack | {name})
                for _ in range(repeats):
                    out.extend(body)
        return out

    def core(self) -> tuple[str, int]:
        """The subroutine that carries the repetition, and its repeat count.

        Pre-optimisation circuits call a single Trotter-step subroutine r
        times. Post-optimisation circuits are ``L . C^m . R``; the core is
        ``C``, the one with a repeat count above one.
        """
        repeated = [(n, k) for n, k in self.toplevel if k > 1]
        if len(repeated) != 1:
            raise ValueError(f"expected exactly one repeated subroutine, got {repeated}")
        return repeated[0]

    def total_counts(self) -> dict[str, int]:
        """Aggregate gate counts for the whole circuit, without expanding it."""
        memo: dict[str, dict[str, int]] = {}

        def counts_of(name: str) -> dict[str, int]:
            if name not in memo:
                acc: dict[str, int] = {}
                for item in self.subroutines[name]:
                    if isinstance(item, Gate):
                        acc[item.name] = acc.get(item.name, 0) + 1
                    else:
                        callee, repeats = item
                        for k, v in counts_of(callee).items():
                            acc[k] = acc.get(k, 0) + v * repeats
                memo[name] = acc
            return memo[name]

        out: dict[str, int] = {}
        for name, repeats in self.toplevel:
            for k, v in counts_of(name).items():
                out[k] = out.get(k, 0) + v * repeats
        return out


def _parse_gate(line: str) -> Gate | None:
    m = _QROT.match(line)
    if m:
        _, t, q = m.groups()
        return Gate("Rz", (int(q),), 2.0 * float(t))

    m = _QGATE.match(line)
    if not m:
        return None
    name, dagger, wire, rest = m.groups()
    wire = int(wire)

    if name == "not":
        ctrl = _CONTROLS.search(rest)
        if ctrl and ctrl.group(1).strip():
            targets = [c.strip() for c in ctrl.group(1).split(",") if c.strip()]
            if len(targets) != 1:
                raise ValueError(f"only singly-controlled not supported: {line!r}")
            sign, control = targets[0][0], int(targets[0][1:])
            if sign != "+":
                raise ValueError(f"negative control not supported: {line!r}")
            return Gate("CNOT", (control, wire))
        return Gate("X", (wire,))

    if name == "H":
        return Gate("H", (wire,))
    if name == "S":
        return Gate("Sdg" if dagger else "S", (wire,))
    if name == "X":
        return Gate("X", (wire,))
    if name == "Z":
        return Gate("Z", (wire,))
    raise ValueError(f"unsupported gate in {line!r}")


def read(path: str | Path) -> QuipperFile:
    """Parse a Quipper ASCII circuit file."""
    path = Path(path)
    n_qubits = 0
    toplevel: list[tuple[str, int]] = []
    subroutines: dict[str, list] = {}
    current: str | None = None

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue

        m = _SUB_HEADER.match(line)
        if m:
            current = m.group(1)
            subroutines[current] = []
            continue

        m = _INPUTS.match(line)
        if m:
            if current is None:
                n_qubits = len([x for x in m.group(1).split(",") if x.strip()])
            continue

        if line.startswith(("Outputs:", "Shape:", "Controllable:")):
            continue

        m = _CALL_TOP.match(line)
        if m:
            repeats, name = int(m.group(1)), m.group(2)
            if current is None:
                toplevel.append((name, repeats))
            else:
                subroutines[current].append((name, repeats))
            continue

        m = _CALL_IN.match(line)
        if m:
            if current is None:
                toplevel.append((m.group(1), 1))
            else:
                subroutines[current].append((m.group(1), 1))
            continue

        gate = _parse_gate(line)
        if gate is None:
            raise ValueError(f"unparsed line in {path.name}: {raw!r}")
        if current is None:
            raise ValueError(f"gate outside a subroutine in {path.name}: {raw!r}")
        subroutines[current].append(gate)

    if not n_qubits:
        raise ValueError(f"no Inputs line found in {path}")
    return QuipperFile(n_qubits, toplevel, subroutines, path)
