"""Bond-block compilation for the even/odd ordering -- the "middle option".

Same formula and same grouping as ``evenodd`` (even bonds | odd bonds |
fields), but each bond's three commuting terms are compiled together:

    exp(-i (a XX + b YY + c ZZ))   with a, b, c = time * (Jx, Jy, Jz)

as one 3-CNOT, 3-rotation block instead of three separate Pauli
exponentials (6 CNOTs, 3 rotations). The block is exact (up to global
phase), so the step implements the same unitary as the term-by-term
even/odd step and the Trotter number r is unchanged; only the gates per
step change.

Circuit: the 3-CNOT construction for the two-qubit canonical gate of
F. Vatan and C. Williams, "Optimal quantum circuits for general two-qubit
gates", Phys. Rev. A 69, 032315 (2004), arXiv:quant-ph/0308006, Sec. V,
derived in Figs. 4 -> 5 -> 6. Checked against the published Fig. 6 (and the
Fig. 5 caption) on 2026-10-21; see ``canonical_block`` for exactly what
matches and what differs. In short: this code is the *adjoint* of Fig. 6 --
the same eight gates on the same qubits with the same angle expressions,
in reverse time order -- because the paper builds
N(alpha, beta, gamma) = exp(+i(alpha XX + beta YY + gamma ZZ)) with
R(theta) = exp(+i theta sigma/2), while this code needs exp(-i(...)) with
PennyLane's R(theta) = exp(-i theta sigma/2). Correctness still rests on the
exact-matrix tests in ``tests/test_blocks.py``; the citation records where
the construction comes from.

Gate set: {H, S, S-dagger, CNOT, RZ}, as ``exponentiate.simcount_template``.
The fixed +-pi/2 parts are emitted as S / S-dagger rather than folded into
RZ angles, so the RZ count stays equal to the number of genuine rotations
(the convention ``exponentiate.py`` uses for the synthesis error budget).

    python -m pysimcount.blocks -n 4 6 8 --orders 2 4
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass

import pennylane as qml

from . import formula
from .exponentiate import simcount_template
from .hamiltonian import Hamiltonian, Term

# ------------------------------------------------------------------ the block

def _ry(theta_rot: float | None, q: int, extra: str | None = None) -> list:
    """RY(theta) on qubit q in the Clifford+RZ gate set, via
    RY(theta) = S H RZ(theta) H S-dagger (operator order; time order is
    reversed). ``extra`` adds a Clifford RZ(+-pi/2) = S / S-dagger (up to
    phase) beside the genuine rotation; both are Z-rotations so they commute."""
    mid = [qml.RZ(theta_rot, q)]
    if extra == "S":
        mid.append(qml.S(q))
    elif extra == "Sdg":
        mid.append(qml.adjoint(qml.S(q)))
    return [qml.adjoint(qml.S(q)), qml.Hadamard(q), *mid, qml.Hadamard(q), qml.S(q)]


def canonical_block(q0: int, q1: int, a: float, b: float, c: float) -> list:
    """exp(-i (a X_q0 X_q1 + b Y_q0 Y_q1 + c Z_q0 Z_q1)), exactly.

    a, b, c are dimensionless (time x coupling). Rotations are PennyLane's,
    RZ(theta) = exp(-i theta Z/2), RY(theta) = exp(-i theta Y/2). Time order
    (left acts first), with RZ(+-pi/2) written as S / S-dagger:

      q0: Sdg --X-- . ------------ RZ(2c-pi/2) --X------
      q1: ------.-- RY(2b-pi/2) --X-- RY(pi/2-2a) --.-- S

    Relation to Vatan & Williams (PRA 69, 032315, Fig. 6), checked against
    the published figure and numerically (``tests/test_blocks.py``,
    ``test_block_is_adjoint_of_vatan_williams_fig6``):

    * Paper, Fig. 6, time order, top wire = q0, bottom wire = q1:
      bottom Rz(pi/2); CNOT bottom->top; top Rz(2 gamma - pi/2) and bottom
      Ry(pi/2 - 2 alpha); CNOT top->bottom; bottom Ry(2 beta - pi/2);
      CNOT bottom->top; top Rz(-pi/2). The Fig. 5 caption defines
      S1 = Rz(pi/2), S2 = Rz(2 gamma - pi/2), T1 = Ry(pi/2 - 2 alpha),
      T2 = Ry(2 beta - pi/2); Fig. 6 is Fig. 5 after the two
      simplifications in the text. The paper states Fig. 6 equals
      N(alpha, beta, gamma) = exp(+i(alpha XX + beta YY + gamma ZZ)) up to a
      missing global phase e^{i pi/4}.
    * What matches: the same eight gates, the same qubit for each gate, the
      same CNOT directions, and the same angle expressions with
      (alpha, beta, gamma) -> (a, b, c).
    * What differs: (1) the time order is reversed -- this code is the
      figure read right to left. (2) The rotation convention: Fig. 6 equals
      e^{-i pi/4} N(alpha, beta, gamma) only if Rz(theta) = exp(+i theta Z/2),
      Ry(theta) = exp(+i theta Y/2) (verified numerically; with PennyLane's
      sign it does not reproduce N at all). Under PennyLane's convention each
      gate here is the inverse of the paper's gate with the same label, so
      reversing the order gives the adjoint: this block = e^{-i pi/4} x
      (Fig. 6)^dagger = N(a, b, c)^dagger = exp(-i(aXX + bYY + cZZ)) -- the
      opposite sign of the paper's N, which is what the Trotter step needs.
      (3) The +-pi/2 parts are emitted as S / S-dagger, whose phases
      e^{+-i pi/4} make the block exact *including* the global phase (the
      paper's Fig. 6 omits e^{i pi/4}). The Nam optimiser drops global
      phases, so nothing downstream depends on (3)."""
    ops = [qml.adjoint(qml.S(q0)), qml.CNOT([q1, q0])]
    ops += _ry(2 * b, q1, extra="Sdg")            # RY(2b - pi/2)
    ops += [qml.CNOT([q0, q1])]
    ops += [qml.RZ(2 * c, q0), qml.adjoint(qml.S(q0))]  # RZ(2c - pi/2)
    ops += _ry(-2 * a, q1, extra="S")             # RY(pi/2 - 2a)
    ops += [qml.CNOT([q1, q0]), qml.S(q1)]
    return ops


# ------------------------------------------------------------------ the step

def _commute(p: Term, q: Term) -> bool:
    """Pauli words commute iff they clash (differ, both non-identity) on an
    even number of qubits."""
    clashes = sum(1 for x, y in zip(p.word, q.word) if x != "I" and y != "I" and x != y)
    return clashes % 2 == 0


def _bond_copies(terms: list[Term]) -> list[tuple[float, float, float]] | None:
    """[(Jx, Jy, Jz)] * k if ``terms`` are k copies each of XX, YY, ZZ on one
    pair, else None.

    k = 2 happens in Suzuki's 4th order: the reversed half of one S2 and the
    first half of the next run at the same time, so a bond's terms appear
    twice in one commuting run. Each copy becomes its own block. Merging them
    into one block with doubled angles would be valid, but the term-by-term
    variants do not get that merge for free, so it would bias the raw
    comparison; merging is left to the optimiser for every variant."""
    letters: dict[str, list[float]] = {}
    for t in terms:
        s = t.support
        p = {t.word[q] for q in s}
        if len(s) != 2 or len(p) != 1:
            return None
        letters.setdefault(p.pop(), []).append(t.coeff)
    if sorted(letters) != ["X", "Y", "Z"] or len({len(v) for v in letters.values()}) != 1:
        return None
    return list(zip(letters["X"], letters["Y"], letters["Z"]))


def _compile_run(H: Hamiltonian, run: list[tuple[int, float]]) -> tuple[list, int]:
    """Compile a run of mutually commuting terms sharing one time. Order is
    free inside such a run, so terms are bucketed by support: a complete
    XX/YY/ZZ bond becomes one block, single-qubit terms use the ordinary
    template. A two-qubit bucket that is not a complete bond raises -- this
    builder is only defined for orderings that keep each bond in one run."""
    time = run[0][1]
    buckets: dict[tuple[int, ...], list[Term]] = {}
    for j, _ in run:
        buckets.setdefault(tuple(H.terms[j].support), []).append(H.terms[j])

    ops: list = []
    n_blocks = 0
    for support, terms in buckets.items():
        if len(support) == 1:
            for t in terms:
                ops += simcount_template(t, time)
            continue
        copies = _bond_copies(terms)
        if copies is None:
            raise ValueError(
                f"terms on qubits {support} in one commuting run are not complete "
                f"XX/YY/ZZ bonds ({[t.word for t in terms]}); use the even/odd ordering"
            )
        q0, q1 = support
        for Jx, Jy, Jz in copies:
            ops += canonical_block(q0, q1, time * Jx, time * Jy, time * Jz)
            n_blocks += 1
    return ops, n_blocks


def build_step_blocked(H: Hamiltonian, order: int, d: float) -> tuple[qml.tape.QuantumScript, int]:
    """One product-formula step with bonds compiled as blocks.

    Walks the ordinary schedule and cuts it into maximal runs of consecutive
    entries that share a time and pairwise commute; each run is compiled by
    ``_compile_run``. Returns the tape and the number of blocks emitted.

    Times within a run are compared exactly: they are the same float
    (``formula.second_order`` uses one ``halfd`` / ``d`` per sub-step)."""
    sched = formula.schedule(order, H, d)
    ops: list = []
    n_blocks = 0
    i = 0
    while i < len(sched):
        run = [sched[i]]
        j = i + 1
        while (j < len(sched) and sched[j][1] == sched[i][1]
               and all(_commute(H.terms[sched[j][0]], H.terms[k]) for k, _ in run)):
            run.append(sched[j])
            j += 1
        run_ops, nb = _compile_run(H, run)
        ops += run_ops
        n_blocks += nb
        i = j
    return qml.tape.QuantumScript(ops), n_blocks


# ------------------------------------------------------------------ comparison

VARIANTS = ("xyz", "evenodd", "blocked")


@dataclass
class Counts:
    cnot: int
    rot: int
    total: int


def _counts(tape: qml.tape.QuantumScript) -> Counts:
    from . import resources

    c = resources.count(tape)
    return Counts(c.get("not", 0), resources.rotations(c), resources.total(c))


def _nam(tape: qml.tape.QuantumScript, n: int) -> qml.tape.QuantumScript:
    from .nam import bridge
    from .nam import optimise as nam_optimise

    return bridge.to_tape(nam_optimise(bridge.from_tape(tape, n)).circuit)


@dataclass
class Result:
    n: int
    seed: int
    order: int
    method: str
    r: dict[str, int]
    raw: dict[str, Counts]   # per step, unoptimised
    nam: dict[str, Counts]   # per step, Nam Light on one isolated step


def compare(n: int, seed: int, order: int, method: str = "tte", gatebase: str = "cz") -> Result:
    """Per-step and total costs of the three variants for one recorded instance.

    r comes from ``method`` ('tte' or 'empirical') via ``evenodd.compare``;
    'blocked' reuses the even/odd r because it is the same unitary."""
    from . import evenodd, simcount
    from .circuit import build_step

    [row] = [x for x in evenodd.compare(n, seed, [order], gatebase, empirical=(method == "empirical"))
             if x.method == method]
    r = {"xyz": row.r_xyz, "evenodd": row.r_evenodd, "blocked": row.r_evenodd}

    H_xyz = simcount.run(n, "pf2com", "cz", seed).hamiltonian
    H_eo = evenodd.even_odd(H_xyz)
    t = float(n)
    tapes = {
        "xyz": build_step(H_xyz, order, t / r["xyz"]),
        "evenodd": build_step(H_eo, order, t / r["evenodd"]),
        "blocked": build_step_blocked(H_eo, order, t / r["blocked"])[0],
    }
    # ASSUMPTION: Nam Light applied to ONE isolated step. This misses
    # cancellations across step boundaries, so it overstates the optimised
    # per-step cost for every variant (see nam/README.md, "marginal" vs
    # isolated). Used here only to compare variants with each other. (unconfirmed)
    return Result(n, seed, order, method, r,
                  {v: _counts(tp) for v, tp in tapes.items()},
                  {v: _counts(_nam(tp, n)) for v, tp in tapes.items()})


def render(results: list[Result]) -> str:
    lines = ["=== X-Y-Z vs even/odd vs even/odd with bond blocks  (cz, t = n) ===",
             "per step: CNOT / rotations / total gates.   total = r x per step.", ""]
    hdr = (f"{'n':>3} {'ord':>3} {'r from':>9} {'variant':>8} {'r':>8} "
           f"{'raw/step':>18} {'Nam/step':>18} {'raw CNOT total':>15} {'Nam CNOT total':>15}")
    lines.append(hdr)
    for x in results:
        for v in VARIANTS:
            a, b = x.raw[v], x.nam[v]
            lines.append(
                f"{x.n:>3} {x.order:>3} {x.method:>9} {v:>8} {x.r[v]:>8,} "
                f"{f'{a.cnot}/{a.rot}/{a.total}':>18} {f'{b.cnot}/{b.rot}/{b.total}':>18} "
                f"{a.cnot * x.r[v]:>15,} {b.cnot * x.r[v]:>15,}"
            )
        lines.append("")
    lines.append("Nam/step = Nam Light on one isolated step (no cross-step cancellation).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="pysimcount.blocks", description=__doc__.split("\n\n")[0])
    ap.add_argument("-n", "--sizes", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("-m", "--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--method", default="tte", choices=["tte", "empirical"],
                    help="how r is chosen (empirical = brute force, n <= ~10)")
    ap.add_argument("-g", "--gatebase", default="cz", choices=["cz"],
                    help="only cz: block rotations are not yet routed through synthesis")
    args = ap.parse_args(argv)

    results = [compare(n, s, o, args.method, args.gatebase)
               for n in args.sizes for s in args.seeds for o in args.orders]
    print(render(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
