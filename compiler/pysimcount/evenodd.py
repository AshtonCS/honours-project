"""Even/odd bond grouping for the periodic Heisenberg chain -- the "cheap test".

simcount orders the Heisenberg terms Pauli-major (all XX, all YY, all ZZ,
then fields), which the product formula sees as three commuting groups
XX | YY | ZZ+Z (the 'X-Y-Z ordering' of Childs et al. 2021). This module
reorders the *same* terms into

    even bonds (XX+YY+ZZ) | odd bonds (XX+YY+ZZ) | fields (Z)

and compares the Trotter number r of the two orderings. Nothing is
recompiled differently: every term is still exponentiated on its own, so
any difference in r comes from the grouping alone.

Why three groups and not two: a field h_j Z_j anticommutes with X_j X_{j+1}
and Y_j Y_{j+1} (one clashing site), so fields cannot join a bond group
without compiling each bond as an exact two-qubit block (not done here).

Bond parity: bond (i, i+1) has index i; the periodic wrap bond, stored by
``hamiltonian.heisenberg`` as the word on qubits {0, n-1}, is bond n-1.
For even n the wrap bond is odd, so each group has disjoint supports.
For odd n it would collide with bond 0, hence the even-n restriction.

Only the two order-sensitive step counters are compared:
``tte`` (Childs et al. 2021, nested commutators) and brute-force
``empirical.steps``. ``ana``/``min``/``com``/``emp`` depend only on n,
the term count and the largest coefficient, so they cannot distinguish
the orderings.

    python -m pysimcount.evenodd -n 4 6 8 --orders 1 2 4 --empirical
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass

from .hamiltonian import Hamiltonian, Term


def bond_index(term: Term, n: int) -> int | None:
    """Bond index of a nearest-neighbour two-qubit term, or None for a
    single-qubit term. Raises for anything else (no silent fallback).

    Bond (i, i+1) -> i. The periodic wrap bond on qubits {0, n-1} -> n-1."""
    s = term.support
    if len(s) == 1:
        return None
    if len(s) != 2:
        raise ValueError(f"term {term.word!r} is not a one- or two-site term")
    a, b = s
    if b == a + 1:
        return a
    if (a, b) == (0, n - 1):
        return n - 1
    raise ValueError(f"term {term.word!r} is not a nearest-neighbour bond on a chain of {n}")


def even_odd(H: Hamiltonian) -> Hamiltonian:
    """Return H with its terms reordered as even bonds | odd bonds | fields.

    Same terms, same coefficients, so the same operator H. Within each group
    the original (Pauli-major) order is kept; terms within a group commute,
    so that order does not affect the Trotter error.

    ``split`` is set to the number of bond terms, so the second-order
    formula applies the bonds at d/2 on either side and the fields once at d
    in the middle -- the same position fields occupy in simcount's ordering.

    Requires even n >= 4 with a periodic chain (n = 2 has no odd bonds)."""
    n = H.n_qubits
    if n < 4 or n % 2:
        raise ValueError(f"even/odd grouping needs even n >= 4 on a periodic chain, got n={n}")

    even: list[Term] = []
    odd: list[Term] = []
    fields: list[Term] = []
    for term in H.terms:
        k = bond_index(term, n)
        if k is None:
            fields.append(term)
        elif k % 2 == 0:
            even.append(term)
        else:
            odd.append(term)

    # A periodic chain of n sites has n bonds, n/2 of each parity, 3 terms each.
    if len(even) != 3 * n // 2 or len(odd) != 3 * n // 2:
        raise ValueError(
            f"expected {3 * n // 2} terms per bond parity on a periodic chain, "
            f"got even={len(even)}, odd={len(odd)}"
        )

    # ASSUMPTION: even bonds outermost, odd next, fields innermost (applied
    # once at d in S2's middle, as in simcount's ordering). Swapping even/odd,
    # or putting fields outermost, is a different formula with a different
    # error and may change r. (unconfirmed)
    return Hamiltonian(even + odd + fields, split=len(even) + len(odd), name=f"{H.name}-evenodd")


# ------------------------------------------------------------------ comparison

@dataclass
class Row:
    n: int
    seed: int
    order: int
    method: str  # "tte" or "empirical"
    r_xyz: int
    r_evenodd: int
    gates_xyz: int  # r * gates per step (simcount template, cz, unoptimised)
    gates_evenodd: int


def _total_gates(H: Hamiltonian, order: int, t: float, eps: float, r: int) -> int:
    from . import resources
    from .circuit import simulate

    sim = simulate(H, f"pf{order}com", t, eps, r=r)  # bound name unused when r is given
    return resources.total(resources.count(sim.step)) * r


def compare(n: int, seed: int, orders: list[int], gatebase: str = "cz",
            empirical: bool = False) -> list[Row]:
    """r for the X-Y-Z and even/odd orderings of one recorded instance.

    The Hamiltonian is simcount's recorded instance for (n, seed), read from
    the committed fixture. t = n and eps_trotter come from the gatebase,
    exactly as in ``compare``/``bounds``."""
    from . import compare as cmp
    from . import empirical as emp
    from . import simcount, tte

    H_xyz = simcount.run(n, "pf2com", "cz", seed).hamiltonian  # any algo prints the same H
    H_eo = even_odd(H_xyz)
    t = float(n)  # Main.hs: t = size
    eps = cmp.GATEBASES[gatebase][0]

    methods = ["tte"] + (["empirical"] if empirical else [])
    rows = []
    for order in orders:
        for method in methods:
            fn = tte.steps if method == "tte" else emp.steps
            r_a = fn(order, H_xyz, t, eps)
            r_b = fn(order, H_eo, t, eps)
            rows.append(Row(n, seed, order, method, r_a, r_b,
                            _total_gates(H_xyz, order, t, eps, r_a),
                            _total_gates(H_eo, order, t, eps, r_b)))
    return rows


def render(rows: list[Row], gatebase: str) -> str:
    from . import compare as cmp

    eps = cmp.GATEBASES[gatebase][0]
    head = (f"=== X-Y-Z vs even/odd grouping  (t = n, eps_trotter = {eps:g}, "
            f"gates = r x per-step, unoptimised) ===")
    cols = f"{'n':>3} {'seed':>4} {'order':>5} {'method':>10} {'r xyz':>10} {'r evenodd':>10} " \
           f"{'r ratio':>8} {'gates xyz':>12} {'gates evenodd':>14}"
    lines = [head, cols]
    for x in rows:
        lines.append(
            f"{x.n:>3} {x.seed:>4} {x.order:>5} {x.method:>10} {x.r_xyz:>10,} {x.r_evenodd:>10,} "
            f"{x.r_evenodd / x.r_xyz:>8.3f} {x.gates_xyz:>12,} {x.gates_evenodd:>14,}"
        )
    lines.append("\nr ratio = r_evenodd / r_xyz  (< 1 means even/odd needs fewer steps).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="pysimcount.evenodd", description=__doc__.split("\n\n")[0])
    ap.add_argument("-n", "--sizes", type=int, nargs="+", default=[4, 6, 8],
                    help="even system sizes with a recorded fixture (4, 6, 8, 10, 20)")
    ap.add_argument("-m", "--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 4],
                    help="tte supports 1, 2, 4")
    ap.add_argument("-g", "--gatebase", default="cz", choices=["cz", "cz2ct", "ct"])
    ap.add_argument("--empirical", action="store_true",
                    help="also brute-force the true minimal r (dense matrices; n <= ~10)")
    args = ap.parse_args(argv)

    rows: list[Row] = []
    for n in args.sizes:
        for seed in args.seeds:
            rows += compare(n, seed, args.orders, args.gatebase, args.empirical)
    print(render(rows, args.gatebase))
    return 0


if __name__ == "__main__":
    sys.exit(main())
