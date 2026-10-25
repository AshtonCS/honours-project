"""Product formulae as symbolic schedules.

A formula turns a time step ``d`` into a list of ``(term_index, time)``
pairs: "apply exp(-i * time * H_term)" in that order. Nothing downstream
depends on which formula produced the schedule, so adding a new formula is
just adding a new schedule function here.

The schedules mirror ``simulate_1st`` / ``simulate_2kth`` in simcount's
``PF.hs``, including the arithmetic order, so rotation angles agree bit for
bit with the Haskell program.
"""

from __future__ import annotations

from .hamiltonian import Hamiltonian

Schedule = list[tuple[int, float]]


def first_order(H: Hamiltonian, d: float) -> Schedule:
    """Lie-Trotter: every term once, at the full step."""
    return [(j, d) for j in range(H.n_terms)]


def second_order(H: Hamiltonian, d: float) -> Schedule:
    """simcount's ``sss2``: the first ``H.split`` terms at d/2, the remaining
    (mutually commuting) terms at d, then the first block reversed at d/2.

    With ``split == n_terms`` this is the textbook symmetric formula in
    which the last term appears twice consecutively at d/2 (PennyLane's
    ``TrotterProduct(order=2)`` behaviour)."""
    halfd = d / 2
    head = list(range(H.split))
    tail = list(range(H.split, H.n_terms))
    return (
        [(j, halfd) for j in head]
        + [(j, d) for j in tail]
        + [(j, halfd) for j in reversed(head)]
    )


def suzuki(order: int, H: Hamiltonian, d: float) -> Schedule:
    """Suzuki's recursive construction S_{2k} (simcount's ``s2k``)."""
    if order % 2 or order < 2:
        raise ValueError("Suzuki formulae exist for even orders >= 2")
    k = order // 2
    if k == 1:
        return second_order(H, d)
    pk = (4 - (4 ** (1 / (2 * k - 1)))) ** (-1)
    a = suzuki(order - 2, H, pk * d)
    b = suzuki(order - 2, H, (1 - 4 * pk) * d)
    return a + a + b + a + a


def schedule(order: int, H: Hamiltonian, d: float) -> Schedule:
    if order == 1:
        return first_order(H, d)
    return suzuki(order, H, d)
