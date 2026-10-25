"""The Light and Heavy optimisers of Nam et al. (arXiv:1710.07345, section 4.5).

Both are a fixed sequence of subroutines, repeated until no further reduction
is achieved:

    Light:  1, 3, 2, 3, 1, 2, 4, 3, 2
    Heavy:  1, 3, 2, 3, 1, 2, 5

Heavy is not implementable from the publication -- Routine 5 depends on
external rewrite-rule library files that were never released -- and the paper
reports it gives no improvement over Light for product-formula circuits, so
Light is the target here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .gates import Circuit
from .routines import ROUTINES

LIGHT: tuple[int, ...] = (1, 3, 2, 3, 1, 2, 4, 3, 2)
HEAVY: tuple[int, ...] = (1, 3, 2, 3, 1, 2, 5)


@dataclass
class OptimiseResult:
    """Outcome of an optimiser run."""

    circuit: Circuit
    rounds: int
    before: dict[str, int]
    after: dict[str, int]
    skipped: tuple[int, ...] = ()
    history: list[dict[str, int]] = field(default_factory=list)

    def factor(self, gate: str | None = None) -> float:
        """Reduction factor for one gate name, or for the total."""
        pre = sum(self.before.values()) if gate is None else self.before.get(gate, 0)
        post = sum(self.after.values()) if gate is None else self.after.get(gate, 0)
        return float("inf") if post == 0 else pre / post


def optimise(
    circuit: Circuit,
    sequence: tuple[int, ...] = LIGHT,
    skip_unimplemented: bool = False,
    max_rounds: int = 32,
) -> OptimiseResult:
    """Apply ``sequence`` repeatedly until the gate count stops falling.

    Args:
        circuit: the circuit to optimise; not mutated.
        sequence: subroutine numbers, in order. Defaults to :data:`LIGHT`.
        skip_unimplemented: if True, silently skip subroutines that raise
            ``NotImplementedError`` (currently Routine 4 and Routine 5). The
            default is False so a missing subroutine is never applied by
            accident -- with Routine 4 absent the Rz count barely moves, and
            that must not look like a real result.
        max_rounds: safety bound on the fixpoint loop.

    Raises:
        NotImplementedError: if the sequence names an unimplemented subroutine
            and ``skip_unimplemented`` is False.
    """
    before = circuit.counts()
    current = circuit.copy()
    skipped: list[int] = []
    history: list[dict[str, int]] = []

    rounds = 0
    for _ in range(max_rounds):
        size_before_round = len(current)
        for number in sequence:
            routine = ROUTINES[number]
            try:
                current = routine(current)
            except NotImplementedError:
                if not skip_unimplemented:
                    raise
                if number not in skipped:
                    skipped.append(number)
        rounds += 1
        history.append(current.counts())
        if len(current) >= size_before_round:
            break

    return OptimiseResult(
        circuit=current,
        rounds=rounds,
        before=before,
        after=current.counts(),
        skipped=tuple(skipped),
        history=history,
    )
