"""Comparison harness against the published before/after circuits.

``simcount/samples`` ships, for each algorithm and system size, the circuit
before optimisation and the circuit after optimisation with the Nam et al.
optimiser. The pre-optimisation circuit is the *input* our own pipeline
reproduces exactly, and the post-optimisation circuit is the *target*. That
makes a 50-case validation set for an optimiser implementation:

    algorithms  pf4ana, pf4min, pf4com, pf4emp, pf6emp
    sizes       13, 16, 20, 25, 32, 40, 50, 63, 79, 100
    gate sets   cz (Clifford+Rz), ct (Clifford+T)

Both circuits are structured around a repeated core: the pre-optimisation file
calls one Trotter-step subroutine r times, and the post-optimisation file is
``L . C^m . R``. Comparison is therefore done per core block, which is about a
thousand gates rather than tens of millions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import quipper
from .gates import Circuit
from .optimiser import LIGHT, OptimiseResult, optimise

# simcount's published sample circuits (njross/simcount, samples/). Set SIMCOUNT_SAMPLES to
# point elsewhere; the default is where they live on the author's machine.
SAMPLES_ROOT = Path(os.environ.get("SIMCOUNT_SAMPLES", Path.home() / "Documents/simcount/samples"))

ALGORITHMS = ("pf4ana", "pf4min", "pf4com", "pf4emp", "pf6emp")
SIZES = (13, 16, 20, 25, 32, 40, 50, 63, 79, 100)
GATE_SETS = ("cz", "ct")


def path_for(algo: str, size: int, stage: str, gate_set: str = "cz") -> Path:
    """Path to a sample circuit. ``stage`` is ``preoptim`` or ``postoptim``."""
    if stage not in ("preoptim", "postoptim"):
        raise ValueError(f"stage must be preoptim or postoptim, got {stage!r}")
    return SAMPLES_ROOT / algo / f"{algo}_{stage}_{gate_set}_{size}"


@dataclass
class SamplePair:
    """One (algorithm, size, gate set) case: their input and their output."""

    algo: str
    size: int
    gate_set: str
    pre: quipper.QuipperFile
    post: quipper.QuipperFile

    @property
    def pre_core(self) -> Circuit:
        name, _ = self.pre.core()
        return self.pre.expand(name)

    @property
    def post_core(self) -> Circuit:
        name, _ = self.post.core()
        return self.post.expand(name)

    @property
    def repeats(self) -> tuple[int, int]:
        return self.pre.core()[1], self.post.core()[1]

    def target_factors(self) -> dict[str, float]:
        """Their per-block reduction factors, keyed by gate name plus 'total'."""
        pre, post = self.pre_core.counts(), self.post_core.counts()
        out: dict[str, float] = {}
        for name in sorted(set(pre) | set(post)):
            p, q = pre.get(name, 0), post.get(name, 0)
            out[name] = float("inf") if q == 0 else p / q
        out["total"] = sum(pre.values()) / sum(post.values())
        return out


def load(algo: str, size: int, gate_set: str = "cz") -> SamplePair:
    """Load the before/after pair for one case."""
    return SamplePair(
        algo,
        size,
        gate_set,
        quipper.read(path_for(algo, size, "preoptim", gate_set)),
        quipper.read(path_for(algo, size, "postoptim", gate_set)),
    )


def available(gate_set: str = "cz") -> list[tuple[str, int]]:
    """Cases present on disk."""
    out: list[tuple[str, int]] = []
    for algo in ALGORITHMS:
        for size in SIZES:
            if (
                path_for(algo, size, "preoptim", gate_set).exists()
                and path_for(algo, size, "postoptim", gate_set).exists()
            ):
                out.append((algo, size))
    return out


@dataclass
class Comparison:
    """Our optimiser's result on a sample, beside their published result."""

    pair: SamplePair
    result: OptimiseResult

    def rows(self) -> list[tuple[str, int, int, int]]:
        """``(gate, their_pre, our_post, their_post)`` per gate, then total."""
        pre = self.pair.pre_core.counts()
        theirs = self.pair.post_core.counts()
        ours = self.result.after
        rows = [
            (name, pre.get(name, 0), ours.get(name, 0), theirs.get(name, 0))
            for name in sorted(set(pre) | set(ours) | set(theirs))
        ]
        rows.append(
            ("total", sum(pre.values()), sum(ours.values()), sum(theirs.values()))
        )
        return rows

    def render(self) -> str:
        p = self.pair
        head = (
            f"=== {p.algo} n={p.size} -g {p.gate_set} "
            f"(core block; their r={p.repeats[0]}, m={p.repeats[1]}) ==="
        )
        skipped = (
            f"  [routines skipped: {', '.join(map(str, self.result.skipped))}]"
            if self.result.skipped
            else ""
        )
        lines = [
            head,
            f"rounds={self.result.rounds}{skipped}",
            f"{'gate':>6}  {'pre':>8}  {'ours':>8}  {'theirs':>8}  {'our x':>7}  {'their x':>8}",
        ]
        for name, pre, ours, theirs in self.rows():
            fo = "inf" if ours == 0 else f"{pre / ours:.2f}"
            ft = "inf" if theirs == 0 else f"{pre / theirs:.2f}"
            lines.append(
                f"{name:>6}  {pre:>8,}  {ours:>8,}  {theirs:>8,}  {fo:>7}  {ft:>8}"
            )
        return "\n".join(lines)


def compare(
    algo: str = "pf4com",
    size: int = 13,
    gate_set: str = "cz",
    sequence: tuple[int, ...] = LIGHT,
    skip_unimplemented: bool = True,
) -> Comparison:
    """Optimise their pre-optimisation core block and compare with their output."""
    pair = load(algo, size, gate_set)
    result = optimise(
        pair.pre_core, sequence=sequence, skip_unimplemented=skip_unimplemented
    )
    return Comparison(pair, result)
