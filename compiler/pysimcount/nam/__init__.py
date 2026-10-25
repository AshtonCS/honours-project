"""Reimplementation of the Nam et al. circuit optimiser (arXiv:1710.07345).

Routines 1-3 and the Light driver are implemented; Routine 4
(phase-polynomial rotation merging) is a documented stub, and Routine 5 is
not implementable from the publication. See ``README.md`` in this directory.
"""

from .gates import Circuit, Gate
from .optimiser import HEAVY, LIGHT, optimise
from .routines import (
    routine1_hadamard_reduction,
    routine2_single_qubit_cancellation,
    routine3_two_qubit_cancellation,
    routine4_rotation_merging,
)

__all__ = [
    "Circuit", "Gate", "optimise", "LIGHT", "HEAVY",
    "routine1_hadamard_reduction", "routine2_single_qubit_cancellation",
    "routine3_two_qubit_cancellation", "routine4_rotation_merging",
]
