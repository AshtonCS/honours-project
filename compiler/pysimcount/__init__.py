"""PennyLane re-implementation of njross/simcount's product-formula pipeline.

Stages (each independent of the others):

    hamiltonian  ->  bounds (r)  ->  formula (schedule)  ->  exponentiate (gates)
                 ->  resources (step x r)  ->  synthesis (Rz -> Clifford+T)
                 ->  optimise (opt-in)

``python -m pysimcount compare -n 4 -a pf2com -g cz`` prints the two
implementations side by side.
"""

from .bounds import steps
from .circuit import build_step, simulate
from .hamiltonian import Hamiltonian, Term, from_pennylane, heisenberg, parse_simcount

__all__ = [
    "Hamiltonian", "Term", "heisenberg", "parse_simcount", "from_pennylane",
    "steps", "simulate", "build_step",
]
