"""Build one product-formula step as a PennyLane tape.

The full simulation is ``r`` repetitions of this step (simcount boxes it as
subroutine ``S`` and calls it ``r`` times), so resource counts are
``step x r`` and only the step is ever materialised unless an optimisation
pass explicitly needs several consecutive steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml

from . import bounds, formula
from .exponentiate import TEMPLATES, Template
from .hamiltonian import Hamiltonian


@dataclass
class Simulation:
    H: Hamiltonian
    order: int
    bound: str
    t: float
    eps_trotter: float
    r: int
    step: qml.tape.QuantumScript
    template: str

    @property
    def algo(self) -> str:
        return f"pf{self.order}{self.bound}"

    @property
    def n_rz_per_step(self) -> int:
        return sum(op.name == "RZ" for op in self.step.operations)

    @property
    def zrots(self) -> int:
        return self.n_rz_per_step * self.r

    def step_matrix(self) -> np.ndarray:
        return qml.matrix(self.step, wire_order=range(self.H.n_qubits))

    def matrix(self) -> np.ndarray:
        """Unitary of the whole simulation (small systems only)."""
        return np.linalg.matrix_power(self.step_matrix(), self.r)

    def exact_matrix(self) -> np.ndarray:
        from scipy.linalg import expm

        return expm(-1j * self.t * self.H.matrix())

    def trotter_error(self) -> float:
        """Spectral-norm distance between the product formula and exp(-iHt)."""
        return float(np.linalg.norm(self.matrix() - self.exact_matrix(), 2))


def build_step(H: Hamiltonian, order: int, d: float, template: str | Template = "simcount"):
    fn = TEMPLATES[template] if isinstance(template, str) else template
    ops = []
    for j, time in formula.schedule(order, H, d):
        ops += fn(H.terms[j], time)
    return qml.tape.QuantumScript(ops)


def simulate(
    H: Hamiltonian,
    algo: str,
    t: float,
    eps_trotter: float,
    r: int | None = None,
    template: str = "simcount",
) -> Simulation:
    """Choose r from the bound named in ``algo`` (or use the given r) and
    build the corresponding step."""
    order, bound = bounds.parse_algo(algo)
    if r is None:
        r = bounds.steps(algo, H, t, eps_trotter)
    d = t / r  # ``tr = t / fromIntegral r`` in PF.hs
    step = build_step(H, order, d, template)
    return Simulation(H, order, bound, t, eps_trotter, r, step, template)
