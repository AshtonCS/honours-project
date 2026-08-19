# Test Suite for Pauli Hamiltonian Compiler

import pytest

from pauli_simulator import PauliTerm, Hamiltonian, HamiltonianSimulator


# ============================================================
# Helper functions
# ============================================================

def rz_gates(output):
    """Return only RZ gates from a compiled circuit."""
    return [gate for gate in output if gate[0] == "RZ"]


def gate_names(output):
    """Return gate names in order."""
    return [gate[0] for gate in output]


# ============================================================
# 1. Basic PauliTerm / Compiler tests
# ============================================================

def test_single_z_pauli():
    """A single Z term should require only an RZ rotation."""

    hamiltonian = Hamiltonian(
        [PauliTerm(1, [("Z", [0])])],
        num_qubits=1
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    output, counts = hamiltonian_simulator.standard_simulation()

    assert output == [
        ("RZ", [0], 2)
    ]

    assert counts["RZ"] == 1
    assert counts["CNOT"] == 0
    assert counts["total"] == 1


def test_single_x_pauli():
    """X should be converted to Z basis using H gates."""

    hamiltonian = Hamiltonian(
        [PauliTerm(1, [("X", [0])])],
        num_qubits=1
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    output, counts = hamiltonian_simulator.standard_simulation()

    assert gate_names(output) == ["H", "RZ", "H"]

    assert rz_gates(output) == [
        ("RZ", [0], 2)
    ]

    assert counts["RZ"] == 1
    assert counts["total"] == 3


def test_single_y_pauli():
    """Y should be converted using S and H gates."""

    hamiltonian = Hamiltonian(
        [PauliTerm(1, [("Y", [0])])],
        num_qubits=1
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    output, counts = hamiltonian_simulator.standard_simulation()

    assert gate_names(output) == ["S", "H", "RZ", "H", "S"]

    assert rz_gates(output) == [
        ("RZ", [0], 2)
    ]

    assert counts["RZ"] == 1


# ============================================================
# 2. Standard evolution
# ============================================================

def test_standard_two_term_hamiltonian():
    """
    H = Z0 + Z1

    Standard evolution should compile each Hamiltonian term once.
    """

    hamiltonian = Hamiltonian(
        [
            PauliTerm(1, [("Z", [0])]),
            PauliTerm(1, [("Z", [1])])
        ],
        num_qubits=2
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    output, counts = hamiltonian_simulator.standard_simulation()

    assert rz_gates(output) == [
        ("RZ", [0], 2),
        ("RZ", [1], 2)
    ]

    assert counts["RZ"] == 2
    assert counts["CNOT"] == 0
    assert counts["total"] == 2


def test_standard_coefficient_scales_angle():
    """
    H = 2 Z0

    The coefficient should double the rotation angle.
    """

    hamiltonian = Hamiltonian(
        [PauliTerm(2, [("Z", [0])])],
        num_qubits=1
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    output, _ = hamiltonian_simulator.standard_simulation()

    assert rz_gates(output) == [
        ("RZ", [0], 4)
    ]


def test_standard_time_scales_angle():
    """
    Increasing simulation time should linearly increase the angle.
    """

    hamiltonian = Hamiltonian(
        [PauliTerm(1, [("Z", [0])])],
        num_qubits=1
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=3)

    output, _ = hamiltonian_simulator.standard_simulation()

    assert rz_gates(output) == [
        ("RZ", [0], 6)
    ]


# ============================================================
# 3. First-order Trotter
# ============================================================

def test_first_order_single_step_equals_standard():
    """
    First-order Trotter with one step should be identical
    to standard product-form evolution.
    """

    hamiltonian = Hamiltonian(
        [
            PauliTerm(1, [("Z", [0])]),
            PauliTerm(2, [("Z", [1])])
        ],
        num_qubits=2
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    standard_output, standard_counts = (
        hamiltonian_simulator.standard_simulation()
    )

    trotter_output, trotter_counts = (
        hamiltonian_simulator.first_order_trotter_simulation(1)
    )

    assert trotter_output == standard_output
    assert trotter_counts == standard_counts


def test_first_order_two_steps():
    """
    For two commuting single-qubit terms:

        [exp(-iH1 dt) exp(-iH2 dt)]^2

    should combine to the expected total rotations.
    """

    hamiltonian = Hamiltonian(
        [
            PauliTerm(1, [("Z", [0])]),
            PauliTerm(2, [("Z", [1])])
        ],
        num_qubits=2
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    output, counts = hamiltonian_simulator.first_order_trotter_simulation(2)

    assert rz_gates(output) == [
        ("RZ", [0], 1),
        ("RZ", [1], 2),
        ("RZ", [0], 1),
        ("RZ", [1], 2)
    ]

    assert counts["RZ"] == 4


def test_first_order_noncommuting_terms():
    """
    X0 and Z0 do not commute.

    The circuit should contain separate basis changes and
    rotations rather than being reduced to a single rotation.
    """

    hamiltonian = Hamiltonian(
        [
            PauliTerm(1, [("X", [0])]),
            PauliTerm(1, [("Z", [0])])
        ],
        num_qubits=1
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    output, counts = hamiltonian_simulator.first_order_trotter_simulation(2)

    assert counts["RZ"] == 4

    # X requires H-RZ-H for each occurrence.
    assert counts["total"] > 4


# ============================================================
# 4. Second-order Trotter
# ============================================================

def test_second_order_single_step_two_terms():
    """
    For H = H1 + H2:

        exp(-iH1 dt/2)
        exp(-iH2 dt)
        exp(-iH1 dt/2)

    """

    hamiltonian = Hamiltonian(
        [
            PauliTerm(1, [("Z", [0])]),
            PauliTerm(2, [("Z", [1])])
        ],
        num_qubits=2
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    output, counts = hamiltonian_simulator.second_order_trotter_simulation(1)

    assert rz_gates(output) == [
        ("RZ", [0], 1),
        ("RZ", [1], 4),
        ("RZ", [0], 1)
    ]

    assert counts["RZ"] == 3


def test_second_order_two_steps():
    """
    For two Trotter steps:

        H1/2 H2 H1/2 H1/2 H2 H1/2

    The middle H1/2 terms should merge into H1.
    """

    hamiltonian = Hamiltonian(
        [
            PauliTerm(1, [("Z", [0])]),
            PauliTerm(2, [("Z", [1])])
        ],
        num_qubits=2
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    output, counts = hamiltonian_simulator.second_order_trotter_simulation(2)

    print(rz_gates(output))

    assert rz_gates(output) == [
        ("RZ", [0], 0.5),
        ("RZ", [1], 2),
        ("RZ", [0], 1),
        ("RZ", [1], 2),
        ("RZ", [0], 0.5)
    ]

    assert counts["RZ"] == 5


def test_second_order_three_terms():
    """
    For three terms:

        H1/2 H2/2 H3 H2/2 H1/2
    """

    hamiltonian = Hamiltonian(
        [
            PauliTerm(1, [("Z", [0])]),
            PauliTerm(2, [("Z", [1])]),
            PauliTerm(3, [("Z", [2])])
        ],
        num_qubits=3
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    output, counts = hamiltonian_simulator.second_order_trotter_simulation(1)

    assert rz_gates(output) == [
        ("RZ", [0], 1),
        ("RZ", [1], 2),
        ("RZ", [2], 6),
        ("RZ", [1], 2),
        ("RZ", [0], 1)
    ]

    assert counts["RZ"] == 5


def test_second_order_is_symmetric():
    """
    A second-order Trotter step must be palindromic at the
    Hamiltonian-term level.
    """

    hamiltonian = Hamiltonian(
        [
            PauliTerm(1, [("Z", [0])]),
            PauliTerm(2, [("Z", [1])]),
            PauliTerm(3, [("Z", [2])])
        ],
        num_qubits=3
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    # num_steps=1 avoids inter-step optimisation.
    output, _ = hamiltonian_simulator.second_order_trotter_simulation(1)

    rz = rz_gates(output)

    angles = [gate[2] for gate in rz]

    assert angles == [
        1, 2, 6, 2, 1
    ]

    assert angles == angles[::-1]


# ============================================================
# 5. Optimisation tests
# ============================================================

def test_rz_merging():
    """
    RZ(a) RZ(b) should become RZ(a+b).
    """

    hamiltonian = Hamiltonian(
        [PauliTerm(1, [("Z", [0])])],
        num_qubits=1
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    output = [
        ("RZ", [0], 1),
        ("RZ", [0], 2)
    ]

    optimised = hamiltonian_simulator.optimise(output)

    assert optimised == [
        ("RZ", [0], 3)
    ]


def test_inverse_gates_cancel():
    """
    H H = I
    """

    hamiltonian = Hamiltonian(
        [PauliTerm(1, [("Z", [0])])],
        num_qubits=1
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    output = [
        ("H", [0], None),
        ("H", [0], None)
    ]

    optimised = hamiltonian_simulator.optimise(output)

    assert optimised == []


def test_inverse_parameterised_gates_cancel():
    """
    S(-1) S(+1) should cancel under the current gate representation.
    """

    hamiltonian = Hamiltonian(
        [PauliTerm(1, [("Z", [0])])],
        num_qubits=1
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    output = [
        ("S", [0], -1),
        ("S", [0], 1)
    ]

    optimised = hamiltonian_simulator.optimise(output)

    assert optimised == []


# ============================================================
# 6. Gate-count scaling tests
# ============================================================

def test_first_order_rz_scaling():
    """
    Before cross-step optimisation, first-order Trotterisation
    uses one RZ per Hamiltonian term per Trotter step.

    For two terms and n steps:

        RZ count = 2n
    """

    hamiltonian = Hamiltonian(
        [
            PauliTerm(1, [("Z", [0])]),
            PauliTerm(1, [("Z", [1])])
        ],
        num_qubits=2
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    for n in [1, 2, 3, 5]:
        output, counts = hamiltonian_simulator.first_order_trotter_simulation(n)

        assert counts["RZ"] == 2 * n


def test_second_order_rz_scaling():
    """
    For two terms and n second-order Trotter steps,
    the optimised circuit should contain:

        2n + 1

    RZ gates.

    This is because the first term has two half steps at
    the boundaries and full steps internally.
    """

    hamiltonian = Hamiltonian(
        [
            PauliTerm(1, [("Z", [0])]),
            PauliTerm(1, [("Z", [1])])
        ],
        num_qubits=2
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    for n in [1, 2, 3, 5]:
        output, counts = hamiltonian_simulator.second_order_trotter_simulation(n)

        assert counts["RZ"] == 2 * n + 1


def test_second_order_uses_more_gates_than_first_order():
    """
    For non-commuting terms, second-order Trotterisation should
    generally require more gates than first-order Trotterisation
    at the same number of steps.
    """

    hamiltonian = Hamiltonian(
        [
            PauliTerm(1, [("X", [0])]),
            PauliTerm(1, [("Z", [0])])
        ],
        num_qubits=1
    )

    hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t=1)

    first_output, first_counts = (
        hamiltonian_simulator.first_order_trotter_simulation(3)
    )

    second_output, second_counts = (
        hamiltonian_simulator.second_order_trotter_simulation(3)
    )

    assert second_counts["total"] > first_counts["total"]