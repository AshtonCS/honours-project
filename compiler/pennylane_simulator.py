import pennylane as qp


def build_heisenberg(n_qubits, coupling):
    return qp.spin.heisenberg(
        lattice="chain", # implies 1D chain of qubits
        n_cells=[n_qubits],
        coupling=coupling, #coupling constants, J_x, J_y, J_z in the literature
        boundary_condition=False, # determines whether the chain is open (1 - 2 - ... - N) or closed (1 - 2 - ... - N - 1)
    )


def build_trotter_circuit(H, n_qubits, time, steps, order):
    dev = qp.device("default.qubit", wires=n_qubits)

    @qp.qnode(dev)
    def circuit():
        qp.TrotterProduct(
            H,
            time=time,
            n=steps,
            order=order,
        )

    return circuit


def optimise_circuit(circuit):
    return qp.compile(
        circuit,
        basis_set=["CNOT", "RX", "RY", "RZ"],
        num_passes=2,
    )


def gridsynth_decompose(circuit, epsilon):
    return qp.clifford_t_decomposition(
        circuit,
        epsilon=epsilon,
        method="gridsynth",
    )


def get_gate_counts(circuit):
    resources = qp.specs(circuit)()["resources"]
    gates = resources.gate_types

    t_count = (
        gates.get("T", 0)
        + gates.get("Adjoint(T)", 0)
    )

    cnot_count = gates.get("CNOT", 0)

    clifford_count = sum(
        gates.get(gate, 0)
        for gate in [
            "Identity",
            "PauliX",
            "PauliY",
            "PauliZ",
            "SX",
            "S",
            "Adjoint(S)",
            "Hadamard",
            "CNOT",
            "CY",
            "CZ",
            "SWAP",
            "ISWAP",
        ]
    )

    return {
        "Total Gates": resources.num_gates,
        "Clifford+T Gates": clifford_count + t_count,
        "Clifford Gates": clifford_count,
        "CNOTs": cnot_count,
        "T Gates": t_count,
        "Gate Breakdown": gates,
    }


# --------------------------------------------------
# Parameters
# --------------------------------------------------

n_qubits = 4
coupling = [1.0, 1.0, 1.0]
time = 1.0
steps = 2
order = 2
epsilon = 1e-4


# --------------------------------------------------
# Build and compile Heisenberg Hamiltonian
# --------------------------------------------------

H = build_heisenberg(
    n_qubits,
    coupling,
)

circuit = build_trotter_circuit(
    H,
    n_qubits,
    time,
    steps,
    order,
)

optimised = optimise_circuit(circuit)

final_circuit = gridsynth_decompose(
    optimised,
    epsilon,
)


# --------------------------------------------------
# Output
# --------------------------------------------------

print("Hamiltonian:")
print(H)

print("\nOptimised circuit:")
print(qp.draw(optimised)())

print("\nClifford+T circuit:")
print(qp.draw(final_circuit)())

counts = get_gate_counts(final_circuit)

print("\nGate counts:")
for name, count in counts.items():
    print(f"{name}: {count}")