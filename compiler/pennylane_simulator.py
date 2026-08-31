import pennylane as qp
import time


def build_heisenberg(n_qubits, coupling):
    return qp.spin.heisenberg(
        lattice="chain", # implies 1D chain of qubits
        n_cells=[n_qubits],
        coupling=coupling, #coupling constants, J_x, J_y, J_z in the literature
        boundary_condition=False, # False: 1-2-...-N; True: 1-2-...-N-1
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

def build_trotter_step(H, time, steps, order):
    def circuit():
        qp.TrotterProduct(
            H,
            time=time/steps,
            n=1,
            order=order,
        )

    return circuit

def repeat_operations(operations, n_qubits, steps):

    dev = qp.device("default.qubit", wires=n_qubits)

    @qp.qnode(dev)
    def circuit():

        for _ in range(steps):
            for op in operations:
                qp.apply(op)

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

def trotter_error(H, time, steps, order):
    trotter = qp.TrotterProduct(
        H,
        time=time,
        n=steps,
        order=order,
    )

    return float(
        trotter.error(method="commutator-bound").error
    )


def find_trotter_steps(H, time, order, epsilon):
    lower = 0
    upper = 1

    while trotter_error(H, time, upper, order) > epsilon:
        lower = upper
        upper *= 2

    while lower + 1 < upper:
        middle = (lower + upper) // 2

        if trotter_error(H, time, middle, order) <= epsilon:
            upper = middle
        else:
            lower = middle

    return upper

def build_full_circuit(n_qubits, coupling, t, order, epsilon):
    epsilon_synthesis = epsilon / 2
    epsilon_trotter = epsilon / 2

    H = build_heisenberg(
        n_qubits,
        coupling,
    )

    steps = find_trotter_steps(
        H,
        time=t,
        order=order,
        epsilon=epsilon_trotter,
    )

    steps = 10

    print("Minimum Trotter steps:", steps)

    circuit = build_trotter_circuit(
        H,
        n_qubits,
        t,
        steps,
        order,
    )

    optimised = optimise_circuit(circuit)

    print("Hamiltonian:")
    print(H)

    counts = get_gate_counts(optimised)

    print("\nGate counts BEFORE GridSynth:")
    for name, count in counts.items():
        print(f"{name}: {count}")

    final_circuit = gridsynth_decompose(
        optimised,
        epsilon_synthesis,
    )

    counts = get_gate_counts(final_circuit)

    print("\nGate counts AFTER GridSynth:")
    for name, count in counts.items():
        print(f"{name}: {count}")

    return final_circuit, counts

def build_stepwise_circuit(n_qubits, coupling, t, order, epsilon):
    epsilon_synthesis = epsilon / 2
    epsilon_trotter = epsilon / 2

    H = build_heisenberg(
        n_qubits,
        coupling,
    )

    steps = find_trotter_steps(
        H,
        time=t,
        order=order,
        epsilon=epsilon_trotter,
    )

    steps = 10

    print("Minimum Trotter steps:", steps)

    trotter_step = build_trotter_step(
        H,
        t,
        steps,
        order
    )

    optimised_step = optimise_circuit(trotter_step)

    step_qscript = qp.tape.make_qscript(optimised_step)()

    step_operations = step_qscript.operations

    circuit = repeat_operations(
        step_operations,
        n_qubits,
        steps
    )

    print("Hamiltonian:")
    print(H)

    counts = get_gate_counts(circuit)

    print("\nGate counts BEFORE GridSynth:")
    for name, count in counts.items():
        print(f"{name}: {count}")

    final_circuit = gridsynth_decompose(
        circuit,
        epsilon_synthesis
    )

    counts = get_gate_counts(final_circuit)

    print("\nGate counts AFTER GridSynth:")
    for name, count in counts.items():
        print(f"{name}: {count}")

    return final_circuit, counts

n_qubits = 4
coupling = [1.0, 2.0, 1.5]
t = 1.0
order = 2
epsilon = 1e-4

full_circuit, gate_counts = build_full_circuit(
    n_qubits,
    coupling,
    t,
    order,
    epsilon
)

stepwise_circuit, gate_counts = build_stepwise_circuit(
    n_qubits,
    coupling,
    t,
    order,
    epsilon
)

