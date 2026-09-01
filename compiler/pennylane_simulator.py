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

def determine_trotter_steps(H, time, order, epsilon, fixed_steps=None):
    if fixed_steps is not None:
        return fixed_steps

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

def print_time(label, start):
    print(f"{label}: {time.perf_counter() - start:.3f}s")

def build_full_circuit(n_qubits, coupling, t, order, epsilon, fixed_steps=None):

    start = time.perf_counter()

    epsilon_synthesis = epsilon / 2
    epsilon_trotter = epsilon / 2

    H = build_heisenberg(n_qubits, coupling)

    print("\n=== Full-Circuit Optimisation ===")
    print("Hamiltonian:")
    print(H)
    print_time("Hamiltonian construction", start)

    start = time.perf_counter()

    steps = determine_trotter_steps(
        H,
        time=t,
        order=order,
        epsilon=epsilon_trotter,
        fixed_steps=fixed_steps
    )

    print("Trotter steps used:", steps)
    print_time("Trotter-step calculation", start)

    start = time.perf_counter()

    circuit = build_trotter_circuit(
        H,
        n_qubits,
        t,
        steps,
        order
    )
    print_time("Circuit construction", start)

    start = time.perf_counter()

    optimised = optimise_circuit(circuit)
    print_time("Optimisation", start)

    start = time.perf_counter()

    counts = get_gate_counts(optimised)
    print_time("Gate counting before GridSynth", start)

    print("Gate counts before GridSynth:")
    for gate, count in counts["Gate Breakdown"].items():
        print(f"  {gate}: {count}")

    start = time.perf_counter()

    final_circuit = gridsynth_decompose(
        optimised,
        epsilon_synthesis
    )

    print_time("GridSynth decomposition", start)

    start = time.perf_counter()

    counts = get_gate_counts(final_circuit)

    print("Gate counts after GridSynth:")
    for gate, count in counts["Gate Breakdown"].items():
        print(f"  {gate}: {count}")

    print_time("Gate counting after GridSynth", start)

    return final_circuit, counts

def build_stepwise_circuit(n_qubits, coupling, t, order, epsilon, fixed_steps=None):

    start = time.perf_counter()

    epsilon_synthesis = epsilon / 2
    epsilon_trotter = epsilon / 2

    H = build_heisenberg(n_qubits, coupling)

    print("\n=== Stepwise Optimisation ===")
    print("Hamiltonian:")
    print(H)
    print_time("Hamiltonian construction", start)

    start = time.perf_counter()

    steps = determine_trotter_steps(
        H,
        time=t,
        order=order,
        epsilon=epsilon_trotter,
        fixed_steps=fixed_steps
    )

    print("Trotter steps used:", steps)
    print_time("Trotter-step calculation", start)

    start = time.perf_counter()

    trotter_step = build_trotter_step(
        H,
        t,
        steps,
        order
    )
    print_time("Single Trotter step construction", start)

    start = time.perf_counter()

    optimised_step = optimise_circuit(trotter_step)
    print_time("Single-step optimisation", start)

    start = time.perf_counter()

    step_qscript = qp.tape.make_qscript(optimised_step)()
    step_operations = step_qscript.operations
    print_time("Optimised-step extraction", start)

    start = time.perf_counter()

    circuit = repeat_operations(
        step_operations,
        n_qubits,
        steps
    )
    print_time("Step repetition/full circuit construction", start)

    start = time.perf_counter()

    counts = get_gate_counts(circuit)
    print_time("Gate counting before GridSynth", start)

    print("Gate counts before GridSynth:")
    for gate, count in counts["Gate Breakdown"].items():
        print(f"  {gate}: {count}")

    start = time.perf_counter()

    final_circuit = gridsynth_decompose(
        circuit,
        epsilon_synthesis
    )

    print_time("GridSynth decomposition", start)

    start = time.perf_counter()

    counts = get_gate_counts(final_circuit)

    print("Gate counts after GridSynth:")
    for gate, count in counts["Gate Breakdown"].items():
        print(f"  {gate}: {count}")
            
    print_time("Gate counting after GridSynth", start)

    return final_circuit, counts

n_qubits = 4
coupling = [1.0, 2.0, 1.5]
t = 1.0
order = 2
epsilon = 1e-4

full_circuit, gate_counts_1 = build_full_circuit(
    n_qubits,
    coupling,
    t,
    order,
    epsilon,
    fixed_steps=50
)

stepwise_circuit, gate_counts_2 = build_stepwise_circuit(
    n_qubits,
    coupling,
    t,
    order,
    epsilon,
    fixed_steps=50
)

print("\n=== Gate Counts ===")
print("Full circuit:", gate_counts_1)
print("Stepwise circuit:", gate_counts_2)