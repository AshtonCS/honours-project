import pennylane as qp
import numpy as np
import time
import math

# Configuration Settings

n_qubits = 4
t = 4.0
order = 2
epsilon = 1e-3

coupling = [1.0, 1.0, 1.0]

random_field = np.array([
    -0.8444192498333132,
     0.7998200058907112,
     0.2371705373647761,
    -0.5823152775681468,
])

fixed_steps = None

boundary_condition = True
trotter_error_method = "commutator-bound"
rz_synthesis = False
simulation_mode = "single step" # "global", "stepwise", "single step", "two step"

def build_heisenberg(
    n_qubits,
    coupling,
    field_strength=1.0,
    seed=None,
    random_field=None,
):
    if random_field is None:
        rng = np.random.default_rng(seed)
        random_field = rng.uniform(
            -field_strength,
            field_strength,
            size=n_qubits,
        )
    else:
        random_field = np.asarray(random_field, dtype=float)

        if random_field.shape != (n_qubits,):
            raise ValueError(
                f"random_field must contain exactly {n_qubits} values"
            )

    exchange = qp.spin.heisenberg(
        lattice="chain", # implies 1D chain of qubits
        n_cells=[n_qubits],
        coupling=coupling, #coupling constants, J_x, J_y, J_z in the literature
        boundary_condition=boundary_condition, # periodic chain: 1-2-...-N-1
    )

    exchange_coefficients, exchange_operators = exchange.terms()

    return qp.Hamiltonian(
        list(exchange_coefficients) + list(random_field),
        exchange_operators
        + [qp.PauliZ(wire) for wire in range(n_qubits)],
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


def scale_gate_counts(counts, repetitions):
    scaled_breakdown = {
        gate: count * repetitions
        for gate, count in counts["Gate Breakdown"].items()
    }

    return {
        "Total Gates": counts["Total Gates"] * repetitions,
        "Clifford+T Gates": counts["Clifford+T Gates"] * repetitions,
        "Clifford Gates": counts["Clifford Gates"] * repetitions,
        "CNOTs": counts["CNOTs"] * repetitions,
        "T Gates": counts["T Gates"] * repetitions,
        "Gate Breakdown": scaled_breakdown,
    }

def print_gate_counts(counts, label):
    print(f"\n=== {label} ===")
    for gate, count in counts["Gate Breakdown"].items():
        print(f"  {gate}: {count}")

def trotter_error(H, time, steps, order):
    trotter = qp.TrotterProduct(
        H,
        time=time,
        n=steps,
        order=order,
    )

    return float(
        trotter.error(method=trotter_error_method).error
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

    print(f"Determined Trotter steps: {upper}")

    return upper

def build_trotter_block(H, n_qubits, time, steps, order, block_steps):
    step_time = time / steps

    dev = qp.device("default.qubit", wires=n_qubits)

    @qp.qnode(dev)
    def block():
        for _ in range(block_steps):
            qp.TrotterProduct(
                H,
                time=step_time,
                n=1,
                order=order,
            )
        return qp.state()

    return block

import math


def simcount_pf1com_error(n, m, l, t, steps):
    """
    SimCount's first-order commutator error bound (pf1com).

    Parameters
    ----------
    n : int
        Number of qubits/spins.
    m : int
        Number of Hamiltonian terms.
    l : float
        Maximum absolute Hamiltonian coefficient.
    t : float
        Total simulation time.
    steps : int
        Number of product-formula steps (r).

    Returns
    -------
    float
        Upper bound on the first-order product-formula error.
    """

    # SimCount's com1(n)
    if n == 1:
        com1 = 0
    elif n == 2:
        com1 = 4
    else:
        com1 = 10 * n

    mlt = m * l * t

    x = (l * t) ** 2 / steps
    x_prime = com1

    y = mlt ** 3 / (3 * steps ** 2)
    y_prime = math.exp(mlt / steps)

    return x * x_prime + y * y_prime


def simcount_pf2com_error(n, m, l, t, steps):
    """
    SimCount's second-order commutator error bound (pf2com).
    """

    # SimCount's com2(n)
    if n == 3:
        com2 = 194
    else:
        com2 = 40 * n**2 - 58 * n

    mlt = m * l * t

    x = (l * t) ** 3 / steps**2
    x_prime = com2

    y = 4 * mlt**4 / (3 * steps**3)
    y_prime = math.exp((2 * mlt) / steps)

    return x * x_prime + y * y_prime


def simcount_pf4com_error(n, m, l, t, steps):
    """
    SimCount's fourth-order commutator error bound (pf4com).
    """

    # SimCount's fourth-order coefficient
    p = 0.414490771794376
    fp1 = 4 * p - 1

    # System-size-dependent commutator bound
    if n == 3:
        com4 = 23073564672
    elif n == 4:
        com4 = 94192316416
    elif n == 5:
        com4 = 278878851840
    else:
        com4 = (
            1280000000 * n**4
            - 7701760000 * n**3
            + 23685120000 * n**2
            - 30224677632 * n
        )

    mlt = m * l * t

    x = (
        ((fp1 / 2) * l * t) ** 5
        / (120 * steps**4)
    )
    x_prime = com4

    y = (
        2 * (5 * fp1 * mlt) ** 6
        / (720 * steps**5)
    )
    y_prime = math.exp(
        (5 * fp1 * mlt) / steps
    )

    return x * x_prime + y * y_prime


def determine_simcount_steps(error_function, n, m, l, t, epsilon):
    """
    Determine the minimum number of product-formula steps
    satisfying the supplied error bound.
    """

    lower = 0
    upper = 1

    # Exponential search
    while error_function(n, m, l, t, upper) > epsilon:
        lower = upper
        upper *= 2

    # Binary search
    while lower + 1 < upper:
        middle = (lower + upper) // 2

        if error_function(n, m, l, t, middle) <= epsilon:
            upper = middle
        else:
            lower = middle

    return upper

def print_time(label, start):
    print(f"{label}: {time.perf_counter() - start:.3f}s")

def build_circuit(
    n_qubits,
    coupling,
    t,
    order,
    epsilon,
    fixed_steps=None,
    random_field=None,
    simulation_mode="global",
    gridsynth=True,
):
    valid_modes = {"global", "stepwise", "single step", "two step"}

    if simulation_mode not in valid_modes:
        raise ValueError(
            "simulation_mode must be 'global', 'stepwise', "
            "'single step', or 'two step'"
        )

    if order not in [1, 2, 4]:
            raise ValueError(
                "order must be 1, 2, or 4 to align with Simcount's commutator bound implementations"
            )

    epsilon_trotter = epsilon
    epsilon_synthesis = None
    if gridsynth:
        epsilon_trotter = epsilon / 2
        epsilon_synthesis = epsilon / 2

    start = time.perf_counter()

    H = build_heisenberg(
        n_qubits,
        coupling,
        random_field=random_field,
    )

    print(f"\n=== {simulation_mode} simulation ===")
    print_time("Hamiltonian construction", start)

    start = time.perf_counter()

    # steps = determine_trotter_steps(
    #     H,
    #     time=t,
    #     order=order,
    #     epsilon=epsilon_trotter,
    #     fixed_steps=fixed_steps,
    # )

    if order == 1:
        error_function = simcount_pf1com_error
    elif order == 2:
        error_function = simcount_pf2com_error
    else:
        error_function = simcount_pf4com_error

    steps = determine_simcount_steps(
        error_function=error_function,
        n=n_qubits,
        m=4*n_qubits,
        l = max(max(abs(x) for x in coupling),
                max(abs(x) for x in random_field)),
        t=t,
        epsilon=epsilon_trotter,
    ) if fixed_steps is None else fixed_steps

    print("Trotter steps used:", steps)
    print_time("Trotter-step calculation", start)

    if simulation_mode == "global":
        circuit = build_trotter_circuit(
            H,
            n_qubits,
            t,
            steps,
            order,
        )
        raw_counts = get_gate_counts(circuit)

        start = time.perf_counter()
        circuit = optimise_circuit(circuit)
        print_time("Optimisation", start)

        before_synthesis_counts = get_gate_counts(circuit)

        if gridsynth:
            start = time.perf_counter()
            circuit = gridsynth_decompose(circuit, epsilon_synthesis)
            print_time("GridSynth decomposition", start)

        after_synthesis_counts = get_gate_counts(circuit)
        print_gate_counts(raw_counts, f"Raw {simulation_mode} circuit")
        print_gate_counts(before_synthesis_counts, "Before RZ synthesis")
        print_gate_counts(after_synthesis_counts, "After RZ synthesis")
        return circuit, before_synthesis_counts, after_synthesis_counts, raw_counts

    if simulation_mode == "stepwise":
        trotter_step = build_trotter_step(
            H,
            t,
            steps,
            order,
        )
        optimised_step = optimise_circuit(trotter_step)
        step_qscript = qp.tape.make_qscript(optimised_step)()
        step_operations = step_qscript.operations

        circuit = repeat_operations(
            step_operations,
            n_qubits,
            steps,
        )

        raw_counts = get_gate_counts(circuit)

        before_synthesis_counts = get_gate_counts(circuit)

        if gridsynth:
            circuit = gridsynth_decompose(circuit, epsilon_synthesis)

        after_synthesis_counts = get_gate_counts(circuit)
        print_gate_counts(before_synthesis_counts, "Before RZ synthesis")
        print_gate_counts(after_synthesis_counts, "After RZ synthesis")
        return circuit, before_synthesis_counts, after_synthesis_counts, raw_counts

    else:
        block_steps = 1 if simulation_mode == "single step" else 2

        if steps % block_steps != 0:
            raise ValueError(
                f"steps must be divisible by {block_steps} in "
                f"'{simulation_mode}' mode"
            )

        repetitions = steps // block_steps

        raw_block = build_trotter_block(
            H,
            n_qubits,
            t,
            steps,
            order,
            block_steps,
        )

        raw_block_counts = get_gate_counts(raw_block)
        raw_counts = scale_gate_counts(raw_block_counts, repetitions)

        optimised_block = optimise_circuit(raw_block)

        before_synthesis_block_counts = get_gate_counts(optimised_block)
        before_synthesis_counts = scale_gate_counts(
            before_synthesis_block_counts,
            repetitions,
        )

        if gridsynth:
            block = gridsynth_decompose(
                optimised_block,
                epsilon_synthesis/repetitions,
            )
        else:
            block = optimised_block

        block_qscript = qp.tape.make_qscript(block)()
        block_operations = block_qscript.operations

        block_counts = get_gate_counts(block)
        after_synthesis_counts = scale_gate_counts(
            block_counts,
            repetitions,
        )

        circuit = repeat_operations(
            block_operations,
            n_qubits,
            repetitions,
        )
        print_gate_counts(raw_counts, f"Raw {simulation_mode} circuit")
        print_gate_counts(
            before_synthesis_counts,
            f"Before RZ synthesis ({simulation_mode})",
        )
        print_gate_counts(
            after_synthesis_counts,
            f"After RZ synthesis ({simulation_mode})",
        )
        return (
            circuit,
            before_synthesis_counts,
            after_synthesis_counts,
            raw_counts,
        )

circuit, before_synthesis_counts, after_synthesis_counts, raw_counts = build_circuit(
    n_qubits,
    coupling,
    t,
    order,
    epsilon,
    fixed_steps=fixed_steps,
    random_field=random_field,
    simulation_mode=simulation_mode,
    gridsynth=True,
)



