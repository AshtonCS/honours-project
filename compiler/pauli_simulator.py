class PauliTerm:
    def __init__(self, coefficient, paulis):
        self.coefficient = coefficient
        self.paulis = paulis

class Compiler:
    def __init__(self, pauli_term, angle):
        self.pauli_term = pauli_term
        self.angle = angle

    def compile(self):
        if not self.pauli_term.paulis:
            return []
        last_non_idt_idx = self.pauli_term.paulis[-1][1][0]
        
        compute_gates = []
        output = []

        for gate in self.pauli_term.paulis:
            if gate[0] == "X":
                compute_gates.append(("H", [gate[1][0]], None))
            elif gate[0] == "Y":
                compute_gates.append(("S", [gate[1][0]], -1))
                compute_gates.append(("H", [gate[1][0]], None))
            if gate != self.pauli_term.paulis[-1]:
                compute_gates.append(("CNOT", [gate[1][0], last_non_idt_idx], None))
        output = compute_gates.copy()
        uncompute_gates = [self.inverse_gate(gate) for gate in compute_gates[::-1]]
        output.append(("RZ", [last_non_idt_idx], self.angle * 2))
        output.extend(uncompute_gates)
        return output

    def inverse_gate(self, gate):
        return gate[0], gate[1], -gate[2] if gate[2] is not None else None

class Hamiltonian:
    def __init__(self, terms, num_qubits):
        self.terms = terms
        self.num_qubits = num_qubits

class HamiltonianSimulator:
    def __init__(self, hamiltonian, t):
        self.hamiltonian = hamiltonian
        self.t = t

    def standard_simulation(self):
        output = []
        for term in self.hamiltonian.terms:
            compiler = Compiler(term, self.t * term.coefficient)
            compiled_term = compiler.compile()
            output.extend(compiled_term)
        output = self.optimise(output)
        counts = self.count_gates(output)
        return output, counts

    def first_order_trotter_simulation(self, num_steps):
        output = []
        for term in self.hamiltonian.terms:
            compiler = Compiler(term, self.t * term.coefficient / num_steps)
            compiled_term = compiler.compile()
            output.extend(compiled_term)

        trotter_step = output.copy()
        for _ in range(num_steps - 1):
            output.extend(trotter_step)
        
        output = self.optimise(output)
        counts = self.count_gates(output)
        return output, counts

    def second_order_trotter_simulation(self, num_steps):
            output = []
            compiled_term_stack = []
            for i, term in enumerate(self.hamiltonian.terms):
                if i == len(self.hamiltonian.terms) - 1:
                    angle = self.t * term.coefficient / num_steps
                else:
                    angle = self.t * term.coefficient / (2 * num_steps)

                compiler = Compiler(term, angle)
                compiled_term = compiler.compile()

                output.extend(compiled_term)
                compiled_term_stack.append(compiled_term)

            for term in reversed(compiled_term_stack[:-1]):
                output.extend(term)
    
            trotter_step = output.copy()

            for _ in range(num_steps - 1):
                output.extend(trotter_step)
            
            output = self.optimise(output)
            counts = self.count_gates(output)
            return output, counts
    
    def optimise(self, output):
        optimised = []

        for gate in output:
            if optimised and self.can_cancel(optimised[-1], gate):
                optimised.pop()
            elif optimised and self.can_merge_rz(optimised[-1], gate):
                merged_gate = self.merge_rz(optimised.pop(), gate)
                optimised.append(merged_gate)
            else:
                optimised.append(gate)

        return optimised

    def count_gates(self, output):
        counts = {"total": 0, "single_qubit": 0, "two_qubit": 0, "CNOT": 0, "RZ": 0}

        for gate in output:
            gate_counts = self.get_gate_counts(gate)
            for key in counts:
                counts[key] += gate_counts[key]
        return counts

    def can_cancel(self, gate1, gate2):
        return (gate1[0] == gate2[0] 
                and gate1[1] == gate2[1]
                and (
                    (gate1[2] is None and gate2[2] is None)
                    or
                    (gate1[2] is not None
                        and gate2[2] is not None
                        and gate1[2] == -gate2[2])
            )
        )

    def print_output(self, output):
        for gate in output:
            print(gate[0], gate[1], gate[2] if gate[2] is not None else "")

    def get_gate_counts(self, gate):
        counts = {"total": 0, "single_qubit": 0, "two_qubit": 0, "CNOT": 0, "RZ": 0}
        counts["total"] += 1
        if gate[0] in ["H", "S", "RZ"]:
            counts["single_qubit"] += 1
            if gate[0] == "RZ":
                counts["RZ"] += 1
        elif gate[0] == "CNOT":
            counts["two_qubit"] += 1
            counts["CNOT"] += 1
        return counts

    def can_merge_rz(self, gate1, gate2):
        return (gate1[0] == "RZ" and gate2[0] == "RZ" and gate1[1] == gate2[1])

    def merge_rz(self, gate1, gate2):
        angle = gate1[2] + gate2[2]
        return ("RZ", gate1[1], angle)


hamiltonian = Hamiltonian([PauliTerm(1, [("X", [0]), ("Y", [1]), ("Z", [3]), ("X", [4])]),
                           PauliTerm(2, [("X", [0]), ("Z", [1]), ("Y", [2])])], num_qubits=6)

t = 3

hamiltonian_simulator = HamiltonianSimulator(hamiltonian, t)

std_output, std_counts = hamiltonian_simulator.standard_simulation()
hamiltonian_simulator.print_output(std_output)
print("Gate counts:", std_counts)

num_steps = 3
first_order_trotter_output, first_order_trotter_counts = hamiltonian_simulator.first_order_trotter_simulation(num_steps)
hamiltonian_simulator.print_output(first_order_trotter_output)
print("Gate counts:", first_order_trotter_counts)

num_steps = 3
second_order_trotter_output, second_order_trotter_counts = hamiltonian_simulator.second_order_trotter_simulation(num_steps)
hamiltonian_simulator.print_output(second_order_trotter_output)
print("Gate counts:", second_order_trotter_counts)