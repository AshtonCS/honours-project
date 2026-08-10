import numpy as np
from param import output

class PauliTerm:
    def __init__(self, coefficient, gates):
        self.coefficient = coefficient
        self.gates = gates

class Compiler:
    def __init__(self, pauliTerm, angle):
        self.pauliTerm = pauliTerm
        self.angle = angle
        self.last_non_idt_idx = pauliTerm.gates[-1][1][0] if pauliTerm.gates else 0
        self.output = []

    def compile(self):
        computeGates = []

        for gate in self.pauliTerm.gates:
            if gate[0] == "X":
                computeGates.append(("H", [gate[1][0]], None))
            elif gate[0] == "Y":
                computeGates.append(("S", [gate[1][0]], -1))
                computeGates.append(("H", [gate[1][0]], None))
            if gate != self.pauliTerm.gates[-1]:
                computeGates.append(("CNOT", [gate[1][0], self.last_non_idt_idx], None))
        self.output = computeGates.copy()
        uncomputeGates = [self.inverse_gate(gate) for gate in computeGates[::-1]]
        self.output.append(("RZ", [self.last_non_idt_idx], self.angle * 2))
        self.output.extend(uncomputeGates)

    def inverse_gate(self, gate):
        return gate[0], gate[1], -gate[2] if gate[2] is not None else None

    def print_output(self):
        for gate in self.output:
            print(gate[0], gate[1], gate[2] if gate[2] is not None else "")

class Hamiltonian:
    def __init__(self, terms, num_qubits):
        self.terms = terms
        self.num_qubits = num_qubits

class Simulator:
    def __init__(self, hamiltonian, t):
        self.hamiltonian = hamiltonian
        self.t = t
        self.output = []
        self.counts = {"total": 0, "single_qubit": 0, "two_qubit": 0, "CNOT": 0, "RZ": 0}

    def standard_simulation(self):
        for term in self.hamiltonian.terms:
            compiler = Compiler(term, self.t * term.coefficient)
            compiler.compile()
            self.output.extend(compiler.output)
        self.optimise()
        self.count_gates()
        self.print_output()
        print(self.counts)

    def first_order_trotter_simulation(self, num_steps):
        for term in self.hamiltonian.terms:
            compiler = Compiler(term, self.t * term.coefficient / num_steps)
            compiler.compile()
            self.output.extend(compiler.output)

        trotter_step = self.output.copy()
        for _ in range(num_steps - 1):
            self.output.extend(trotter_step)
        
        self.optimise()
        self.count_gates()
        self.print_output()
        print(self.counts)
    
    def optimise(self):
        optimised = []

        for gate in self.output:
            if optimised and self.can_cancel(optimised[-1], gate):
                optimised.pop()
            elif optimised and self.can_merge_rz(optimised[-1], gate):
                merged_gate = self.merge_rz(optimised.pop(), gate)
                optimised.append(merged_gate)
            else:
                optimised.append(gate)

        self.output = optimised

    def count_gates(self):
        self.counts = {"total": 0, "single_qubit": 0, "two_qubit": 0, "CNOT": 0, "RZ": 0}

        for gate in self.output:
            gate_counts = self.get_gate_counts(gate)
            for key in self.counts:
                self.counts[key] += gate_counts[key]

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

    def print_output(self):
        for gate in self.output:
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

        if np.isclose(angle, 0):
            return None

        return ("RZ", gate1[1], angle)


hamiltonian = Hamiltonian([PauliTerm(1, [("X", [0]), ("Y", [1]), ("Z", [3]), ("X", [4])]),
                           PauliTerm(2, [("X", [0]), ("Z", [1]), ("Y", [2])])], num_qubits=6)

t = 3
simulator = Simulator(hamiltonian, t)
simulator.standard_simulation()

trotter_simulator = Simulator(hamiltonian, t)
num_steps = 3
trotter_simulator.first_order_trotter_simulation(num_steps)