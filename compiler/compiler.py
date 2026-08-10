import numpy as np

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
        self.counts = {"total": 0, "single_qubit": 0, "two_qubit": 0, "CNOT": 0, "RZ": 0}

    def compile(self):
        computeGates = []

        for gate in self.pauliTerm.gates:
            if gate[0] == "X":
                computeGates.append(("H", [gate[1][0]], None))
                self.counts["total"] += 1
                self.counts["single_qubit"] += 1
            elif gate[0] == "Y":
                computeGates.append(("S", [gate[1][0]], -1))
                computeGates.append(("H", [gate[1][0]], None))
                self.counts["total"] += 2
                self.counts["single_qubit"] += 2
            if gate != self.pauliTerm.gates[-1]:
                computeGates.append(("CNOT", [gate[1][0], self.last_non_idt_idx], None))
                self.counts["total"] += 1
                self.counts["two_qubit"] += 1
                self.counts["CNOT"] += 1
        self.output = computeGates.copy()
        uncomputeGates = [self.inverse_gate(gate) for gate in computeGates[::-1]]
        self.counts = {key: self.counts[key] * 2 for key in self.counts}
        self.counts["total"] += 1
        self.counts["RZ"] += 1
        self.counts["single_qubit"] += 1
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
        self.counts = {"total": 0, "single_qubit": 0, "two_qubit": 0, "CNOT": 0, "RZ": 0}

    def standard_simulation(self):
        for term in self.hamiltonian.terms:
            compiler = Compiler(term, self.t * term.coefficient)
            compiler.compile()
            self.counts = {key: self.counts[key] + compiler.counts[key] for key in self.counts}
            compiler.print_output()
        print(self.counts)

# pauliString = PauliTerm(1, [("X", [0]), ("Y", [1]), ("Z", [3]), ("X", [4])])
# angle = 45
# compiler = Compiler(pauliString, angle)
# compiler.compile()
# compiler.print_output()

hamiltonian = Hamiltonian([PauliTerm(1, [("X", [0]), ("Y", [1]), ("Z", [3]), ("X", [4])]),
                           PauliTerm(2, [("Z", [0]), ("X", [1]), ("Y", [2])])], num_qubits=6)
t = 45
simulator = Simulator(hamiltonian, t)
simulator.standard_simulation()