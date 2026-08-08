import numpy as np

class Compiler:
    def __init__(self, gates, num_qubits, coefficient):
        self.gates = gates
        self.num_qubits = num_qubits
        self.last_non_idt_idx = gates[-1][1][0] if gates else 0
        self.coefficient = coefficient
        self.output = []

    def compile(self):
        computeGates = []
        for gate in self.gates:
            if gate[0] == "X":
                computeGates.append(("H", [gate[1][0]]))
            elif gate[0] == "Y":
                computeGates.append(("H", [gate[1][0]]))
                computeGates.append(("S", [gate[1][0]]))
            if gate[0] != "Z" and gate != gates[-1]:
                computeGates.append(("CNOT", [gate[1][0], self.last_non_idt_idx]))
        self.output = computeGates.copy()
        self.output.append(("RZ", [self.last_non_idt_idx], self.coefficient))
        self.output.extend(computeGates[::-1])
            
gates = [("X", [0]), ("Y", [1]), ("Z", [3]), ("X", [4])]
num_qubits = 6
coefficient = 1
compiler = Compiler(gates, num_qubits, coefficient)
print(compiler.gates)
compiler.compile()
print(compiler.output)