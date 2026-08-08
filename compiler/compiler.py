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

    def compile(self):
        computeGates = []
        for gate in self.pauliTerm.gates:
            if gate[0] == "X":
                computeGates.append(("H", [gate[1][0]], None))
            elif gate[0] == "Y":
                computeGates.append(("H", [gate[1][0]], None))
                computeGates.append(("S", [gate[1][0]], +1))
            if gate[0] != "Z" and gate != self.pauliTerm.gates[-1]:
                computeGates.append(("CNOT", [gate[1][0], self.last_non_idt_idx], None))
        self.output = computeGates.copy()
        self.output.append(("RZ", [self.last_non_idt_idx], self.angle))
        uncomputeGates = [self.inverse_gate(gate) for gate in computeGates[::-1]]
        self.output.extend(uncomputeGates)

    def inverse_gate(self, gate):
        return gate[0], gate[1], -gate[2] if gate[2] is not None else None

    def print_output(self):
        for gate in self.output:
            print(gate[0], gate[1], gate[2] if gate[2] is not None else "")

pauliString = PauliTerm(1, [("X", [0]), ("Y", [1]), ("Z", [3]), ("X", [4])])
angle = 45
compiler = Compiler(pauliString, angle)
compiler.compile()
compiler.print_output()