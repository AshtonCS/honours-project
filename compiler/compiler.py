import numpy as np

class Compiler:
    def __init__(self, gates):
        self.gates = gates
        self.output = []

    def compile(self):
        for gate in self.gates:
            if gate[0] != 0:
                self.output.append(gate)
        self.output = np.array(self.output)
        return self.output

gates = np.array([[0, 2], [1, 2], [2, 4], [3, 6]])
compiler = Compiler(gates)
print(compiler.gates)
compiler.compile()
print(compiler.output)