import numpy as np

class Compiler:
    def __init__(self, gates):
        self.gates = gates
        self.output = []

    def compile(self):
        for gate in self.gates:
            if gate[0] == 0:
                continue
            if self.output and np.array_equal(self.output[-1], gate):
                self.output.pop()
                continue
            self.output.append(gate)
        self.output = np.array(self.output)
        return self.output

gates = np.array([[0, 0], [3, 1], [1, 2], [1, 2], [2, 3]])
compiler = Compiler(gates)
print(compiler.gates)
compiler.compile()
print(compiler.output)