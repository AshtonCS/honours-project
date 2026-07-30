import numpy as np

class Compiler:
    def __init__(self, gates):
        self.gates = gates

gates = np.array([[1, 2], [2, 4], [3, 6]])
compiler = Compiler(gates)