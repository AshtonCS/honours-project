# Implementation Strategy

## Completed Tasks
1. Operational Program, Input/Output same circuit in NumPy 
  * Pauli circuits are formatted as NumPy arrays of gates, with two indexes representing the Pauli operation and the qubit it acts on
  * The Compiler class takes a set of gates representing a circuit as input and returns a simplified circuit
2. Remove Identity Gates from Output
3. Cancel Pairs of Consecutive Identical Gates on the Same Qubit

## Remaining Tasks
4. Cancel Pairs of Non-Consecutive Identical Gates on the Same Qubit
5. Obey Pauli Multiplication Rules
6. Track Global Phases
7. Restructure Circuits to incorporate Multi-Qubit Gates
