# Project Log

## Accuracy and Bounds - 20/08/2026

I am writing to get my thoughts down on paper as there are a lot of different movie components to this paper that I am trying to emulate. Currently, my focus is on incorporating an accuracy bound epsilon into my code. Currently I have the number of trotter steps as a direct input to my process, which is unrealistic and doesn’t match Childs’ original work.

In reality, you would more likely define an epsilon as a lower bound of the accuracy of the produced circuit and then use a method to select a number of Trotter steps to guarantee such performance. The paper mentions four such methods, but the two that I can see are most used are Commutator Bounding and Empirical Bounding.

### Commutator Bounding

Using commuting terms to improve bounds is a particularly apt approach to Heisenberg model Hamiltonians and Pauli hamiltonians in general, given their generally higher commutativity. The paper identifies that naive computation is computationally prohibitive, scaling with O(n^{2k+1}), so they use techniques that exploit the structure of the Hamiltonian. They choose to focus specifically on one-dimensional, nearest neighbour Heisenberg Hamiltonians, which implies that they don’t need to loop through the terms to check commutativity, they can quickly derive the total contribution as a function on the number of qubits N. I would like to confirm if we are seeking to focus on for our study. If so, I will aim to replicate this logic from their implementation as a python function.

### Empirical Bounding

The empirical method used in this study just allocate fixed values of r depending on the system size, given that actual simulation error can only be directly computed for small instances. No guarantee of correctness, but is believed to better capture the true performance of product formula simulations and indicate the extent to which the other rigorous bounds like Commutator Bounding are loose.

## Rz Decomposition - 20/08/2026

I want to discuss the strategy I should use to decompose a Rz gate into Clifford+T gates for resource estimation. I can see that Childs et al. used an approach from N.J. Ross' "Optimal ancilla-free Cliﬀord+T approximation of z-rotations", first Trotterising, then optimising at the Rz level, then decomposing the rotations into Clifford+T gates to determine final gate counts.

I am also aware that Solovay Kitaev and more simple approximation methods that don't require direct decomposition may be able to achieve the same goal, so would like to discuss what might be best for my project.