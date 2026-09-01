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

## Experimentation and Comparison to njross/simcount - 26/08/2026

### Today’s progress / decisions

* Settled on using the nearest-neighbour Heisenberg Hamiltonian as the main baseline.
* Using 4 qubits for now, with Jx = Jy = Jz = 1.
* Open boundary conditions rather than wrapping the chain around.
* Using second-order Trotterisation.
* Overall accuracy set to 1e-4.
* Splitting the error budget 50/50 between Trotter error and synthesis error.
* Using PennyLane’s commutator-bound to work out the number of Trotter steps rather than just picking a number manually.
* For the current parameters this gives around 2772 steps.
* Decided that I don’t want to optimise the implementation beyond what simcount is doing at this stage. The aim is to reproduce their general methodology in Python/PennyLane first.
* Important distinction from earlier: I don’t want to just do “cost of one Trotter step times number of steps” if I can avoid it. I want the actual Clifford+T circuit/resource count so that cancellations and the individual rotation synthesis costs are accounted for.
* Currently using PennyLane Gridsynth for the rotation synthesis and qp.specs() for the final resource counts.

**How this lines up with Childs / simcount**

* The overall structure is basically what I want from the Childs/simcount approach:

  * Hamiltonian
  * Trotter approximation
  * determine steps needed for accuracy
  * split error between Trotterisation and synthesis
  * Clifford+T synthesis
  * count resources
* The main difference is implementation. I’m explicitly generating the circuit using PennyLane rather than using the much more compact resource-counting approach used in simcount.
* This should make my implementation easier to inspect and understand, even if it is much less efficient computationally.

**Advantages**

* Much more concrete than estimating resources from a formula.
* I can actually inspect the circuit at each stage.
* Easy to change the Hamiltonian, number of qubits, Trotter order, accuracy etc.
* Gives separate T, CNOT, Clifford and total gate counts.
* Should make it easier to validate individual parts of the compiler before moving onto optimisation experiments.
* Python/PennyLane means I can potentially extend this later without having to completely rewrite the project.

**Disadvantages / problems**

* The explicit circuits get VERY big very quickly.
* 10 steps already gave ~30k gates.
* 100 steps gave ~258k gates.
* 500 steps gave ~731k gates.
* Gate counting took ~1.3 sec, 21 sec and 141 sec respectively.
* So 2772 steps is probably going to be painful if I try to fully expand and analyse it this way.
* The Trotter-step search itself is also taking ~14–24 seconds, although this is less of a concern than the huge circuit.
* PennyLane seems to defer some of the expensive processing, so the timings for “optimisation” and “Gridsynth” aren't necessarily showing the actual amount of work being done.
* Tried replacing qp.specs() with direct circuit access, but ran into issues with PennyLane QNodes/construct(). Need to investigate the correct way of accessing the generated operations without actually executing the quantum circuit.

**Next steps**

* Work out the correct way to access/count the generated Clifford+T operations without running the quantum simulation.
* Try the full ~2772-step case once the counting process is sorted.
* Compare the results against simcount for an equivalent test case.
* Check that my interpretation of the Childs error bound and error allocation is correct.
* Once the baseline is validated, start looking at scaling with number of qubits.
* Keep optimisation separate for now — first get a reliable baseline implementation matching the existing methodology.

## Notes on Circuit Preparation and Rz Decomposition - 26/08/2026

I have been experimenting with aspects of my code - notably, given the classical computational overhead of simulating circuits by generating full expansions before optimisation, I decided to attempt implementing a naive approach that would write and optimise a single trotter step before repeating it N times. This test was supposed to indicate that these simulations could be conducted to a lower degree of accuracy significantly more efficiently and potentially set up an approach that incorporates gate cancellation around the step boundaries for increased effectiveness.

My initial results after implementing a naive stepwise approach were surprising - while I saw a significant reduction in the time taken to produce optimised circuits using the stepwise approach, I was seeing exactly the same gate counts outputted for either process. This made limited sense given that I was conscious that the new approach ignored boundary conditions.

After adding some in-progress displays, it became clear that this was not the case - the two methods produced different results, with full circuits being more streamlined than stepwise ones, but running qp.clifford_t_decomposition(circuit, epsilon=epsilon, method="gridsynth") somehow reduced the gate count of the stepwise implementation to map the full implementation. I am now looking into this matter to understand what optimisation this function may run.

