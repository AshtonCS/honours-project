"""Tests for the Nam et al. optimiser reimplementation.

Two independent layers, matching the plan for this module:

* soundness -- every rewrite preserves the unitary (up to global phase);
* fidelity  -- our reduction factors, beside the published ones.

Soundness is the layer that matters: an optimiser that silently changes the
circuit is worse than no optimiser. It is checked by full unitary comparison
on small circuits and by random-state equivalence on the n=13 reference
circuits.
"""

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from plsimcount import simcount, simulate  # noqa: E402
from plsimcount.nam import bridge, optimiser, quipper, rules, samples, verify  # noqa: E402
from plsimcount.nam.gates import Circuit, Gate  # noqa: E402
from plsimcount.nam.routines import (  # noqa: E402
    routine1_hadamard_reduction,
    routine2_single_qubit_cancellation,
    routine3_two_qubit_cancellation,
    routine4_rotation_merging,
)

FIELDS_N4 = [-0.8444192498333132, 0.7998200058907112, 0.2371705373647761, -0.5823152775681468]


def _same_unitary(a: Circuit, b: Circuit, atol: float = 1e-10) -> bool:
    ua, ub = a.matrix(), b.matrix()
    trace = np.trace(ub.conj().T @ ua)
    phase = np.exp(1j * np.angle(trace)) if abs(trace) > 1e-12 else 1.0
    return bool(np.allclose(ua, phase * ub, atol=atol, rtol=0.0))


# --- rules --------------------------------------------------------------------

def test_figure4_rules_hold():
    """Every Hadamard-reduction identity multiplies out correctly."""
    rows = rules.verify_hadamard_rules()
    assert len(rows) == 5
    for name, exact, up_to_phase in rows:
        assert up_to_phase, f"{name} is not even correct up to phase"
    exact_by_name = {name: exact for name, exact, _ in rows}
    # The two single-wire rules hold only up to exp(i pi/4); the rest exactly.
    assert exact_by_name["H-S-H"] is False
    assert exact_by_name["H-Sdg-H"] is False
    assert exact_by_name["HH-CNOT-HH"] is True
    assert exact_by_name["H-S-CNOT-Sdg-H"] is True
    assert exact_by_name["H-Sdg-CNOT-S-H"] is True


def test_every_hadamard_rule_reduces_h_count():
    for rule in rules.HADAMARD_RULES:
        assert rule.h_reduction() > 0, rule.name


@pytest.mark.parametrize("theta", [1e-3, 1e-5, 1e-6, 1e-7, 1e-9])
def test_small_rotation_does_not_commute_through_cnot_target(theta):
    """Regression: numpy's default rtol=1e-05 in np.allclose declared any
    rotation with |theta| <~ 1e-5 to commute with a CNOT on its own target.
    Angles here are t/r and reach 1e-7, so the default silently corrupted
    every circuit built from a loose step-count bound (pf4ana, pf4min)."""
    rz, cx = Gate("Rz", (1,), theta), Gate("CNOT", (0, 1))
    assert not rules.commutes(rz, cx)
    assert not rules.pattern_commutes(rz, cx)


def test_rotation_on_control_does_commute():
    rz, cx = Gate("Rz", (0,), 1e-7), Gate("CNOT", (0, 1))
    assert rules.commutes(rz, cx)
    assert rules.pattern_commutes(rz, cx)


def test_exact_commutation_is_a_superset_of_the_patterns():
    """The paper's pattern list is sufficient, not necessary; ours is exact."""
    pairs = [
        (Gate("CNOT", (0, 1)), Gate("CNOT", (0, 2))),
        (Gate("CNOT", (0, 1)), Gate("CNOT", (2, 1))),
        (Gate("X", (1,)), Gate("CNOT", (0, 1))),
        (Gate("Rz", (0,), 0.3), Gate("CNOT", (0, 1))),
    ]
    for a, b in pairs:
        assert rules.pattern_commutes(a, b) and rules.commutes(a, b)
    # Z on the control commutes but is not in the paper's figure.
    assert rules.commutes(Gate("Z", (0,)), Gate("CNOT", (0, 1)))


# --- gates --------------------------------------------------------------------

def test_rotation_cancellation_uses_a_relative_tolerance():
    """Regression: with an absolute tolerance, two small same-sign rotations
    tested as cancelling, and deleting them let the surrounding CNOTs cascade
    away -- the pf4ana samples collapsed to zero CNOT gates."""
    assert not Gate("Rz", (0,), 3e-13).cancels_with(Gate("Rz", (0,), 4e-13))
    assert not Gate("Rz", (0,), 1e-7).cancels_with(Gate("Rz", (0,), 1e-7))
    assert Gate("Rz", (0,), 0.5).cancels_with(Gate("Rz", (0,), -0.5))
    assert Gate("Rz", (0,), 1e-7).cancels_with(Gate("Rz", (0,), -1e-7))


def test_fixed_gate_cancellation():
    assert Gate("H", (0,)).cancels_with(Gate("H", (0,)))
    assert Gate("S", (0,)).cancels_with(Gate("Sdg", (0,)))
    assert not Gate("S", (0,)).cancels_with(Gate("S", (0,)))
    assert not Gate("H", (0,)).cancels_with(Gate("H", (1,)))


def test_cnot_embedding_respects_wire_order():
    from plsimcount.nam.gates import embed

    expected = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    assert np.allclose(embed(Gate("CNOT", (1, 0)), 2), expected)


def test_gate_validation_raises():
    with pytest.raises(ValueError):
        Gate("Rz", (0,))  # no angle
    with pytest.raises(ValueError):
        Gate("CNOT", (0, 0))  # same wire twice
    with pytest.raises(ValueError):
        Gate("Nonsense", (0,))


# --- routines -----------------------------------------------------------------

ROUTINE_CASES = [
    ("R1 HH-CNOT-HH", [Gate("H", (0,)), Gate("H", (1,)), Gate("CNOT", (0, 1)),
                       Gate("H", (0,)), Gate("H", (1,))], 2, routine1_hadamard_reduction, 1),
    ("R1 H-S-CNOT-Sdg-H", [Gate("H", (1,)), Gate("S", (1,)), Gate("CNOT", (0, 1)),
                           Gate("Sdg", (1,)), Gate("H", (1,))], 2, routine1_hadamard_reduction, 3),
    ("R2 cancel through disjoint", [Gate("H", (0,)), Gate("CNOT", (1, 2)),
                                    Gate("H", (0,))], 3, routine2_single_qubit_cancellation, 1),
    ("R2 merge through control", [Gate("Rz", (0,), 0.3), Gate("CNOT", (0, 1)),
                                  Gate("Rz", (0,), 0.4)], 2, routine2_single_qubit_cancellation, 2),
    ("R2 no merge through target", [Gate("Rz", (1,), 0.3), Gate("CNOT", (0, 1)),
                                    Gate("Rz", (1,), 0.4)], 2, routine2_single_qubit_cancellation, 3),
    ("R3 cancel through Rz on control", [Gate("CNOT", (0, 1)), Gate("Rz", (0,), 0.3),
                                         Gate("CNOT", (0, 1))], 2, routine3_two_qubit_cancellation, 1),
    ("R3 blocked by H", [Gate("CNOT", (0, 1)), Gate("H", (1,)),
                         Gate("CNOT", (0, 1))], 2, routine3_two_qubit_cancellation, 3),
]


@pytest.mark.parametrize("label,gates,n,routine,expected_len", ROUTINE_CASES)
def test_routine_is_sound_and_reduces_as_expected(label, gates, n, routine, expected_len):
    before = Circuit(gates, n)
    after = routine(before)
    assert _same_unitary(before, after), label
    assert len(after) == expected_len, label


def test_routine4_merges_the_papers_worked_example():
    """Section 4.4 / equations 5-6: two of the four rotations act on the same
    function of the inputs, so they combine and one gate goes."""
    example = Circuit(
        [
            Gate("Rz", (1,), 0.31),      # acts on  y
            Gate("CNOT", (0, 1)),
            Gate("Rz", (1,), 0.77),      # acts on  x ^ y
            Gate("CNOT", (0, 1)),
            Gate("Rz", (0,), 1.19),      # acts on  x
            Gate("Rz", (1,), 0.53),      # acts on  y   <- merges with the first
            Gate("CNOT", (1, 0)),
        ],
        2,
    )
    merged = routine4_rotation_merging(example)
    assert sum(g.is_rotation for g in example.gates) == 4
    assert sum(g.is_rotation for g in merged.gates) == 3
    assert _same_unitary(example, merged)
    # the surviving rotation carries the sum of the two angles
    angles = sorted(g.param for g in merged.gates if g.is_rotation)
    assert angles == pytest.approx(sorted([0.31 + 0.53, 0.77, 1.19]))


def test_routine4_keeps_the_survivors_own_wire():
    """Two different wires can carry the same function at the same time, so the
    combined rotation must stay on the wire of the rotation it was merged into,
    not the wire of the one absorbed. Getting this wrong moves a gate."""
    # after CNOT(0,1) then CNOT(0,1), wire 1 carries {1} again -- the same
    # function wire 1 started with, but reached by a different route.
    circuit = Circuit(
        [
            Gate("Rz", (1,), 0.4),
            Gate("CNOT", (0, 1)),
            Gate("CNOT", (0, 1)),
            Gate("Rz", (1,), 0.6),
        ],
        2,
    )
    merged = routine4_rotation_merging(circuit)
    rotations = [g for g in merged.gates if g.is_rotation]
    assert len(rotations) == 1
    assert rotations[0].wires == (1,)
    assert rotations[0].param == pytest.approx(1.0)
    assert _same_unitary(circuit, merged)


@pytest.mark.parametrize("n", [4, 5, 6])
@pytest.mark.parametrize("algo", ["pf2com", "pf4com"])
def test_routine4_alone_preserves_the_unitary(n, algo):
    H = simcount.run(n, "pf2com", "cz", 1).hamiltonian
    sim = simulate(H, algo, float(n), 1e-3, r=3)
    before = bridge.from_tape(sim.step, n)
    after = routine4_rotation_merging(before)
    assert _same_unitary(before, after)
    assert len(after) < len(before)


def test_light_now_runs_every_routine():
    """Routine 4 is implemented, so the full Light sequence needs no skipping."""
    H = simcount.run(4, "pf2com", "cz", 1).hamiltonian
    before = bridge.from_tape(simulate(H, "pf2com", 4.0, 1e-3).step, 4)
    result = optimiser.optimise(before, optimiser.LIGHT, skip_unimplemented=False)
    assert result.skipped == ()
    assert _same_unitary(before, result.circuit)
    assert len(result.circuit) < len(before)


def test_routine4_reduces_rotations_on_the_reference_block():
    """The gate categories Routines 1-3 could not touch. Short of the published
    1.40x because the run-splitting here is conservative -- see the DEVIATION
    note on routine4_rotation_merging."""
    comparison = samples.compare(
        "pf4com", 13, sequence=optimiser.LIGHT, skip_unimplemented=False
    )
    rows = {name: (pre, ours, theirs) for name, pre, ours, theirs in comparison.rows()}
    pre_rz, ours_rz, theirs_rz = rows["Rz"]
    pre_cx, ours_cx, theirs_cx = rows["CNOT"]
    assert ours_rz < pre_rz, "Routine 4 must reduce the rotation count"
    assert ours_cx < pre_cx, "the merge should free CNOT gates for Routine 3"
    assert ours_rz > theirs_rz, "conservative splitting: expected to fall short"
    ok, err = verify.equivalent(
        comparison.pair.pre_core, comparison.result.circuit, trials=2
    )
    assert ok, f"optimiser changed the circuit, error {err:.2e}"


def test_pass_sequences_match_the_paper():
    assert optimiser.LIGHT == (1, 3, 2, 3, 1, 2, 4, 3, 2)
    assert optimiser.HEAVY == (1, 3, 2, 3, 1, 2, 5)


# --- end to end on real circuits ---------------------------------------------

@pytest.mark.parametrize("n", [4, 5, 6])
@pytest.mark.parametrize("algo", ["pf2com", "pf4com"])
def test_light_preserves_our_own_step_circuits(n, algo):
    H = simcount.run(n, "pf2com", "cz", 1).hamiltonian
    sim = simulate(H, algo, float(n), 1e-3, r=3)
    before = bridge.from_tape(sim.step, n)
    result = optimiser.optimise(before, skip_unimplemented=True)
    assert _same_unitary(before, result.circuit)
    assert len(result.circuit) < len(before)


def test_bridge_round_trip():
    H = simcount.run(4, "pf2com", "cz", 1).hamiltonian
    sim = simulate(H, "pf2com", 4.0, 1e-3, r=2)
    circuit = bridge.from_tape(sim.step, 4)
    tape = bridge.to_tape(circuit)
    assert len(tape.operations) == len(circuit)
    assert bridge.from_tape(tape, 4).counts() == circuit.counts()


def test_bridge_rejects_gates_outside_the_set():
    import pennylane as qml

    tape = qml.tape.QuantumScript([qml.RX(0.3, 0)])
    with pytest.raises(ValueError, match="outside the optimiser's gate set"):
        bridge.from_tape(tape, 1)


# --- the reference circuits ---------------------------------------------------

def test_quipper_parses_the_reference_totals():
    """Independently counted from the published ASCII, and equal to what our
    own pipeline produces for the same case."""
    pre = quipper.read(samples.path_for("pf4com", 13, "preoptim"))
    post = quipper.read(samples.path_for("pf4com", 13, "postoptim"))
    assert pre.n_qubits == 13
    assert pre.toplevel == [("S4", 23268)]
    assert sum(pre.total_counts().values()) == 77_133_420
    assert pre.total_counts()["Rz"] == 10_586_940
    assert pre.total_counts()["CNOT"] == 18_149_040
    assert sum(post.total_counts().values()) == 28_736_071
    # Post-optimisation circuits are L . C^m . R
    assert [name for name, _ in post.toplevel] == ["L", "C", "R"]


def test_reference_reduction_factors_match_the_paper():
    """The paper reports ~33.3% CNOT and ~28.5% Rz reduction for PF circuits."""
    targets = samples.load("pf4com", 13).target_factors()
    assert targets["CNOT"] == pytest.approx(1.5, abs=0.01)
    assert targets["Rz"] == pytest.approx(1.4, abs=0.01)


@pytest.mark.parametrize("algo", samples.ALGORITHMS)
def test_optimiser_is_sound_on_the_reference_circuits(algo):
    """Random-state equivalence at n=13, where a full unitary is out of reach."""
    comparison = samples.compare(algo, 13)
    ok, err = verify.equivalent(
        comparison.pair.pre_core, comparison.result.circuit, trials=2
    )
    assert ok, f"{algo}: optimiser changed the circuit, error {err:.2e}"


@pytest.mark.parametrize("algo", samples.ALGORITHMS)
def test_reference_circuits_still_contain_entangling_gates(algo):
    """A Heisenberg simulation cannot be implemented with no CNOT gates.
    Regression for the collapse caused by the two tolerance bugs."""
    comparison = samples.compare(algo, 13)
    assert comparison.result.after.get("CNOT", 0) > 0


# --- Routine 4: per-wire stretches and Clifford-angle handling ----------------

def test_clifford_angles_map_back_to_clifford_gates():
    """A merged angle landing on a multiple of pi/2 must not be emitted as an Rz.
    Leaving it as a rotation would hand a free gate to the synthesis stage to
    approximate, inflating the T-count."""
    from plsimcount.nam.routines import from_rotation

    expected = {0: None, 1: "S", 2: "Z", 3: "Sdg"}
    for turns in range(8):
        gate = from_rotation(turns * np.pi / 2, (0,))
        want = expected[turns % 4]
        assert (gate.name if gate is not None else None) == want, turns
    assert from_rotation(0.37, (0,)).name == "Rz"


def test_s_gates_are_recognised_as_rotations():
    """Section 4.1: the gate set is {NOT, H, Rz, CNOT} and the phase gate is
    rz(pi/2), so S must take part in merging like any other rotation."""
    from plsimcount.nam.routines import as_rotation

    assert as_rotation(Gate("S", (0,))) == pytest.approx(np.pi / 2)
    assert as_rotation(Gate("Sdg", (0,))) == pytest.approx(-np.pi / 2)
    assert as_rotation(Gate("Z", (0,))) == pytest.approx(np.pi)
    assert as_rotation(Gate("Rz", (0,), 0.4)) == pytest.approx(0.4)
    assert as_rotation(Gate("H", (0,))) is None
    # two S gates on the same function combine to a Z, not to an Rz
    circuit = Circuit([Gate("S", (0,)), Gate("CNOT", (0, 1)), Gate("S", (0,))], 2)
    merged = routine4_rotation_merging(circuit)
    assert [g.name for g in merged.gates] == ["Z", "CNOT"]
    assert _same_unitary(circuit, merged)


def test_a_hadamard_only_retires_its_own_wire():
    """The paper grows a stretch outwards per wire rather than ending it at any
    Hadamard. Two rotations on wire 0 must still merge across an H on wire 1."""
    from plsimcount.nam.routines import phase_stretches

    circuit = Circuit(
        [
            Gate("Rz", (0,), 0.3),
            Gate("H", (1,)),          # unrelated wire: must not end the stretch
            Gate("Rz", (0,), 0.4),
        ],
        2,
    )
    assert len(phase_stretches(circuit)) == 1
    merged = routine4_rotation_merging(circuit)
    assert sum(1 for g in merged.gates if g.is_rotation) == 1
    assert _same_unitary(circuit, merged)


def test_a_hadamard_does_end_the_stretch_on_its_own_wire():
    """The converse: a wire that has been retired cannot carry a merge."""
    from plsimcount.nam.routines import phase_stretches

    circuit = Circuit(
        [Gate("Rz", (0,), 0.3), Gate("H", (0,)), Gate("Rz", (0,), 0.4)], 1
    )
    assert phase_stretches(circuit) == []
    merged = routine4_rotation_merging(circuit)
    assert sum(1 for g in merged.gates if g.is_rotation) == 2
    assert _same_unitary(circuit, merged)


def test_no_clifford_angle_rotations_leak_into_the_output():
    """Across a real circuit, every surviving Rz must have a genuinely
    non-Clifford angle."""
    H = simcount.run(4, "pf2com", "cz", 1).hamiltonian
    before = bridge.from_tape(simulate(H, "pf2com", 4.0, 1e-3).step, 4)
    after = optimiser.optimise(before).circuit
    for gate in after.gates:
        if gate.is_rotation:
            quarter_turns = gate.param / (np.pi / 2)
            assert abs(quarter_turns - round(quarter_turns)) > 1e-9, gate


def test_marginal_cost_per_repeated_block_is_the_fair_comparison():
    """Their published circuit is L . C^m . R, so C assumes a neighbour on both
    sides. Optimising one isolated block overstates the per-block cost, and the
    marginal cost of an extra block is the number to compare against C."""
    pre = samples.load("pf4com", 13).pre_core
    sizes = []
    for copies in (2, 3):
        joined = Circuit(list(pre.gates) * copies, pre.n_qubits)
        sizes.append(len(optimiser.optimise(joined).circuit))
    marginal = sizes[1] - sizes[0]
    isolated = len(optimiser.optimise(pre).circuit)
    assert marginal < isolated, "an isolated block cannot merge across its edges"
    # 85% of the published reduction; see README for what has been ruled out.
    theirs = len(samples.load("pf4com", 13).post_core)
    recovered = (len(pre) / marginal) / (len(pre) / theirs)
    assert 0.80 < recovered < 0.90, recovered
