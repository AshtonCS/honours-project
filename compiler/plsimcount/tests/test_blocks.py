"""Bond blocks: the 3-CNOT circuit is exact, and the blocked step is the
same unitary as the term-by-term even/odd step."""

import warnings

import numpy as np
import pennylane as qml
import pytest
from scipy.linalg import expm

warnings.filterwarnings("ignore")

from plsimcount import blocks, evenodd, formula  # noqa: E402
from plsimcount.circuit import build_step  # noqa: E402
from plsimcount.hamiltonian import Hamiltonian, Term, heisenberg  # noqa: E402

FIELDS_N4 = [-0.8444192498333132, 0.7998200058907112, 0.2371705373647761, -0.5823152775681468]
FIELDS_N6 = [0.31, -0.72, 0.05, 0.88, -0.43, 0.19]  # arbitrary, hand-picked


def _dist_up_to_phase(U, V):
    ph = np.vdot(V.ravel(), U.ravel())
    return np.linalg.norm(U - (ph / abs(ph)) * V, 2)


def _word(n, placements):
    return "".join(placements.get(q, "I") for q in range(n))


@pytest.mark.parametrize("q0,q1", [(0, 1), (1, 2), (0, 3)])  # (0, 3) = wrap bond on 4 qubits
@pytest.mark.parametrize("seed", range(5))
def test_block_is_exact(q0, q1, seed):
    a, b, c = np.random.default_rng(seed).uniform(-2, 2, 3)
    n = 4
    G = Hamiltonian([Term(a, _word(n, {q0: "X", q1: "X"})), Term(b, _word(n, {q0: "Y", q1: "Y"})),
                     Term(c, _word(n, {q0: "Z", q1: "Z"}))]).matrix()
    U = qml.matrix(qml.tape.QuantumScript(blocks.canonical_block(q0, q1, a, b, c)), wire_order=range(n))
    assert _dist_up_to_phase(U, expm(-1j * G)) < 1e-12


def test_block_gate_counts():
    """3 CNOT and 3 genuine rotations, against 6 and 3 term by term."""
    ops = blocks.canonical_block(0, 1, 0.1, 0.2, 0.3)
    names = [op.name for op in ops]
    assert names.count("CNOT") == 3
    assert names.count("RZ") == 3
    assert sorted(float(op.data[0]) for op in ops if op.name == "RZ") == pytest.approx([-0.2, 0.4, 0.6])


@pytest.mark.parametrize("n,fields,order", [(4, FIELDS_N4, 1), (4, FIELDS_N4, 2), (4, FIELDS_N4, 4),
                                            (6, FIELDS_N6, 2)])
def test_blocked_step_matches_termwise(n, fields, order):
    H = evenodd.even_odd(heisenberg(n, fields))
    d = 0.37
    blocked, _ = blocks.build_step_blocked(H, order, d)
    U_b = qml.matrix(blocked, wire_order=range(n))
    U_t = qml.matrix(build_step(H, order, d), wire_order=range(n))
    assert _dist_up_to_phase(U_b, U_t) < 1e-10


def test_hand_count_pf2_n4():
    """n=4, S2: even group (2 bonds) twice at d/2, odd group (2 bonds) twice
    at d/2, fields once -> 8 blocks = 24 CNOTs, where term-by-term has
    12 bond terms x 2 applications x 2 CNOTs = 48."""
    H = evenodd.even_odd(heisenberg(4, FIELDS_N4))
    tape, n_blocks = blocks.build_step_blocked(H, 2, 0.1)
    assert n_blocks == 8
    assert sum(op.name == "CNOT" for op in tape.operations) == 24
    assert sum(op.name == "CNOT" for op in build_step(H, 2, 0.1).operations) == 48
    # rotations unchanged: 12 bond terms x 2 + 4 fields
    assert sum(op.name == "RZ" for op in tape.operations) == 28


def test_xyz_ordering_is_rejected():
    """In simcount's ordering a bond's XX, YY, ZZ are never in one commuting
    run, so blocking must fail loudly rather than quietly fall back."""
    H = heisenberg(4, FIELDS_N4)
    with pytest.raises(ValueError, match="complete"):
        blocks.build_step_blocked(H, 2, 0.1)


def test_nam_preserves_blocked_step():
    from plsimcount.nam import bridge, optimise, verify

    H = evenodd.even_odd(heisenberg(4, FIELDS_N4))
    tape, _ = blocks.build_step_blocked(H, 2, 0.3)
    c = bridge.from_tape(tape, 4)
    ok, err = verify.equivalent(c, optimise(c).circuit)
    assert ok, err


def test_schedule_is_unchanged():
    """Blocking regroups the existing schedule; it must not alter it."""
    H = evenodd.even_odd(heisenberg(4, FIELDS_N4))
    assert len(formula.schedule(4, H, 0.1)) == 5 * (2 * 12 + 4)
