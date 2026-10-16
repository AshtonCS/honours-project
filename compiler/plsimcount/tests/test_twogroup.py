"""Full two-group option: exact blocks, correct grouping, correct formula."""

import warnings

import numpy as np
import pennylane as qml
import pytest
from scipy.linalg import expm

warnings.filterwarnings("ignore")

from plsimcount import twogroup  # noqa: E402
from plsimcount.hamiltonian import Hamiltonian, heisenberg  # noqa: E402

FIELDS_N4 = [-0.8444192498333132, 0.7998200058907112, 0.2371705373647761, -0.5823152775681468]
FIELDS_N6 = [0.31, -0.72, 0.05, 0.88, -0.43, 0.19]  # arbitrary, hand-picked


def _dist_up_to_phase(U, V):
    ph = np.vdot(V.ravel(), U.ravel())
    return np.linalg.norm(U - (ph / abs(ph)) * V, 2)


@pytest.mark.parametrize("q0,q1", [(0, 1), (2, 3), (0, 3)])
@pytest.mark.parametrize("seed", range(6))
def test_field_block_is_exact(q0, q1, seed):
    rng = np.random.default_rng(seed)
    J, Jz, h0, h1 = rng.uniform(-1.5, 1.5, 4)
    tau = rng.uniform(-1, 1)
    bond = twogroup.Bond(q0, q1, J, J, Jz, h0, h1)
    G = Hamiltonian(bond.terms(4)).matrix()
    U = qml.matrix(qml.tape.QuantumScript(twogroup.field_block(bond, tau)), wire_order=range(4))
    assert _dist_up_to_phase(U, expm(-1j * tau * G)) < 1e-11


@pytest.mark.parametrize("h0,h1", [(0.3, 0.3), (0.0, 0.7), (0.5, -0.5)])
def test_field_block_edge_cases(h0, h1):
    """Equal fields (delta = 0: diagonal Euler branch) and one-sided fields."""
    bond = twogroup.Bond(0, 1, 1.0, 1.0, 1.0, h0, h1)
    for tau in (0.1, np.pi / 4, 1.3):
        G = Hamiltonian(bond.terms(2)).matrix()
        U = qml.matrix(qml.tape.QuantumScript(twogroup.field_block(bond, tau)), wire_order=range(2))
        assert _dist_up_to_phase(U, expm(-1j * tau * G)) < 1e-11


def test_field_block_counts():
    ops = twogroup.field_block(twogroup.Bond(0, 1, 1.0, 1.0, 1.0, 0.2, -0.4), 0.1)
    names = [op.name for op in ops]
    assert names.count("CNOT") == 3
    assert names.count("RZ") == 7


def test_field_block_rejects_anisotropic_xy():
    with pytest.raises(ValueError, match="Jx == Jy"):
        twogroup.field_block(twogroup.Bond(0, 1, 1.0, 0.5, 1.0, 0.2, 0.1), 0.1)


def test_split_groups_n4():
    even, odd = twogroup.split_groups(heisenberg(4, FIELDS_N4))
    assert [(b.q0, b.q1) for b in even] == [(0, 1), (2, 3)]
    assert [(b.q0, b.q1) for b in odd] == [(1, 2), (0, 3)]  # (0, 3) is the wrap bond
    assert [(b.h0, b.h1) for b in even] == [(FIELDS_N4[0], FIELDS_N4[1]), (FIELDS_N4[2], FIELDS_N4[3])]
    assert all(b.h0 == b.h1 == 0 for b in odd)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_split_rejects_unsupported_sizes(n):
    with pytest.raises(ValueError, match="even n >= 4"):
        twogroup.split_groups(heisenberg(n, [0.1] * n))


def test_groups_sum_to_H():
    H = heisenberg(6, FIELDS_N6)
    for outer in twogroup.OUTER:
        assert np.allclose(sum(twogroup._group_matrices(H, outer)), H.matrix())


@pytest.mark.parametrize("outer", twogroup.OUTER)
@pytest.mark.parametrize("n,fields,order", [(4, FIELDS_N4, 1), (4, FIELDS_N4, 2), (4, FIELDS_N4, 4),
                                            (6, FIELDS_N6, 2)])
def test_circuit_matches_group_product(outer, n, fields, order):
    H = heisenberg(n, fields)
    d = 0.37
    U_c = qml.matrix(twogroup.build_step(H, order, d, outer), wire_order=range(n))
    U_x = twogroup.step_unitary(twogroup._group_matrices(H, outer), order, d)
    assert _dist_up_to_phase(U_c, U_x) < 1e-10


@pytest.mark.parametrize("outer", twogroup.OUTER)
def test_second_order_scaling(outer):
    H = heisenberg(4, FIELDS_N4)
    mats = twogroup._group_matrices(H, outer)

    def err(d):
        return np.linalg.norm(twogroup.step_unitary(mats, 2, d) - expm(-1j * d * H.matrix()), 2)

    assert err(0.02) / err(0.01) == pytest.approx(8.0, rel=0.02)


@pytest.mark.parametrize("outer,rot", [("even", 34), ("odd", 26)])
def test_hand_count_pf2_n4(outer, rot):
    """S2, n=4: 4.5n = 18 CNOTs either way. Rotations: even-outer has 4 field
    blocks (7 each) + 2 odd blocks (3 each) = 34 = 8.5n; odd-outer has
    2 field blocks + 4 odd blocks = 26 = 6.5n."""
    names = [op.name for op in twogroup.build_step(heisenberg(4, FIELDS_N4), 2, 0.1, outer).operations]
    assert names.count("CNOT") == 18
    assert names.count("RZ") == rot


@pytest.mark.parametrize("outer", twogroup.OUTER)
def test_empirical_r_is_minimal(outer):
    H = heisenberg(4, FIELDS_N4)
    t, eps = 4.0, 1e-3
    r = twogroup.steps_empirical(H, 2, t, eps, outer)
    mats = twogroup._group_matrices(H, outer)
    exact = expm(-1j * t * H.matrix())

    def err(k):
        return np.linalg.norm(np.linalg.matrix_power(twogroup.step_unitary(mats, 2, t / k), k) - exact, 2)

    assert err(r) <= eps < err(r - 1)


@pytest.mark.parametrize("outer", twogroup.OUTER)
@pytest.mark.parametrize("order", [1, 2, 4])
def test_tte_bound_holds(outer, order):
    H = heisenberg(4, FIELDS_N4)
    r = twogroup.steps_tte(H, order, 4.0, 1e-3, outer)
    mats = twogroup._group_matrices(H, outer)
    U = np.linalg.matrix_power(twogroup.step_unitary(mats, order, 4.0 / r), r)
    assert np.linalg.norm(U - expm(-4j * H.matrix()), 2) <= 1e-3
