"""Full two-group option: exact blocks, correct grouping, correct formula."""

import warnings

import numpy as np
import pennylane as qml
import pytest
from scipy.linalg import expm

warnings.filterwarnings("ignore")

from pysimcount import twogroup  # noqa: E402
from pysimcount.hamiltonian import Hamiltonian, heisenberg  # noqa: E402

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


# ------------------------------------------------------------------ any formula as two groups

def _registry():
    from pysimcount.pfcompare import formula_registry
    return formula_registry()


def _sched_unitary(mats, sched):
    U = np.eye(mats[0].shape[0], dtype=complex)
    for g, time in sched:
        U = expm(-1j * time * mats[g]) @ U
    return U


@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_formula_group_schedule_reproduces_suzuki_recursion(order):
    """The stand-in Hamiltonian gives exactly group_schedule for Suzuki."""
    key = "lie-trotter" if order == 1 else f"suzuki{order}"
    got = twogroup.formula_group_schedule(_registry()[key], 0.37)
    want = twogroup.group_schedule(order, 0.37)
    assert [g for g, _ in got] == [g for g, _ in want]
    assert np.allclose([x for _, x in got], [x for _, x in want], rtol=1e-14, atol=0)


@pytest.mark.parametrize("key", ["lie-trotter", "suzuki2", "suzuki4", "suzuki6", "suzuki8", "mo-q6", "mo-q14",
                                 "bm-q10", "mo-o8-q21", "morales-y8m10", "morales-m8"])
def test_every_formula_two_group_total_time_and_merge(key):
    """Each group gets total time d per step, and merging consecutive
    applications of one group leaves the unitary unchanged."""
    spec = _registry()[key]
    d = 0.37
    raw = twogroup.formula_group_schedule(spec, d)
    mer = twogroup.formula_group_schedule(spec, d, merged=True)
    for g in (0, 1):
        assert sum(x for gg, x in raw if gg == g) == pytest.approx(d, rel=1e-12)
    assert all(a[0] != b[0] for a, b in zip(mer, mer[1:]))  # no two consecutive applications of one group
    mats = twogroup._group_matrices(heisenberg(4, FIELDS_N4), "even")
    assert _dist_up_to_phase(_sched_unitary(mats, raw), _sched_unitary(mats, mer)) < 1e-12


@pytest.mark.parametrize("key", ["lie-trotter", "suzuki2", "suzuki4", "mo-q6", "mo-q14", "bm-q10",
                                 "mo-o8-q21", "morales-y8m10", "morales-m8"])
def test_every_formula_two_group_has_its_order(key):
    """Local error of one two-group step falls as d^(p+1): halving d divides
    it by about 2^(p+1). Checked through the local slope between d and d/2."""
    spec = _registry()[key]
    H = heisenberg(4, FIELDS_N4)
    mats = twogroup._group_matrices(H, "even")
    Hm = H.matrix()

    def err(d):
        return np.linalg.norm(_sched_unitary(mats, twogroup.formula_group_schedule(spec, d)) - expm(-1j * d * Hm), 2)

    d = {1: 0.02, 2: 0.04, 4: 0.12, 6: 0.25, 8: 0.35}[spec.order]
    slope = np.log2(err(d) / err(d / 2))
    assert slope == pytest.approx(spec.order + 1, abs=0.35), (key, slope)


@pytest.mark.parametrize("key", ["mo-q6", "morales-m8"])
@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("outer", twogroup.OUTER)
def test_any_formula_circuit_matches_group_product(key, merged, outer):
    H = heisenberg(4, FIELDS_N4)
    sched = twogroup.formula_group_schedule(_registry()[key], 0.37, merged)
    U_c = qml.matrix(twogroup.build_step_schedule(H, sched, outer), wire_order=range(4))
    U_x = _sched_unitary(twogroup._group_matrices(H, outer), sched)
    assert _dist_up_to_phase(U_c, U_x) < 1e-10
