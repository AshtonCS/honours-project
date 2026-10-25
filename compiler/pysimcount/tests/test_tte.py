"""Theory-of-Trotter-error bounds (Childs et al. 2021): validity and regression."""

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from pysimcount import bounds, empirical, simcount, tte  # noqa: E402
from pysimcount.hamiltonian import Hamiltonian, Term, heisenberg  # noqa: E402

FIELDS_N4 = [-0.8444192498333132, 0.7998200058907112, 0.2371705373647761, -0.5823152775681468]


@pytest.fixture(scope="module")
def H4():
    return heisenberg(4, FIELDS_N4)


@pytest.fixture(scope="module")
def ising6():
    # Transverse-field Ising chain: two commuting groups (ZZ..., X...) -> Prop J.1 path
    n = 6
    zz = [Term(-1.0, "".join("Z" if q in (i, i + 1) else "I" for q in range(n))) for i in range(n - 1)]
    x = [Term(-0.7, "".join("X" if q == i else "I" for q in range(n))) for i in range(n)]
    return Hamiltonian(zz + x)


def test_groups_are_xyz(H4):
    assert [len(g) for g in tte.groups(H4)] == [4, 4, 8]


def test_regression_alphas_n4(H4):
    """Prefactors computed independently with PennyLane's PauliSentence."""
    expected = {(1, False): 30.4760, (1, True): 19.7315, (2, False): 51.5620,
                (2, True): 36.0412, (4, False): 5332.6091, (4, True): 4322.6164}
    for (order, exact), v in expected.items():
        assert tte.alpha(H4, order, exact) == pytest.approx(v, abs=1e-3)


@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("d", [0.5, 0.2, 0.1, 0.05, 0.02])
def test_bound_dominates_measured_step_error(H4, order, d):
    ev = empirical.Evolution(H4)
    err = ev.step_error(order, d)
    assert err <= tte.step_error(H4, order, d, exact=True) + 1e-12
    assert tte.step_error(H4, order, d, exact=True) <= tte.step_error(H4, order, d) + 1e-12


@pytest.mark.parametrize("order", [1, 2, 4])
def test_full_evolution_within_eps(H4, order):
    eps, t = 1e-3, 4.0
    r = tte.steps(order, H4, t, eps)
    assert empirical.Evolution(H4).error(order, t, r) <= eps


@pytest.mark.parametrize("order", [1, 2, 4])
def test_tighter_than_simcount_commutator_bound(H4, order):
    r_com = bounds.steps(f"pf{order}com", H4, 4.0, 1e-3)
    r_tte = bounds.steps(f"pf{order}tte", H4, 4.0, 1e-3)
    assert r_tte < r_com


@pytest.mark.parametrize("order", [1, 2, 4])
def test_two_summand_path_on_ising(ising6, order):
    assert len(tte.groups(ising6)) == 2
    ev = empirical.Evolution(ising6)
    for d in (0.3, 0.1, 0.03):
        assert ev.step_error(order, d) <= tte.step_error(ising6, order, d, exact=True) + 1e-12


def test_pauli_algebra_against_dense():
    """[A, B] in the bitmask representation matches the dense commutator
    (compared through singular values, which are basis-order independent)."""
    rng = np.random.default_rng(0)
    n = 4
    words = ["".join(rng.choice(list("IXYZ"), n)) for _ in range(6)]
    A = [Term(float(rng.normal()), w) for w in words[:3]]
    B = [Term(float(rng.normal()), w) for w in words[3:]]
    HA, HB = Hamiltonian(A).matrix(), Hamiltonian(B).matrix()
    dense = HA @ HB - HB @ HA
    C = tte.commutator(tte._sentence(A), tte._sentence(B))
    sv_dense = np.sort(np.linalg.svd(dense, compute_uv=False))
    M = tte._matrix(C).toarray()
    # the sparse matrix lives on the support only; pad its singular values with zeros
    sv_sparse = np.sort(np.concatenate([np.linalg.svd(M, compute_uv=False), np.zeros(16 - M.shape[0])]))
    assert np.allclose(sv_sparse, sv_dense, atol=1e-10)
    assert tte.norm(C) == pytest.approx(np.linalg.norm(dense, 2), rel=1e-9)


# --------------------------------------------------------------- regressions

@pytest.mark.parametrize("scale", [1.0, 1e-2, 1e-4, 1e-6, 1e-8])
@pytest.mark.parametrize("order", [1, 2, 4])
def test_alpha_scales_with_coupling(scale, order):
    """alpha_p must scale as coupling^(p+1).

    Regression: commutator() originally pruned terms with |coeff| <= 1e-14
    *absolutely*, which silently deleted every term for small couplings and
    drove alpha (and hence r) to zero -- unsound in the dangerous direction.
    """
    base = tte.alpha(heisenberg(4, [0.3] * 4), order)
    H = heisenberg(4, [0.3 * scale] * 4, coupling=(scale,) * 3)
    assert tte.alpha(H, order) == pytest.approx(base * scale ** (order + 1), rel=1e-9)


@pytest.mark.parametrize("order", [1, 2, 4])
def test_returned_r_satisfies_its_own_bound(order):
    """steps() inverts a power law with ceil(); check no off-by-one at the
    rounding boundary for a spread of epsilons."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        H = heisenberg(4, rng.uniform(-1, 1, 4))
        for eps in (1e-3, 1e-4, float(10 ** rng.uniform(-6, -2))):
            r = tte.steps(order, H, 4.0, eps)
            assert r * tte.step_error(H, order, 4.0 / r) <= eps * (1 + 1e-12)


def test_group_count_limits_fourth_order():
    """Prop J.1/J.2 are tabulated for 2 or 3 groups only; anything else must
    fail loudly rather than silently use the wrong table. PennyLane's native
    bond-major Heisenberg ordering gives 4 groups."""
    import pennylane as qml

    from pysimcount import from_pennylane

    H = from_pennylane(qml.spin.heisenberg(lattice="chain", n_cells=[4], coupling=[1.0, 1.0, 1.0],
                                           boundary_condition=True))
    assert len(tte.groups(H)) == 4
    tte.alpha(H, 2)  # any number of groups is fine for Props 15/16
    with pytest.raises(ValueError, match="2 or 3 grouped summands"):
        tte.alpha(H, 4)
