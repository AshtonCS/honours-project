"""Even/odd bond grouping: same operator, three commuting groups, valid bounds."""

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from plsimcount import empirical, evenodd, tte  # noqa: E402
from plsimcount.hamiltonian import heisenberg  # noqa: E402

FIELDS_N4 = [-0.8444192498333132, 0.7998200058907112, 0.2371705373647761, -0.5823152775681468]
FIELDS_N6 = [0.31, -0.72, 0.05, 0.88, -0.43, 0.19]  # arbitrary, hand-picked


@pytest.fixture(scope="module")
def H4():
    return evenodd.even_odd(heisenberg(4, FIELDS_N4))


def test_same_operator(H4):
    H = heisenberg(4, FIELDS_N4)
    assert sorted(H.terms, key=lambda t: t.word) == sorted(H4.terms, key=lambda t: t.word)
    assert np.allclose(H.matrix(), H4.matrix())


def test_hand_checked_groups_n4(H4):
    """n=4 periodic: even bonds (0,1),(2,3); odd bonds (1,2) and the wrap (3,0)."""
    groups = tte.groups(H4)
    assert [len(g) for g in groups] == [6, 6, 4]
    assert {t.word for t in groups[0]} == {"XXII", "YYII", "ZZII", "IIXX", "IIYY", "IIZZ"}
    assert {t.word for t in groups[1]} == {"IXXI", "IYYI", "IZZI", "XIIX", "YIIY", "ZIIZ"}
    assert {t.word for t in groups[2]} == {"ZIII", "IZII", "IIZI", "IIIZ"}
    assert H4.split == 12


@pytest.mark.parametrize("n,fields", [(4, FIELDS_N4), (6, FIELDS_N6)])
def test_groups_commute_exactly(n, fields):
    """Every pair inside a group commutes, checked with dense matrices rather
    than the Pauli bitmask algebra the grouping itself uses."""
    from plsimcount.hamiltonian import Hamiltonian

    for g in tte.groups(evenodd.even_odd(heisenberg(n, fields))):
        for a in g:
            for b in g:
                A, B = Hamiltonian([a]).matrix(), Hamiltonian([b]).matrix()
                assert np.allclose(A @ B, B @ A)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_rejects_unsupported_sizes(n):
    with pytest.raises(ValueError, match="even n >= 4"):
        evenodd.even_odd(heisenberg(n, [0.1] * n))


def test_second_order_scaling(H4):
    """Halving the step divides the S2 one-step error by ~2^3 = 8."""
    ev = empirical.Evolution(H4)
    ratio = ev.step_error(2, 0.02) / ev.step_error(2, 0.01)
    assert ratio == pytest.approx(8.0, rel=0.02)


@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("d", [0.5, 0.2, 0.1, 0.05])
def test_bound_dominates_measured_step_error(H4, order, d):
    err = empirical.Evolution(H4).step_error(order, d)
    assert err <= tte.step_error(H4, order, d, exact=True) + 1e-12


@pytest.mark.parametrize("order", [1, 2, 4])
def test_full_evolution_within_eps(H4, order):
    eps, t = 1e-3, 4.0
    r = tte.steps(order, H4, t, eps)
    assert empirical.Evolution(H4).error(order, t, r) <= eps
