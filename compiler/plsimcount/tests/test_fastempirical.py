"""Fast brute-force r (fastempirical.py) against the slow reference (empirical.py)."""

import math
import os
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from plsimcount import empirical, fastempirical as F, formula, simcount, weighted as W  # noqa: E402
from plsimcount.hamiltonian import heisenberg  # noqa: E402

H4 = heisenberg(4, [0.3, -0.7, 0.5, 0.1])


@pytest.mark.parametrize("order", [1, 2, 4, 6])
@pytest.mark.parametrize("palindrome", [True, False])
def test_step_matches_dense(order, palindrome):
    fe, ev = F.FastEvolution(H4), empirical.Evolution(H4)
    a = fe.step(formula.schedule(order, H4, 0.37), use_palindrome=palindrome)
    assert np.allclose(a, ev.step(order, 0.37), atol=1e-13)


def test_exact_matches_expm():
    fe, ev = F.FastEvolution(H4), empirical.Evolution(H4)
    assert np.allclose(fe.exact(4.0), ev.exact(4.0), atol=1e-12)


def test_palindrome_not_used_for_first_order():
    """Lie-Trotter is not symmetric: the full product must be used (and is)."""
    fe = F.FastEvolution(H4)
    blocks = W.commuting_blocks(H4, formula.schedule(1, H4, 0.3))
    assert not fe._is_palindrome(blocks)


def test_odd_y_disables_palindrome():
    from plsimcount.hamiltonian import Hamiltonian, Term
    H = Hamiltonian([Term(1.0, "XY"), Term(0.5, "ZI")])
    assert not F.FastEvolution(H).all_symmetric


@pytest.mark.parametrize("order", [1, 2, 4])
def test_r_matches_empirical_steps(order):
    """Same r as the slow reference on simcount's own n = 4 instance (t = n)."""
    H = simcount.run(4, "pf2emp", "cz", 1).hamiltonian
    slow = empirical.steps(order, H, 4.0, 1e-3)
    fast = F.steps(F.FastEvolution(H), lambda H_, d: formula.schedule(order, H_, d), order, 4.0, 1e-3)
    assert fast.r == slow
    assert fast.err_r <= 1e-3 < fast.err_r_minus_1


def _slow_r(H, schedule_fn, t, eps):
    """empirical.steps' search, for an arbitrary schedule, with dense expm."""
    from scipy.linalg import expm
    ev = empirical.Evolution(H)

    def err(r):
        U = np.eye(2 ** H.n_qubits, dtype=complex)
        for j, time in schedule_fn(H, t / r):
            U = expm(-1j * time * H.terms[j].coeff * ev.mats[j]) @ U
        return float(np.linalg.norm(np.linalg.matrix_power(U, r) - ev.exact(t), 2))

    lo, hi = 0, 1
    while err(hi) > eps:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if err(mid) <= eps:
            hi = mid
        else:
            lo = mid
    return hi


@pytest.mark.parametrize("key", ["mo-q6", "mo-q14"])
def test_r_matches_slow_search_for_paper_schemes(key):
    from plsimcount.pfcompare import formula_registry
    spec = formula_registry()[key]
    H = simcount.run(4, "pf2emp", "cz", 1).hamiltonian
    fast = F.steps(F.FastEvolution(H), spec.schedule, spec.order, 4.0, 1e-3)
    assert fast.r == _slow_r(H, spec.schedule, 4.0, 1e-3)


def test_start_guess_does_not_change_answer():
    H = simcount.run(4, "pf2emp", "cz", 1).hamiltonian
    fe = F.FastEvolution(H)
    fn = lambda H_, d: formula.schedule(2, H_, d)  # noqa: E731
    rs = {F.steps(fe, fn, 2, 4.0, 1e-3, start=s).r for s in (1, 7, 50, 400, 5000)}
    assert len(rs) == 1


@pytest.mark.skipif(not os.environ.get("PLSIMCOUNT_SLOW"), reason="~45 s; set PLSIMCOUNT_SLOW=1")
@pytest.mark.parametrize("order,want", [(2, 3174), (4, 152)])
def test_n10_regression(order, want):
    """compiler/bounds_n10_empirical.txt (empirical.py, n = 10, seed 1, t = 10, eps = 1e-3)."""
    H = simcount.run(10, "pf2emp", "cz", 1).hamiltonian
    res = F.steps(F.FastEvolution(H), lambda H_, d: formula.schedule(order, H_, d), order, 10.0, 1e-3)
    assert res.r == want
