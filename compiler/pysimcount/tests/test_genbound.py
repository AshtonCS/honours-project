"""genbound: the norm bound for any product formula.

1. With budget="count", constant="childs" it must reproduce simcount's
   ana/min (bounds.py) exactly for Lie-Trotter and Suzuki S2-S8.
2. A worked example by hand.
3. The per-step bound (*) must hold against exact diagonalisation for every
   formula in the registry, both budgets and both constants.
"""

import math

import pytest

from pysimcount import bounds, genbound
from pysimcount.hamiltonian import heisenberg, Hamiltonian, Term
from pysimcount.pfcompare import formula_registry, numpy_fields
from pysimcount.pfexact import exact_unitary, formula_step, spectral_norm

REG = formula_registry()
SUZUKI = {"lie-trotter": 1, "suzuki2": 2, "suzuki4": 4, "suzuki6": 6, "suzuki8": 8}


@pytest.mark.parametrize("key", list(SUZUKI))
@pytest.mark.parametrize("kind", ["min", "ana"])
def test_reproduces_simcount_for_suzuki(key, kind):
    ref = bounds.minimised if kind == "min" else bounds.analytic
    order = SUZUKI[key]
    spec = REG[key]
    cases = 0
    for n in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 50, 100, 200, 500):
        for seed in (1, 2):
            H = heisenberg(n, numpy_fields(n, seed))
            for t, eps in ((n, 1e-3), (n, 5e-4), (0.37 * n, 1e-2), (n, 1e-6)):
                assert genbound.steps(spec, H, t, eps, kind=kind) == ref(order, H, t, eps), (n, seed, t, eps)
                cases += 1
    # non-unit Lambda: couplings 1.7, so max_coeff = 1.7
    H = heisenberg(6, numpy_fields(6, 3), coupling=(1.7, 1.7, 1.7))
    assert H.max_coeff == 1.7
    assert genbound.steps(spec, H, 6, 1e-3, kind=kind) == ref(order, H, 6, 1e-3)
    assert cases == 16 * 2 * 4


def test_worked_example_lie_trotter():
    # Two commuting-free terms X, Z with coefficients 1, so m = 2, Lambda = 1;
    # t = 1, eps = 0.1. Count budget: L t = 1 * 2 * 1 * 1 = 2.
    H = Hamiltonian([Term(1.0, "X"), Term(1.0, "Z")], split=1, name="xz")
    spec = REG["lie-trotter"]
    # ana: max(2, e * 2^2 / 0.1) = 108.73... -> 109
    assert genbound.steps(spec, H, 1.0, 0.1, kind="ana") == 109
    # min: smallest r with 4/r * e^(2/r) <= 0.1. r = 41: 4/41 e^(2/41) = 0.10244;
    # r = 42: 4/42 e^(2/42) = 0.09988.
    assert 4 / 41 * math.exp(2 / 41) > 0.1 > 4 / 42 * math.exp(2 / 42)
    assert genbound.steps(spec, H, 1.0, 0.1, kind="min") == 42


def test_worked_example_suzuki4_budgets():
    # S4 on the same H: count budget 2 * 5 sweeps * m = 2 -> L = 20.
    H = Hamiltonian([Term(1.0, "X"), Term(1.0, "Z")], split=1, name="xz")
    spec = REG["suzuki4"]
    assert genbound.count_rate(spec, H) == 20
    # exact budget: sum |w| per term = 4 p + |1 - 4 p| = 8 p - 1, p = 1/(4 - 4^(1/3)).
    p = 1 / (4 - 4 ** (1 / 3))
    assert genbound.exact_rate(spec, H) == pytest.approx(2 * (8 * p - 1), rel=1e-14)


@pytest.mark.parametrize("key", list(REG))
def test_exact_budget_never_exceeds_count(key):
    for n in (4, 6, 10):
        H = heisenberg(n, numpy_fields(n, 1))
        assert genbound.exact_rate(REG[key], H) <= genbound.count_rate(REG[key], H)


def test_count_raises_when_a_weight_exceeds_one():
    from dataclasses import replace
    from pysimcount import formula

    spec = replace(REG["suzuki2"], schedule=lambda H, d: formula.second_order(H, 3 * d))
    H = heisenberg(4, numpy_fields(4, 1))
    with pytest.raises(ValueError, match="counting argument"):
        genbound.count_rate(spec, H)


@pytest.mark.parametrize("key", list(REG))
def test_per_step_bound_holds_exact_diagonalisation(key):
    """||S(tau) - e^{-i tau H}|| <= c_p s^(p+1) e^s on n = 4 and 5, at step
    sizes where the tightest bound sits between 1e-9 (well above roundoff)
    and 1."""
    spec = REG[key]
    checked = 0
    for n in (4, 5):
        H = heisenberg(n, numpy_fields(n, 1))
        rate = genbound.exact_rate(spec, H)
        for x in (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0):
            tau = x / rate  # exact budget s = x
            tight = genbound.step_error_bound(spec, H, tau, budget="exact", constant="taylor")
            if not 1e-9 <= tight <= 1:
                continue
            err = spectral_norm(formula_step(H, spec.schedule(H, tau)) - exact_unitary(H, tau))
            for budget in genbound.BUDGETS:
                for constant in genbound.CONSTANTS:
                    bound = genbound.step_error_bound(spec, H, tau, budget=budget, constant=constant)
                    assert err <= bound, (n, x, budget, constant, err, bound)
            checked += 1
    assert checked >= 2


def test_rejects_bad_inputs():
    H = heisenberg(4, numpy_fields(4, 1))
    spec = REG["suzuki4"]
    for kw in ({"kind": "max"}, {"budget": "tight"}, {"constant": "one"}):
        with pytest.raises(ValueError):
            genbound.steps(spec, H, 4, 1e-3, **kw)
    with pytest.raises(ValueError):
        genbound.steps(spec, H, 4, 0.0)
