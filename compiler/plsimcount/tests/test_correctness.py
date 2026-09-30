"""Physics-level checks that do not depend on simcount.

These run the product formula as an actual unitary on small systems and
verify that the implementation does what the bounds promise.
"""

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from plsimcount import bounds, heisenberg, simulate  # noqa: E402
from plsimcount import synthesis  # noqa: E402

FIELDS_N4 = [-0.8444192498333132, 0.7998200058907112, 0.2371705373647761, -0.5823152775681468]


@pytest.fixture(scope="module")
def H4():
    return heisenberg(4, FIELDS_N4)


@pytest.mark.parametrize("algo", ["pf1com", "pf2com", "pf4com", "pf2min", "pf4ana", "pf6min"])
def test_trotter_error_within_bound(H4, algo):
    """||S(t/r)^r - exp(-iHt)|| <= eps for the r chosen by the bound."""
    eps = 1e-3
    sim = simulate(H4, algo, t=4.0, eps_trotter=eps)
    err = sim.trotter_error()
    assert err <= eps, f"{algo}: error {err:.3e} exceeds {eps}"
    # ... and the bound is not absurdly loose in the wrong direction:
    assert err > 0


@pytest.mark.parametrize("algo,r", [("pf1com", 50), ("pf2com", 20), ("pf4com", 40)])
def test_error_scales_with_order(H4, algo, r):
    """Halving the step should reduce the error by ~2^order."""
    order = int(algo[2])
    e1 = simulate(H4, algo, t=4.0, eps_trotter=1, r=r).trotter_error()
    e2 = simulate(H4, algo, t=4.0, eps_trotter=1, r=2 * r).trotter_error()
    ratio = e1 / e2
    assert 0.6 * 2**order < ratio < 1.5 * 2**order, f"{algo}: ratio {ratio:.2f}"


def test_templates_agree(H4):
    """simcount's gate template and PennyLane's PauliRot give the same unitary."""
    a = simulate(H4, "pf2com", t=4.0, eps_trotter=1, r=7, template="simcount").step_matrix()
    b = simulate(H4, "pf2com", t=4.0, eps_trotter=1, r=7, template="pennylane").step_matrix()
    # equal up to global phase
    phase = np.exp(1j * np.angle(np.trace(b.conj().T @ a)))
    assert np.allclose(a, phase * b, atol=1e-10)


def test_second_order_step_is_palindromic(H4):
    from plsimcount import formula

    sched = formula.second_order(H4, 0.1)
    k = H4.split
    head, mid, tail = sched[:k], sched[k:-k], sched[-k:]
    assert [j for j, _ in head] == [j for j, _ in reversed(tail)]
    assert all(t == pytest.approx(0.05) for _, t in head + tail)
    assert [j for j, _ in mid] == list(range(k, H4.n_terms))
    assert all(t == pytest.approx(0.1) for _, t in mid)
    assert sum(t for _, t in sched) == pytest.approx(0.1 * H4.n_terms)


def test_pygridsynth_meets_epsilon_at_project_precision():
    """The regime PennyLane's own rs_decomposition silently fails in."""
    eps = 0.0005 / 213612
    for theta in (-0.0006106334015657589, 0.0005243151133831433, 0.3, 1.0):
        word = synthesis.synthesise(theta, eps)  # raises if error > eps
        assert 80 <= word.count("T") <= 100


def test_synthesised_step_error(H4):
    """Synthesising every RZ in a step with eps_rot keeps the step within
    n_rz * eps_rot of the exact step."""
    sim = simulate(H4, "pf2com", t=4.0, eps_trotter=1e-3, r=100)
    eps_rot = 1e-6
    ct = synthesis.synthesise_tape(sim.step, eps_rot)
    import pennylane as qml

    U = qml.matrix(ct, wire_order=range(4))
    V = sim.step_matrix()
    phase = np.exp(1j * np.angle(np.trace(V.conj().T @ U)))
    err = np.linalg.norm(V - phase * U, 2)
    assert err <= sim.n_rz_per_step * eps_rot
    assert all(op.name != "RZ" for op in ct.operations)


def test_exponential_search_matches_haskell_semantics():
    assert bounds.exponential_search(lambda r: r >= 1) == 1
    assert bounds.exponential_search(lambda r: r >= 5) == 5
    assert bounds.exponential_search(lambda r: r >= 1024) == 1024
    assert bounds.exponential_search(lambda r: r >= 1025) == 1025


def test_heisenberg_matches_simcount_structure():
    """heisenberg() is not exercised by the parity tests (those parse the
    Hamiltonian out of simcount's own output), so check it directly."""
    from plsimcount import simcount

    for n in (2, 4, 5, 6, 7, 10, 13, 20):
        ref = simcount.run(n, "pf2com", "cz", 1).hamiltonian
        mine = heisenberg(n, [t.coeff for t in ref.terms[-n:]])
        assert [t.word for t in mine.terms] == [t.word for t in ref.terms], n
        assert np.allclose([t.coeff for t in mine.terms], [t.coeff for t in ref.terms])
        assert mine.split == ref.split, n


@pytest.mark.parametrize("k", [1, 2, 3])
def test_optimise_preserves_the_unitary(H4, k):
    """The optimiser must not change what the circuit computes."""
    import pennylane as qml

    from plsimcount import optimise

    sim = simulate(H4, "pf2com", t=4.0, eps_trotter=1e-3, r=6)
    raw = optimise.repeat(sim.step, k)
    opt = optimise.optimise_tape(raw)
    U = qml.matrix(raw, wire_order=range(4))
    V = qml.matrix(opt, wire_order=range(4))
    phase = np.exp(1j * np.angle(np.trace(V.conj().T @ U)))
    assert np.allclose(U, phase * V, atol=1e-10)
    assert len(opt.operations) < len(raw.operations)


def test_synthesised_step_meets_budget_at_production_epsilon(H4):
    """The n=4 production case: eps_rot ~ 2.3e-9 over 213612 rotations."""
    from plsimcount import synthesis

    sim = simulate(H4, "pf2com", t=4.0, eps_trotter=5e-4)
    ct, eps_rot = synthesis.synthesise_step(sim, 5e-4)
    import pennylane as qml

    err = np.linalg.norm(sim.step_matrix() - qml.matrix(ct, wire_order=range(4)), 2)
    assert err <= sim.n_rz_per_step * eps_rot
