"""The comparison runner's helpers (pfcompare.py)."""

import math
import warnings

import pytest

warnings.filterwarnings("ignore")

from plsimcount import formula, pfcompare as P, resources, simcount, synthesis  # noqa: E402
from plsimcount.weighted import build_step_from_schedule  # noqa: E402


def test_fixture_hamiltonian_matches_simcount_run():
    a = P.fixture_hamiltonian(4, 1)
    b = simcount.run(4, "pf2emp", "cz", 1).hamiltonian
    assert [(t.coeff, t.word) for t in a.terms] == [(t.coeff, t.word) for t in b.terms]
    assert a.split == b.split


def test_missing_fixture_raises():
    with pytest.raises(FileNotFoundError):
        P.fixture_hamiltonian(11, 1)


def test_synthesised_counts_equal_full_synthesis():
    """Counting per distinct angle = synthesising the whole tape and counting."""
    H = P.fixture_hamiltonian(4, 1)
    tape = build_step_from_schedule(H, formula.schedule(2, H, 0.05))
    eps_rot = 1e-6
    full = resources.count(synthesis.synthesise_tape(tape, eps_rot))
    assert P.synthesised_counts(resources.count(tape), eps_rot, "pygridsynth") == full


def test_fit_power_law_recovers_parameters():
    a, b = P.fit_power_law([4, 6, 8, 10], [3.0 * n ** 1.7 for n in (4, 6, 8, 10)])
    assert a == pytest.approx(3.0, rel=1e-12)
    assert b == pytest.approx(1.7, rel=1e-12)


def test_registry_orders_and_cycles():
    reg = P.formula_registry()
    assert [(reg[k].order, reg[k].q) for k in ("suzuki2", "suzuki4", "suzuki6", "suzuki8")] == \
        [(2, 1), (4, 5), (6, 25), (8, 125)]
    assert (reg["mo-q6"].order, reg["mo-q6"].q) == (4, 6)
    assert (reg["mo-q14"].order, reg["mo-q14"].q) == (6, 14)
    assert (reg["bm-q10"].order, reg["bm-q10"].q) == (6, 10)
    assert (reg["mo-o8-q21"].order, reg["mo-o8-q21"].q) == (8, 21)
    assert (reg["morales-y8m10"].order, reg["morales-y8m10"].q) == (8, 21)
    assert (reg["morales-m8"].order, reg["morales-m8"].q) == (8, 17)
