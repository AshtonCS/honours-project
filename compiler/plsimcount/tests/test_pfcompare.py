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


def test_lie_trotter_is_simcounts_pf1():
    reg = P.formula_registry()
    spec = reg["lie-trotter"]
    assert (spec.order, spec.q, spec.family) == (1, 1, "suzuki")
    H = P.fixture_hamiltonian(4, 1)
    assert spec.schedule(H, 0.3) == formula.schedule(1, H, 0.3)


def test_bound_orders_match_bounds_module():
    """BOUND_ORDERS lists exactly the orders each bound accepts."""
    from plsimcount import bounds
    H = P.fixture_hamiltonian(4, 1)
    for bound, orders in P.BOUND_ORDERS.items():
        for order in (1, 2, 4, 6, 8):
            if order in orders:
                assert bounds.steps(f"pf{order}{bound}", H, 4.0, 1e-3) >= 1
            else:
                with pytest.raises(ValueError):
                    bounds.steps(f"pf{order}{bound}", H, 4.0, 1e-3)


def test_bound_row_reproduces_simcount_pf2com_n4(tmp_path):
    """A 'com' bound row at n = 4 must be simcount's own pf2com cz run:
    r = 5503 and 204 gates per unmerged step, 1,122,612 in total (verified
    against the simcount binary, 2026-09-22)."""
    import json
    cfg = {"name": "t", "schemes": ["suzuki2"], "sizes": [4], "seed": 1, "gatebases": ["cz"],
           "bound_rows": {"schemes": ["lie-trotter", "suzuki2"], "bounds": ["com", "tte"], "sizes": [4]}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    P.main([str(tmp_path / "config.json")])
    rows = json.loads((tmp_path / "results.json").read_text())["rows"]
    com = [r for r in rows if r["r_source"] == "com" and r["scheme"] == "suzuki2" and not r["merged"]]
    assert len(com) == 1
    assert com[0]["r"] == 5503
    assert com[0]["gates_total"] == 1_122_612
    assert {r["r_source"] for r in rows} == {"measured", "com", "tte"}
