"""Optimised counts (pfoptimised.py): marginal-per-step bookkeeping and soundness."""

import json
import warnings

import pytest

warnings.filterwarnings("ignore")

from plsimcount import pfoptimised as O  # noqa: E402
from plsimcount.pfcompare import HERE, fixture_hamiltonian, formula_registry  # noqa: E402

THESIS = HERE.parent / "results" / "pf-thesis" / "results.json"


def test_marginal_bookkeeping_suzuki2_n4():
    """marginal = opt(3 steps) - opt(2 steps), edge = opt(2) - 2 marginal,
    unoptimised = the pf-thesis merged step (S2, n = 4: 40 CNOT)."""
    c = O.optimise_case(fixture_hamiltonian(4, 1), formula_registry()["suzuki2"], 4, 699)
    assert c["unopt"]["CNOT"] == 40
    for m in O.METRICS:
        assert c["marginal"][m] == c["opt3"][m] - c["opt2"][m]
        assert c["edge"][m] == c["opt2"][m] - 2 * c["marginal"][m]
        assert c["marginal"][m] <= c["isolated"][m] <= c["unopt"][m]
    assert c["verify_unitary_err"] is not None and c["verify_unitary_err"] < 1e-9
    assert c["verify_state_err"] < 1e-9


def test_unoptimised_counts_match_pf_thesis_rows():
    """The step being optimised is exactly pf-thesis's merged cz step."""
    rows = json.loads(THESIS.read_text())["rows"]
    want = {(r["scheme"], r["n"]): r for r in rows if r["r_source"] == "measured" and r["gatebase"] == "cz"
            and r["merged"] and r["instance"] == "simcount"}
    reg = formula_registry()
    for scheme in ("suzuki2", "mo-q6"):
        w = want[(scheme, 4)]
        c = O.optimise_case(fixture_hamiltonian(4, 1), reg[scheme], 4, w["r"])
        assert (c["unopt"]["CNOT"], c["unopt"]["Rz"], c["unopt"]["total"]) == \
            (w["cnot_per_step"], w["rz_per_step"], w["gates_per_step"])


def test_measured_r_from_checks_seed():
    src = json.loads(THESIS.read_text())
    assert O.measured_r_from(src, 1)[("suzuki4", 4)] == 45
    with pytest.raises(ValueError, match="seed"):
        O.measured_r_from(src, 2)
