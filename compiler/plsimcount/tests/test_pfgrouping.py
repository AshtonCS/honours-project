"""Grouping comparison (pfgrouping.py): the r search for every grouping agrees
with the slow references, and the gate counts follow by hand."""

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from plsimcount import empirical, evenodd, fastempirical, formula, pfgrouping as G, twogroup  # noqa: E402
from plsimcount.pfcompare import fixture_hamiltonian  # noqa: E402


def test_group_evolution_step_equals_twogroup_expm():
    """GroupEvolution's eigendecomposition step = twogroup.step_unitary (dense expm)."""
    H = fixture_hamiltonian(4, 1)
    for outer in twogroup.OUTER:
        ge = G.GroupEvolution(H, outer)
        mats = twogroup._group_matrices(H, outer)
        for order in (2, 4):
            a = ge.step(twogroup.group_schedule(order, 0.37))
            b = twogroup.step_unitary(mats, order, 0.37)
            assert np.linalg.norm(a - b, 2) < 1e-12


@pytest.mark.parametrize("outer", twogroup.OUTER)
def test_twogroup_r_equals_reference_search(outer):
    """n = 4, S2, eps = 1e-3: same r as twogroup.steps_empirical (expm + bisection)."""
    H = fixture_hamiltonian(4, 1)
    r_ref = twogroup.steps_empirical(H, 2, 4.0, 1e-3, outer)
    assert G.measured_r(H, f"twogroup-{outer}-out", 2, 4.0, 1e-3).r == r_ref


def test_evenodd_r_equals_reference_search():
    """n = 4, S2: fastempirical on the even/odd ordering = empirical.steps on it."""
    H = fixture_hamiltonian(4, 1)
    r_ref = empirical.steps(2, evenodd.even_odd(H), 4.0, 1e-3)
    assert G.measured_r(H, "evenodd", 2, 4.0, 1e-3).r == r_ref


def test_xyz_r_is_pf_thesis_r():
    """The X-Y-Z variant is pf-thesis's Suzuki S2: r = 699 at n = 4 (cz)."""
    H = fixture_hamiltonian(4, 1)
    assert G.measured_r(H, "xyz", 2, 4.0, 1e-3).r == 699


def test_blocked_has_no_search():
    with pytest.raises(ValueError, match="no r search"):
        G.measured_r(fixture_hamiltonian(4, 1), "blocked", 2, 4.0, 1e-3)


def test_cnot_per_step_by_hand_n4_s2():
    """n = 4, S2 (4 bonds, 3 terms each, 4 fields):
    (a) xyz: simcount's split puts every bond term (XX, YY, ZZ) in the d/2 head, fields at d in the
        middle: 12 bond terms x 2 = 24 two-qubit exps x 2 CNOT = 48 unmerged (pf-thesis's 48).
        Merged, the two ZZ halves around the commuting fields combine: 20 x 2 = 40;
    (b) evenodd: all 12 bond terms twice at d/2 = 24 x 2 = 48 (no merge: odd bonds separate the halves
        of the even ones, and the fields sit between the odd halves but anticommute with XX/YY);
    (c) blocked: 4 bonds x 2 applications = 8 blocks x 3 = 24;
    (d) two groups, E' outer: 2 E' bonds x 2 + 2 O bonds x 1 = 6 blocks x 3 = 18 (same for O outer)."""
    H = fixture_hamiltonian(4, 1)
    want = {"xyz": 48, "evenodd": 48, "blocked": 24, "twogroup-even-out": 18, "twogroup-odd-out": 18}
    for v, c in want.items():
        tape = G.step_tape(H, v, 2, 0.1, merged=False)
        assert sum(op.name == "CNOT" for op in tape.operations) == c, v
    merged = G.step_tape(H, "xyz", 2, 0.1, merged=True)
    assert sum(op.name == "CNOT" for op in merged.operations) == 40


def test_merging_only_for_termwise_variants():
    with pytest.raises(ValueError, match="term-by-term"):
        G.step_tape(fixture_hamiltonian(4, 1), "blocked", 2, 0.1, merged=True)


def test_run_rows_small():
    rows = G.run({"seed": 1, "orders": [2], "sizes": [4]}, log=lambda *a: None)
    by = {(r["variant"], r["merged"]): r for r in rows}
    assert by[("blocked", False)]["r"] == by[("evenodd", False)]["r"]
    assert by[("xyz", False)]["cnot_total"] == 48 * 699
    assert by[("xyz", True)]["cnot_total"] == 40 * 699
    assert set(v for v, _ in by) == set(G.VARIANTS)
