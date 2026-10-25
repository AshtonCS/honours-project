"""Grouping comparison (pfgrouping.py): the r search for every grouping agrees
with the slow references, and the gate counts follow by hand."""

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from pysimcount import empirical, evenodd, fastempirical, formula, pfgrouping as G, twogroup  # noqa: E402
from pysimcount.pfcompare import fixture_hamiltonian  # noqa: E402


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


# ------------------------------------------------------------------ any formula ("schemes" mode)

def test_schemes_mode_suzuki_matches_orders_mode():
    """For Suzuki S2/S4 the any-formula runner gives the same r and unmerged
    CNOT counts as the original Suzuki-only runner."""
    cfg_old = {"orders": [2, 4], "sizes": [4], "seed": 1, "gatebase": "cz"}
    old = {(r["order"], r["variant"]): r for r in G.run(cfg_old, log=lambda *_: None) if not r["merged"]}
    meta = {"fields": {}}
    cfg_new = {"schemes": ["suzuki2", "suzuki4"], "variants": list(G.SCHEME_VARIANTS), "sizes": [4], "seed": 1}
    new = G.run_schemes(cfg_new, meta, log=lambda *_: None)
    for r in new:
        if r["merged"]:
            continue
        o = old[(r["order"], r["variant"])]
        assert (r["r"], r["cnot_per_step"], r["rz_per_step"]) == (o["r"], o["cnot_per_step"], o["rz_per_step"])
    assert meta["fields"]["4"] == pytest.approx([-0.8444192498333132, 0.7998200058907112, 0.2371705373647761,
                                                 -0.5823152775681468])


def test_schemes_mode_merging_two_groups_by_hand():
    """S4 two-group: 5 S2 cycles, A/2 B A/2 each, so 10 outer applications
    of which the 4 at the joins pair up: 6 outer + 5 inner = 11 merged group
    applications against 15 unmerged. n = 4: 2 bonds per group, 3 CNOT each."""
    meta = {"fields": {}}
    rows = G.run_schemes({"schemes": ["suzuki4"], "variants": ["twogroup-odd-out"], "sizes": [4], "seed": 1},
                         meta, log=lambda *_: None)
    by = {r["merged"]: r for r in rows}
    assert by[False]["cnot_per_step"] == 15 * 2 * 3
    assert by[True]["cnot_per_step"] == 11 * 2 * 3
    assert by[False]["r"] == by[True]["r"]


def test_schemes_mode_numpy_instances_record_fields():
    from pysimcount.pfcompare import numpy_fields
    meta = {"fields": {}}
    rows = G.run_schemes({"schemes": ["mo-q6"], "variants": ["xyz", "twogroup-even-out"], "sizes": [4],
                          "seed": 3, "instances": "numpy"}, meta, log=lambda *_: None)
    assert {r["instance"] for r in rows} == {"numpy"}
    assert meta["fields"]["4"] == numpy_fields(4, 3)
    assert len(rows) == 4


def test_schemes_mode_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unknown variants"):
        G.run_schemes({"schemes": ["suzuki2"], "variants": ["blocked"], "sizes": [4], "seed": 1}, {"fields": {}})
