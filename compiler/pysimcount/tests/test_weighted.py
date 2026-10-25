"""General-weights schedules and the merge rule (weighted.py)."""

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from effts import evolution as EE  # noqa: E402
from effts import heisenberg as EH  # noqa: E402
from effts import schemes as S  # noqa: E402
from pysimcount import circuit, formula, resources, weighted as W  # noqa: E402
from pysimcount.empirical import Evolution  # noqa: E402
from pysimcount.hamiltonian import Term, heisenberg  # noqa: E402

FIELDS4 = [0.3, -0.7, 0.5, 0.1]  # arbitrary, hand-picked
H4 = heisenberg(4, FIELDS4)


def dense_step(H, schedule):
    """Reference unitary: first schedule entry acts first (empirical.Evolution convention)."""
    from scipy.linalg import expm
    ev = Evolution(H)
    U = np.eye(2 ** H.n_qubits, dtype=complex)
    for j, time in schedule:
        U = expm(-1j * time * H.terms[j].coeff * ev.mats[j]) @ U
    return U


def test_commute_by_hand():
    assert W.commute(Term(1, "XXI"), Term(1, "YYI"))      # two clashes
    assert not W.commute(Term(1, "XXI"), Term(1, "ZII"))  # one clash
    assert W.commute(Term(1, "XXI"), Term(1, "IIZ"))      # disjoint


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
def test_heisenberg_has_three_stages(n):
    """XX bonds | YY bonds | ZZ bonds + fields: the paper's global grouping."""
    H = heisenberg(n, [0.1] * n)
    groups = W.stage_groups(H)
    assert len(groups) == 3
    assert {H.terms[j].word.replace("I", "") for j in groups[0]} == {"XX"}
    assert {H.terms[j].word.replace("I", "") for j in groups[1]} == {"YY"}
    assert {H.terms[j].word.replace("I", "") for j in groups[2]} == {"ZZ", "Z"}


@pytest.mark.parametrize("order", [2, 4, 6])
def test_suzuki_as_weights_matches_formula(order):
    """Suzuki's own weights through ramp_schedule give simcount's Suzuki unitary
    (the field terms sit differently, but they commute with ZZ)."""
    s = W.suzuki_scheme(order)
    a = dense_step(H4, W.ramp_schedule(H4, s.ramps(), 0.37))
    b = dense_step(H4, formula.schedule(order, H4, 0.37))
    assert np.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("make", [S.recommended_q6, S.recommended_q14, lambda: S.blanes_moan(10)])
def test_matches_effts_step_operator(make):
    """Same unitary as the effts port of the paper's own code (global grouping, J = 1)."""
    s = make()
    ours = dense_step(H4, W.ramp_schedule(H4, s.ramps(), 0.37))
    ref = EE.step_operator(EH.chain(4, "XXZ", FIELDS4), s, 0.37, "global")
    assert np.allclose(ours, ref, atol=1e-12)


@pytest.mark.parametrize("key", ["suzuki2", "suzuki4", "mo-q6", "mo-q14", "bm-q10"])
def test_every_term_gets_total_time_d(key):
    """Weights sum to 1 per term per step, for every formula."""
    from pysimcount.pfcompare import formula_registry
    spec = formula_registry()[key]
    tot = np.zeros(H4.n_terms)
    for j, time in spec.schedule(H4, 0.3):
        tot[j] += time
    assert np.allclose(tot, 0.3, atol=1e-13)


@pytest.mark.parametrize("key", ["suzuki2", "suzuki4", "suzuki6", "mo-q6", "mo-q14", "bm-q10"])
def test_merge_is_exact_and_idempotent(key):
    from pysimcount.pfcompare import formula_registry
    spec = formula_registry()[key]
    sched = spec.schedule(H4, 0.37)
    m = W.merge(H4, sched)
    assert len(m) < len(sched)
    assert W.merge(H4, m) == m
    assert np.allclose(dense_step(H4, m), dense_step(H4, sched), atol=1e-12)


def test_merge_count_by_hand_S2():
    """S2 on n = 4: XX, YY, ZZ at d/2, fields at d, then YY, XX at d/2 (ZZ twice).
    Unmerged 4+4+4+4+4+4+4 = 28 exponentials; merging joins the two ZZ halves: 24."""
    sched = formula.schedule(2, H4, 0.2)
    assert len(sched) == 28
    assert len(W.merge(H4, sched)) == 24


def test_build_step_from_schedule_matches_circuit():
    for order in (1, 2, 4):
        a = resources.count(circuit.build_step(H4, order, 0.2))
        b = resources.count(W.build_step_from_schedule(H4, formula.schedule(order, H4, 0.2)))
        assert a == b


def _morales_direct(H, weights, d):
    """Independent construction of Morales et al.'s Eq. 15 on H's three stages:
    each S2(w d) = e^{-i w d G1/2} e^{-i w d G2/2} e^{-i w d G3} e^{-i w d G2/2} e^{-i w d G1/2},
    chained as S2(w_m) ... S2(w_1) S2(w_0) S2(w_1) ... S2(w_m) with w_0 = 1 - 2 sum w_j.
    Uses only dense expm of the group Hamiltonians, none of weighted.py's ramp code."""
    from scipy.linalg import expm
    mats = Evolution(H).mats
    G = [sum(H.terms[j].coeff * mats[j] for j in g) for g in W.stage_groups(H)]
    assert len(G) == 3
    w = [float(x) for x in weights]
    seq = list(reversed(w)) + [1 - 2 * sum(w)] + w

    def S2(x):
        half = [expm(-1j * x * d / 2 * Gk) for Gk in G[:2]]
        return half[0] @ half[1] @ expm(-1j * x * d * G[2]) @ half[1] @ half[0]

    U = np.eye(2 ** H.n_qubits, dtype=complex)
    for x in seq:
        U = U @ S2(x)
    return U


@pytest.mark.parametrize("key,weights", [("morales-y8m10", "Y8M10"), ("morales-m8", "M8")])
def test_morales_schedule_matches_direct_construction(key, weights):
    from effts import morales as M
    from pysimcount.pfcompare import formula_registry
    spec = formula_registry()[key]
    ours = dense_step(H4, spec.schedule(H4, 0.37))
    assert np.allclose(ours, _morales_direct(H4, getattr(M, weights), 0.37), atol=1e-12)
