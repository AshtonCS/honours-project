"""Paulis, the Heisenberg chain, the Trotter step, and parity with upstream C++."""

import json
from functools import reduce

import numpy as np
import pytest
from scipy.linalg import expm

from effts import evolution as E
from effts import heisenberg as Hm
from effts import reference as R
from effts import schemes as S
from effts.paulis import PauliString

I2 = np.eye(2)
PAULI = {"X": np.array([[0, 1], [1, 0]], dtype=complex),
         "Y": np.array([[0, -1j], [1j, 0]]),
         "Z": np.diag([1.0, -1.0]).astype(complex)}


def kron_string(L, ops):
    """Reference: site 0 is the leftmost kron factor."""
    d = dict(ops)
    return reduce(np.kron, [PAULI[d[s]] if s in d else I2 for s in range(L)])


@pytest.mark.parametrize("ops", [((0, "X"),), ((2, "Y"),), ((1, "Z"),), ((0, "X"), (3, "X")),
                                 ((1, "Y"), (2, "Y")), ((3, "Z"), (0, "Z")), ((0, "Y"), (1, "Z"), (3, "X"))])
def test_pauli_string_matches_kron(ops):
    assert np.array_equal(PauliString(4, ops).dense(), kron_string(4, ops))


def test_exp_apply_matches_expm():
    p = PauliString(3, ((0, "Y"), (2, "X")))
    M = np.random.default_rng(0).normal(size=(8, 8)) + 0j
    assert np.allclose(p.exp_apply(0.37, M), expm(-0.37j * p.dense()) @ M, atol=1e-14)


def test_hamiltonian_by_hand_L3():
    """L = 3 periodic XXZ, J = 1, fields (0.1, -0.2, 0.05): written out term by term."""
    h = [0.1, -0.2, 0.05]
    ch = Hm.chain(3, "XXZ", h)
    H = np.zeros((8, 8), dtype=complex)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        for p in "XYZ":
            H += kron_string(3, ((i, p), (j, p)))
    for i in range(3):
        H += h[i] * kron_string(3, ((i, "Z"),))
    assert np.allclose(ch.hamiltonian(), H, atol=1e-15)


def test_open_boundary_drops_bond_keeps_field():
    """Upstream bound = true: bond (L-1, 0) gets J = 0 but h_{L-1} stays."""
    ch = Hm.chain(4, "XZ", [0.0, 0.0, 0.0, 0.3], periodic=False)
    H = ch.hamiltonian()
    assert np.allclose(H, H.conj().T)
    wrap = kron_string(4, ((3, "X"), (0, "X")))
    assert abs(np.trace(H @ wrap)) < 1e-12  # no X_3 X_0 component
    assert np.trace(H @ kron_string(4, ((3, "Z"),))).real == pytest.approx(0.3 * 16)


def test_trotter_exact_when_terms_commute():
    """J = 0: H = sum h_i Z_i is a sum of commuting terms; every scheme is exact."""
    ch = Hm.chain(4, "Heisenberg", [0.3, -0.1, 0.2, 0.05], J=(0.0, 0.0, 0.0))
    for s in (S.leapfrog(), S.recommended_q14()):
        for g in E.GROUPINGS:
            assert E.error_at(ch, s, 2.0, 3, g) < 1e-13


@pytest.mark.parametrize("grouping", E.GROUPINGS)
@pytest.mark.parametrize("make", [S.leapfrog, S.forest_ruth, S.recommended_q14])
def test_step_operator_matches_dense_expm(grouping, make):
    """Fast Pauli-action product vs a product of dense expm's in the same order."""
    ch = Hm.chain(4, "XXZ", [0.05, -0.08, 0.02, 0.09])
    s = make()
    M = np.eye(16, dtype=complex)
    for w, stage in E.step_factors(ch, s, 0.3, grouping):
        for p, coef in stage.terms:
            M = M @ expm(-1j * w * coef * p.dense())
    assert np.allclose(E.step_operator(ch, s, 0.3, grouping), M, atol=1e-13)


def test_ramp_orders():
    """Global: X..., Y..., Z...; local: bond by bond; down = reverse of up."""
    ch = Hm.chain(3, "XXZ", [0.0, 0.0, 0.0])
    g = [(st.direction, st.bond) for st in E.ramp(ch, "global", "up")]
    assert g == [("X", 0), ("X", 1), ("X", 2), ("Y", 0), ("Y", 1), ("Y", 2), ("Z", 0), ("Z", 1), ("Z", 2)]
    loc = [(st.direction, st.bond) for st in E.ramp(ch, "local", "up")]
    assert loc[:4] == [("X", 0), ("Y", 0), ("Z", 0), ("X", 1)]
    assert E.ramp(ch, "local", "down") == E.ramp(ch, "local", "up")[::-1]


def test_xz_model_has_no_y_stages():
    ch = Hm.chain(4, "XZ", [0.0] * 4)
    assert set(ch.stages) == {"X", "Z"}


# ------------------------------------------------- parity with upstream C++

_FIXTURE = json.loads(R.OPERATOR_FIXTURE.read_text())


@pytest.mark.parametrize("rec", _FIXTURE["records"],
                         ids=lambda r: f"{r['case']['model']}-L{r['case']['L']}-{r['case']['grouping']}-{r['case']['scheme'][:12]}")
def test_upstream_parity(rec):
    """Delta after every step vs upstream Time_Evolution (operator method),
    same model, fields, scheme and step size.  20 cases, 635 values.

    Tolerance: 1e-12 absolute.  Observed worst is 5e-14 (double rounding);
    both codes print/keep ~16 significant digits.
    """
    c = rec["case"]
    J = tuple(c["J"]) if isinstance(c["J"], list) else c["J"]
    ch = Hm.chain(c["L"], c["model"], R._fields(c["L"]), J, c["periodic"])
    ours = E.error_per_step(ch, R.scheme_by_name(c["scheme"]), c["t"], c["n_steps"], c["grouping"])
    np.testing.assert_allclose(ours, rec["errors"], rtol=0, atol=1e-12)


def test_fixture_inputs_are_current():
    """The recorded upstream inputs are exactly what reference.py would write
    today, so a change to scheme data or fields cannot pass silently."""
    for rec in _FIXTURE["records"]:
        assert R.input_file(rec["case"]) == rec["input"]


def test_field_rng_libraries_agree():
    """Upstream's seed-42 fields: libstdc++ and libc++ agree to 1 ulp."""
    data = json.loads(R.FIELDS_FIXTURE.read_text())
    for L in range(2, 13):
        a, b = np.array(data["libstdc++"][str(L)]), np.array(data["libc++"][str(L)])
        assert np.max(np.abs(a - b)) < 1e-16
        assert np.all(np.abs(a) <= 0.1)
