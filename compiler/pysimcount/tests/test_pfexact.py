"""pfexact: exact references checked by hand, and a small end-to-end run."""

import warnings

import numpy as np
import pytest
from scipy.linalg import expm

warnings.filterwarnings("ignore")

from pysimcount import pfexact as X  # noqa: E402
from pysimcount.pfcompare import formula_registry, numpy_fields  # noqa: E402


def test_diagonal_reference_by_hand_n2():
    """n = 2 (open: one bond), h = (0.3, -0.5), Jz = 1, t = 1.
    E(00) = +1 + 0.3 - 0.5 =  0.8     E(01) = -1 + 0.3 + 0.5 = -0.2
    E(10) = -1 - 0.3 - 0.5 = -1.8     E(11) = +1 - 0.3 + 0.5 =  1.2"""
    got = X.diagonal_reference(2, [0.3, -0.5], 1.0)
    want = np.exp(-1j * np.array([0.8, -0.2, -1.8, 1.2]))
    assert np.allclose(got, want, atol=1e-15)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_diagonal_reference_matches_expm_of_the_hamiltonian(n):
    """The independent closed form agrees with expm of the Hamiltonian object
    (periodic for n >= 3), so Part A's reference and H describe one system."""
    f = numpy_fields(n, 1)
    H = X.zz_hamiltonian(n, f)
    assert np.allclose(np.diag(X.diagonal_reference(n, f, 0.7)), expm(-0.7j * H.matrix()), atol=1e-12)


@pytest.mark.parametrize("word", ["XZ", "YY", "ZI", "XYZ"])
def test_pauli_exponential_matches_expm(word):
    P = X.pauli_matrix(word)
    assert np.allclose(X.pauli_exponential(0.8, word, 0.37), expm(-1j * 0.8 * 0.37 * P), atol=1e-14)


def test_exact_unitary_matches_expm():
    from pysimcount.hamiltonian import heisenberg
    H = heisenberg(4, numpy_fields(4, 1))
    assert np.allclose(X.exact_unitary(H, 4.0), expm(-4j * H.matrix()), atol=1e-12)


def test_order_slope_recovers_synthetic_power_law():
    """error = 5 r^-4 plus one point above and one below the window, which
    must be excluded from the fit."""
    rs = [2, 10, 20, 40, 80, 160, 10000]
    errs = [5.0 * r**-4.0 for r in rs]  # 0.31, 5e-4 ... 7.6e-9, 5e-16
    slope, used = X.order_slope(rs, errs, (1e-10, 1e-2))
    assert used == [10, 20, 40, 80, 160]
    assert slope == pytest.approx(4.0, abs=1e-12)


def test_order_slope_needs_two_points():
    assert X.order_slope([1, 2], [1.0, 0.5], (1e-10, 1e-2)) == (None, [])


@pytest.mark.parametrize("key", ["lie-trotter", "suzuki4", "mo-q6"])
def test_commuting_case_is_exact(key):
    spec = formula_registry()[key]
    row = X.commuting_row(4, numpy_fields(4, 1), 4.0, 1, key, spec, 1e-15)
    assert row["pass"], row


def test_commuting_control_detects_wrong_weights():
    spec = formula_registry()["suzuki2"]
    row = X.commuting_row(4, numpy_fields(4, 1), 4.0, 1, "suzuki2", spec, 1e-15, time_scale=1.001)
    assert not row["pass"]
    assert row["max_weight_sum_dev"] > 1e-4


def test_order_check_passes_for_s2_and_fails_when_lie_trotter_claims_order_2():
    from pysimcount.hamiltonian import heisenberg
    reg = formula_registry()
    H = heisenberg(4, numpy_fields(4, 1))
    pts = X.noncommuting_points(H, 4.0, "suzuki2", reg["suzuki2"], [250, 500, 1000, 2000], 1e-15)
    good = X.order_fit(pts, 2, (1e-10, 1e-2), 0.3, 3)
    assert good["compile_pass"] and good["order_pass"], good
    pts = X.noncommuting_points(H, 4.0, "lie-trotter", reg["lie-trotter"], [2000, 4000, 8000, 16000], 1e-15)
    bad = X.order_fit(pts, 2, (1e-10, 1e-2), 0.3, 3)
    assert bad["compile_pass"] and not bad["order_pass"]


def test_config_rejects_sizes_too_large_for_dense_matrices():
    cfg = {"name": "x", "description": "x", "seed": 1, "schemes": ["suzuki2"],
           "commuting": {"sizes": [10], "r": [1]}, "noncommuting": {"sizes": [4], "r_grid": {"default": [1]}},
           "criteria": {}, "controls": {}}
    with pytest.raises(ValueError, match="2..8"):
        X.run(cfg, verbose=False)


# ------------------------------------------------------------------ Part C

def test_local_fit_uses_smallest_usable_d():
    """error = 3 d^5 with floor 1e-12 at margin 100: points below 1e-10 are
    dropped and the fit takes the 4 smallest of the rest."""
    ds = [2 ** (-k / 2) for k in range(21)]
    pts = [{"n": 4, "d": d, "scheme": "s", "name": "s", "order": 4, "exps_per_step": 1000,
            "error": 3 * d**5 + 1e-13, "floor": 1e-12} for d in ds]
    f = X.local_fit(pts, 4, 100, 4, 0.3)
    assert all(e >= 1e-10 for e in f["errors_used"])
    assert len(f["d_used"]) == 4 and f["d_used"] == sorted(f["d_used"])
    assert f["pass"] and f["slope"] == pytest.approx(5.0, abs=0.05)


@pytest.mark.parametrize("key,p", [("suzuki2", 2), ("suzuki4", 4), ("mo-q6", 4)])
def test_local_order(key, p):
    from pysimcount.hamiltonian import heisenberg
    H = heisenberg(4, numpy_fields(4, 1))
    pts = X.local_points(H, key, formula_registry()[key], [2 ** (-k / 2) for k in range(21)], 1e-15)
    f = X.local_fit(pts, p, 100, 4, 0.3)
    assert f["pass"], f


def test_local_controls_fail():
    from pysimcount.hamiltonian import heisenberg
    reg = formula_registry()
    H = heisenberg(4, numpy_fields(4, 1))
    ds = [2 ** (-k / 2) for k in range(21)]
    c1 = X.local_fit(X.local_points(H, "lie-trotter", reg["lie-trotter"], ds, 1e-15), 2, 100, 4, 0.3)
    c2 = X.local_fit(X.local_points(H, "suzuki4", reg["suzuki4"], ds, 1e-15, time_scale=1 + 1e-6), 4, 100, 4, 0.3)
    assert not c1["pass"] and not c2["pass"]
    assert c2["slope"] == pytest.approx(1.0, abs=0.3)
