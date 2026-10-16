"""Independent BCH computation vs upstream's recursion, the .tex, and matrices."""

from fractions import Fraction

import numpy as np
import pytest
from scipy.linalg import expm, logm

from effts import bch
from effts import schemes as S
from effts.basis import BASIS


# --------------------------------------------------------- hand-checkable

def test_word_encoding():
    """AB - BA: "AB" = 0b01 = 1, "BA" = 0b10 = 2 (first letter = top bit)."""
    v = bch.right_nested("AB")
    assert v.tolist() == [0.0, 1.0, -1.0, 0.0]


def test_log_of_single_exponential():
    """log(exp(0.7 A)) = 0.7 A, nothing else."""
    lg = bch.log(bch.exp_letter("A", 0.7))
    assert lg[1].tolist() == pytest.approx([0.7, 0.0])
    for d in range(2, bch.MAX_DEGREE + 1):
        assert np.allclose(lg[d], 0.0, atol=1e-15)


def test_bch_degree_two():
    """log(e^A e^B) = A + B + [A,B]/2 + ...  (textbook BCH)."""
    lg = bch.log(bch.mul(bch.exp_letter("A", 1.0), bch.exp_letter("B", 1.0)))
    assert lg[2].tolist() == pytest.approx([0.0, 0.5, -0.5, 0.0])


def test_leapfrog_err2_by_hand():
    """e^{A/2} e^B e^{A/2}: O_3 = -1/24 [A,[A,B]] + 1/12 [B,[B,A]], so
    Err_2 = sqrt(1/24^2 + 1/12^2) = sqrt(5/576), Eff_2 = 1/Err_2 (q = 1)."""
    s = S.leapfrog()
    assert bch.err(s, 2) == pytest.approx(np.sqrt(5 / 576), rel=1e-14)
    assert bch.efficiency(s) == pytest.approx(np.sqrt(576 / 5), rel=1e-14)


# ------------------------------------------------------------ vs the .tex

@pytest.mark.parametrize("degree", [3, 5, 7, 9])
def test_basis_spans(degree):
    """Free Lie algebra on 2 letters has dims 2, 6, 18, 56 at degrees 3,5,7,9."""
    M = bch.basis_matrix(degree)
    assert np.linalg.matrix_rank(M) == len(BASIS[degree]) == {3: 2, 5: 6, 7: 18, 9: 56}[degree]


@pytest.mark.parametrize("degree", [3, 5, 7, 9])
def test_symmetric_bch_coefficients_match_tex(degree):
    """log(e^{A/2} e^B e^{A/2}) in the upstream basis = the tilde coefficients
    printed in Recursive_formulae.tex (alpha~ = -1/24, ..., epsilon~_56).

    This pins down the reading of the bracket notation [X1 X2 .. Xk] and the
    sign convention of every basis element, independently of upstream code.
    """
    coef = bch.error_coefficients(S.leapfrog())[degree]
    tex = np.array([float(f) for _, f in BASIS[degree]])
    np.testing.assert_allclose(coef, tex, atol=1e-15, rtol=0)


def test_tex_tilde_spot_values():
    """Two values read off the .tex by eye, guarding the extraction itself."""
    assert BASIS[3][0] == ("AAB", Fraction(-1, 24))
    assert BASIS[7][0] == ("AAAAAAB", Fraction(-31, 967680))


# --------------------------------------------- vs upstream's recursion output

@pytest.mark.parametrize("path_scheme", S.all_recommended(), ids=lambda ps: f"{ps[0].parent.name}-{ps[1].q}")
def test_published_err_n(path_scheme):
    """Every Err_n^2 printed in the upstream .out files, for n >= the scheme's
    order, reproduced by the independent BCH computation.

    Tolerance 1e-6 relative: the worst case observed is 7e-8 (order 8, q = 30,
    Err_8^2 ~ 1e-17), limited by double-precision cancellation in our log;
    upstream works in long double/quad.
    """
    _, s = path_scheme
    co = bch.error_coefficients(s)
    for n, e2 in s.reference:
        if n >= s.order:
            assert float(np.sum(co[n + 1] ** 2)) == pytest.approx(e2, rel=1e-6)


@pytest.mark.parametrize("path_scheme", S.all_recommended(), ids=lambda ps: f"{ps[0].parent.name}-{ps[1].q}")
def test_order_conditions(path_scheme):
    """Every published scheme satisfies its order conditions: nu = sigma = 1
    and Err_k = 0 for k < order, to double precision."""
    _, s = path_scheme
    co = bch.error_coefficients(s)
    assert np.max(np.abs(co[1])) < 1e-13
    for k in range(2, s.order, 2):
        assert np.linalg.norm(co[k + 1]) < 1e-13


@pytest.mark.parametrize("table", [2, 3])
def test_paper_tables_are_valid_schemes(table):
    """The paper's printed Tables 2-3 are themselves valid order-4 / order-6
    schemes, with the same Eff as the repo's versions to 4 digits."""
    s = S.table_scheme(table)
    repo = S.recommended_q6() if table == 2 else S.recommended_q14()
    co = bch.error_coefficients(s, s.order + 1)
    for k in range(2, s.order, 2):
        assert np.linalg.norm(co[k + 1]) < 1e-13
    assert bch.efficiency(s) == pytest.approx(bch.efficiency(repo), rel=1e-4)


# ----------------------------------------------------- vs actual matrices

def _random_antihermitian(rng, n):
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    X = (X - X.conj().T) / 2
    return X / np.linalg.norm(X, 2)


def _commutator_matrix(word, A, B):
    mats = {"A": A, "B": B}
    M = mats[word[-1]]
    for ch in reversed(word[:-1]):
        X = mats[ch]
        M = X @ M - M @ X
    return M


def _scheme_matrix(s, h, A, B):
    P = np.eye(A.shape[0], dtype=complex)
    for op, w in s.stages():
        P = P @ expm(w * h * (A if op == "A" else B))
    return P


@pytest.mark.parametrize("make", [S.leapfrog, S.forest_ruth, S.recommended_q6,
                                  S.recommended_q14, lambda: S.recommended(8, 15)[0]])
def test_leading_error_operator(make):
    """log S(h) = h (A + B) + h^{n+1} O_{n+1} + O(h^{n+3}) on random matrices,
    with O_{n+1} built from the computed coefficients and the basis.

    Ties the abstract BCH coefficients to the actual product of matrix
    exponentials: the remainder shrinks like h^2 relative to the leading term.
    """
    s = make()
    n = s.order
    rng = np.random.default_rng(7)
    A, B = _random_antihermitian(rng, 4), _random_antihermitian(rng, 4)
    coef = bch.error_coefficients(s, n + 1)[n + 1]
    O = sum(c * _commutator_matrix(w, A, B) for c, (w, _) in zip(coef, BASIS[n + 1]))
    # Step sizes: the leading term h^{n+1} O must stay well above logm's
    # ~1e-15 rounding.  At h = 0.1 an order-6/8 term is ~1e-14 (noise), so
    # higher orders are probed at larger h; the h^2 convergence is unchanged.
    rel = []
    for h in ((0.2, 0.1) if n <= 4 else (0.8, 0.4)):
        L = logm(_scheme_matrix(s, h, A, B)) - h * (A + B)
        rel.append(np.linalg.norm(L - h ** (n + 1) * O) / np.linalg.norm(h ** (n + 1) * O))
    assert rel[1] < 0.05
    assert rel[1] < rel[0] / 3  # ~h^2 convergence (factor 4 ideally)


@pytest.mark.parametrize("make", [S.leapfrog, S.forest_ruth, S.recommended_q6, S.yoshida6,
                                  lambda: S.blanes_moan(10), S.recommended_q14])
def test_order_by_step_size_scaling(make):
    """|| S(h) - exp(h(A+B)) || ~ h^{n+1}: fitted log-log slope within 0.25 of n + 1.

    Uses nothing from bch.py -- an independent check of each scheme's order.
    """
    s = make()
    rng = np.random.default_rng(11)
    A, B = _random_antihermitian(rng, 6), _random_antihermitian(rng, 6)
    hs = np.array([0.4, 0.3, 0.2]) * (1.0 if s.order <= 4 else 2.0)
    errs = [np.linalg.norm(_scheme_matrix(s, h, A, B) - expm(h * (A + B)), 2) for h in hs]
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert abs(slope - (s.order + 1)) < 0.25, slope
