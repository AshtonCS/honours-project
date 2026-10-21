"""The high-precision check on synthesised Clifford+T words."""

import numpy as np
import pytest

from plsimcount import synthesis


def _double_error(word: str, theta: float) -> float:
    U = np.eye(2, dtype=complex)
    for ch in word:
        U = U @ synthesis._MATS[ch]
    return synthesis.distance(U, synthesis.rz_matrix(theta))


@pytest.mark.parametrize("theta", [0.7316, -1.234, 3e-6])
@pytest.mark.parametrize("eps", [1e-6, 1e-9])
def test_agrees_with_double_precision_where_that_is_valid(theta, eps):
    word = synthesis.synthesise(theta, eps)
    assert synthesis.word_error(word, theta, False, eps) == pytest.approx(_double_error(word, theta), abs=1e-13)


def test_hand_checkable_cases():
    # T = diag(1, e^{i pi/4}) = e^{i pi/8} RZ(pi/4): exact up to phase, off by a phase otherwise.
    # Up to phase, the only error left is the target itself: np.pi/4 is a double, which
    # differs from pi/4 by delta, and the check must see that delta (error |delta|/2).
    import mpmath
    with mpmath.workdps(40):
        half_delta = float(abs(mpmath.mpf(np.pi / 4) - mpmath.pi / 4) / 2)
    assert synthesis.word_error("T", np.pi / 4, True, 1e-20) == pytest.approx(half_delta, rel=1e-6)
    assert synthesis.word_error("T", np.pi / 4, False, 1e-20) == pytest.approx(2 * np.sin(np.pi / 16))
    # the identity is RZ(0) exactly
    assert synthesis.word_error("I", 0.0, False, 1e-20) == 0.0


@pytest.mark.parametrize("eps", [1e-15, 1e-17])
def test_reaches_bound_precisions(eps):
    """Below ~1e-14 the double-precision check could not certify any word."""
    word = synthesis.synthesise(0.7316, eps)
    assert synthesis.word_error(word, 0.7316, False, eps) <= eps


def test_a_coarse_word_is_rejected_at_fine_eps():
    coarse = synthesis.synthesise(0.7316, 1e-6)
    assert synthesis.word_error(coarse, 0.7316, False, 1e-12) > 1e-12
