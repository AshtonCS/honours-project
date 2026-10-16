"""Morales et al.'s random-matrix error constants, reproduced (experiments.py).

Their setting is only partly specified (matrix distribution and t are
assumptions, see experiments.random_pair / random_matrix_constant), so the
tolerances are loose on chi itself and tight on Y8m10's cost score, which
the full 2,000-sample run reproduces exactly (2.56 vs 2.56).
"""

import pytest

from effts import experiments as X
from effts import morales as Mo


def test_y8m10_constant_and_cost_score():
    chi = X.random_matrix_constant(Mo.y8m10(), n_samples=400, seed=2026)
    paper_chi, paper_score = X.MORALES_TABLE8["Y8m10"]
    assert chi / paper_chi == pytest.approx(1.0, abs=0.25)
    assert 21 * chi ** (1 / 8) == pytest.approx(paper_score, abs=0.08)


def test_ranking_y8m10_beats_y8m10b_on_random_matrices():
    """Their Table 8 ranks Y8m10 ahead of Y8m10b on spectral error (by ~11x in chi)."""
    a = X.random_matrix_constant(Mo.y8m10(), n_samples=200, seed=7)
    b = X.random_matrix_constant(Mo.y8m10b(), n_samples=200, seed=7)
    assert 5 < b / a < 20
