"""The paper's published numbers: Table 4 and Figs. 5-6 (digitised).

Figure data come from the arXiv v3 HTML's inline SVG (``digitise.py``),
precise to ~3e-4 decades.  The tolerance used, 1e-3 decades (0.23%), is
about 3x that, and well below anything visible on the plot.
"""

import json
import math

import numpy as np
import pytest

from effts import digitise
from effts import evolution as E
from effts import experiments as X

DECADES = 1e-3


@pytest.mark.parametrize("row", X.table4(), ids=lambda r: r[0])
def test_table4(row):
    """Every Eff_n and x-bar of Table 4 to the printed 3 decimals."""
    label, eff, eff_paper, x, x_paper = row
    assert round(eff, 3) == pytest.approx(eff_paper, abs=1.0001e-3)
    assert round(x, 3) == pytest.approx(x_paper, abs=1.0001e-3)
    assert abs(eff - eff_paper) <= 5.0001e-4 or abs(eff - eff_paper) / eff_paper < 5e-4
    assert abs(x - x_paper) <= 5.0001e-4


@pytest.mark.parametrize("panel", ["global", "local"])
def test_fig5(panel):
    """Every digitised point of the 6 reproducible Fig. 5 curves (~300 per panel)."""
    curves = X.fig5_comparison(panel)
    assert [c["q"] for c in curves] == [1, 3, 6, 7, 10, 14]
    for c in curves:
        d = np.abs(np.log10(c["ours"] / c["paper"]))
        assert d.max() < DECADES, (c["label"], d.max())


def test_fig6_small_L():
    """Fig. 6 markers for L = 2..8 (L = 9..12 via `python -m effts fig6`)."""
    for r in X.fig6_comparison(list(range(2, 9))):
        assert abs(math.log10(r["ours"] / r["paper"])) < DECADES, r


def test_fig6_rounding_rule():
    """The paper does not say how N_t is chosen when q does not divide 500.

    round() matches every marker; floor() and ceil() each fail for some q by
    far more than the digitisation error.  This is what fixes the rule.
    """
    Ls = [4, 6]
    for rule in ("floor", "ceil"):
        worst = max(abs(math.log10(r["ours"] / r["paper"])) for r in X.fig6_comparison(Ls, rule))
        assert worst > 10 * DECADES, rule


def test_digitised_fixture_is_from_v3():
    meta = json.loads(digitise.FIG5_FIXTURE.read_text())["_meta"]
    assert meta["source"].endswith("2601.18756v3")


def test_ranking_claim_holds_above_cost_140():
    """Sec. 3.1: "our recommended q=14 scheme performs better than the
    historical schemes for orders n <= 6 at each cost".  True in both
    groupings for every cost >= 140 checked (where Delta <= ~0.05)."""
    for g in ("global", "local"):
        for row in X.ranking_at_costs(g, [140, 280, 560, 1000, 2000]):
            errs = {q: v[1] for q, v in row.items() if q != "cost"}
            assert min(errs, key=errs.get) == 14, (g, row["cost"], errs)


def test_ranking_claim_fails_at_low_cost():
    """...but not "at each cost": at cost 84 (Delta ~ 0.3-0.5, off the sqrt(2)
    plateau but before the scaling region) Leapfrog and the paper's own q = 6
    scheme both beat q = 14, in both groupings.  The crossing is in the
    paper's own Fig. 5 data (digitised), not only in this port, and it
    contradicts "even at lower costs where the Leapfrog scheme was thought to
    excel, we find an improvement".  Crossovers (paper data): Leapfrog vs q=14
    at cost ~90 (global) / ~105 (local); q=6 vs q=14 at ~120 / ~133."""
    for g in ("global", "local"):
        row = X.ranking_at_costs(g, [84])[0]
        assert row[1][0] == row[14][0] == row[6][0] == 84  # identical cost
        assert row[1][1] < row[14][1] and row[6][1] < row[14][1]
    # Same comparison on the paper's own digitised curves (log-log interpolation
    # at cost 84; Leapfrog is sampled every 5 in cost, q=6 every 6, q=14 every 14).
    fig5 = json.loads(digitise.FIG5_FIXTURE.read_text())["curves"]
    for g in ("global", "local"):
        curve = {c["q"]: np.array(c["points"]) for c in fig5 if c["panel"] == g}

        def at84(q):
            c, e = curve[q][:, 0], curve[q][:, 1]
            return np.exp(np.interp(np.log(84.0), np.log(c), np.log(e)))
        assert at84(1) < at84(14) and at84(6) < at84(14), (g, at84(1), at84(6), at84(14))
