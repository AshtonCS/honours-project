"""Scheme notations (file / stage / ramp), parsing, and the paper's tables."""

import numpy as np
import pytest

from effts import schemes as S


def test_forest_ruth_by_hand():
    """Docstring worked example: file a = [-0.3512, 1.3512], b = [-1.7024, 2.7024]
    -> A stages theta/2, (1-theta)/2, (1-theta)/2, theta/2 and B stages
    theta, 1 - 2 theta, theta, with theta = 1.3512 (Forest & Ruth)."""
    s = S.forest_ruth()
    theta = 1.351207191959657634
    assert s.q == 3 and s.order == 4
    np.testing.assert_allclose(s.a, [theta / 2, (1 - theta) / 2, (1 - theta) / 2, theta / 2], atol=1e-15)
    np.testing.assert_allclose(s.b, [theta, 1 - 2 * theta, theta], atol=1e-15)


def test_leapfrog_ramps_and_origin():
    """q = 1: c_1 = a_1 = 1/2, d_1 = b_1 - c_1 = 1/2, which is the origin 1/(2q)."""
    s = S.leapfrog()
    c, d = s.ramps()
    assert c.tolist() == [0.5] and d.tolist() == [0.5]
    assert s.distance() == 0.0


@pytest.mark.parametrize("path_scheme", S.all_recommended(), ids=lambda ps: f"{ps[0].parent.name}-{ps[1].q}")
def test_file_notation_round_trip(path_scheme):
    _, s = path_scheme
    a_file, b_file = S.to_file_notation(s)
    back = S.from_file_notation(a_file, b_file, s.order)
    np.testing.assert_allclose(back.a, s.a, atol=1e-15)
    np.testing.assert_allclose(back.b, s.b, atol=1e-15)


def test_all_recommended_parse():
    """63 schemes in 30 files, orders 2-8, q = 1..30, each in its own folder's order."""
    rows = S.all_recommended()
    assert len(rows) == 63
    for path, s in rows:
        assert path.parent.name == f"order_n{s.order}"
        assert path.name == f"schemes_q{s.q}.out"


# Upstream 'Distance from the origin' lines that do NOT match their own
# scheme's coefficients: (order, q, upstream x).  All order 8, none used by
# the paper's tables or figures.  For q = 22, 24, 28 the printed x is exactly
# the distance of an earlier version of the scheme in the upstream archive
# (All_schemes/*/order_n8), whose parameters differ by 2e-5..3e-3: the
# coefficients were refined after the distance was written.  For q = 17 the
# archive copy has identical parameters and gives our value, so the printed
# x is unexplained.  q = 21, 23 differ by ~1e-8..1e-7 (presumably the same
# cause; not traced).
KNOWN_STALE_DISTANCES = {
    (8, 17, 1.0111108703378144), (8, 21, 0.6892508850), (8, 22, 1.5953973015),
    (8, 22, 0.8916492825), (8, 23, 1.0845069915), (8, 24, 1.5101836959),
    (8, 28, 1.7107677098),
}


def _is_known_stale(s):
    return any(s.order == n and s.q == q and abs(s.reference_distance - x) < 1e-9
               for n, q, x in KNOWN_STALE_DISTANCES)


@pytest.mark.parametrize("path_scheme", S.all_recommended(), ids=lambda ps: f"{ps[0].parent.name}-{ps[1].q}")
def test_distance_matches_upstream(path_scheme):
    """Eq. (23) x-bar vs the upstream 'Distance from the origin' line.

    Exact (1e-12) for 56 of 63 schemes.  The 7 known-stale upstream values
    must still *differ* -- if upstream fixes them this test says so.
    """
    _, s = path_scheme
    if s.reference_distance is None:
        pytest.skip("no distance recorded upstream")
    if _is_known_stale(s):
        assert abs(s.distance() - s.reference_distance) > 1e-9
    else:
        assert s.distance() == pytest.approx(s.reference_distance, rel=1e-12, abs=1e-15)


def test_known_stale_list_is_complete():
    hits = [s for _, s in S.all_recommended() if s.reference_distance is not None and _is_known_stale(s)]
    assert len(hits) == len(KNOWN_STALE_DISTANCES)


def test_ramps_are_symmetric():
    """d_i = c_{q+1-i} for a palindromic scheme (Tables 2-3 headers)."""
    for _, s in S.all_recommended():
        c, d = s.ramps()
        np.testing.assert_allclose(d, c[::-1], atol=1e-12)


# ----------------------------------------------- disagreement with the paper

def test_table2_matches_repo_to_1e10():
    """Table 2 (q=6) vs the repo's 2nd-highest-efficiency q=6 scheme.

    They agree to ~1e-10 (max over the six c_i), not to the 18 digits printed.
    Recorded, not fixed.
    """
    c_repo, _ = S.recommended_q6().ramps()
    diff = np.max(np.abs(c_repo - np.array(S.TABLE2_C)))
    assert 1e-12 < diff < 1e-9


def test_table3_differs_from_repo():
    """Table 3 (q=14) and the repo's scheme differ by ~1e-6 in the ramps.

    Both are valid order-6 schemes (see test_effts_bch); at q = 14 the order-6
    solutions form a family with free parameters (paper Table 1), and both
    sit at essentially the same Err_6.  This records the disagreement so it
    cannot change silently.
    """
    c_repo, _ = S.recommended_q14().ramps()
    diff = np.max(np.abs(c_repo - np.array(S.TABLE3_C)))
    assert 1e-7 < diff < 1e-5


def test_paper_origin_typo():
    """Sec. 2.3 says origin c_i = d_i = 1/q; Eq. (23) and sum(c) = 1/2 need 1/(2q).

    The equal-ramp scheme with c_i = 1/(2q) is a valid scheme (sum = 1/2)
    with x-bar = 0; c_i = 1/q is not a valid scheme at all.
    """
    for q in (1, 3, 6):
        s = S.from_ramps(tuple([1 / (2 * q)] * q), 2)
        assert s.distance() == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        S.from_ramps(tuple([1 / 6] * 6), 2)


def test_scheme_validation_raises():
    with pytest.raises(ValueError):
        S.Scheme((0.5, 0.6), (1.0,), 2)  # sum(a) != 1
    with pytest.raises(ValueError):
        S.Scheme((0.3, 0.7), (1.0,), 2)  # not palindromic
