"""Morales et al.'s unprocessed order-8 formulas (effts/morales.py)."""

import numpy as np
import pytest

from effts import bch
from effts import morales as M


@pytest.mark.parametrize("make,q", [(M.y8m10, 21), (M.y8m10b, 21), (M.m8, 17), (M.m7, 15)])
def test_order_8_conditions(make, q):
    """Every lower-order BCH coefficient vanishes, which catches a mistyped weight.

    Clean residues are ~1e-17. Measured effect of changing one digit of w_4:
    1st decimal -> 1e-2, 6th -> 1e-7, 14th -> 1e-15 (all fail the 1e-13 bar
    except the last). Errors below ~1e-12 are not caught, but they are also
    below double precision and cannot change any result."""
    s = make()
    assert s.q == q
    co = bch.error_coefficients(s)
    assert np.max(np.abs(co[1])) < 1e-13
    for k in (2, 4, 6):
        assert bch.err(s, k) < 1e-13


@pytest.mark.parametrize("pos", [3, 8, 12])  # 1st, 6th and 10th decimal of w_4
def test_a_mistyped_digit_is_caught(pos):
    """The check above has teeth: change one digit of one weight and it fails."""
    w = list(M.Y8M10)
    w[3] = w[3][:pos] + ("1" if w[3][pos] != "1" else "2") + w[3][pos + 1:]
    s = M.composition(tuple(w), 8, "corrupted")
    assert max(bch.err(s, k) for k in (2, 4, 6)) > 1e-13


def test_cycles_are_symmetric_leapfrog_steps():
    """c_i = d_i = w_i / 2: each cycle is one Leapfrog step (their Eq. 15)."""
    c, d = M.y8m10().ramps()
    np.testing.assert_allclose(c, d, atol=1e-15)
    np.testing.assert_allclose(2 * c[10], 1 - 2 * sum(float(x) for x in M.Y8M10), atol=1e-15)  # w_0, the middle step


def test_spot_values_read_by_eye():
    """Two values read off the paper's Table 2 by eye, guarding the extraction."""
    assert M.Y8M10[0] == "0.59358060400850625863514059265224"
    assert M.Y8M10[-1] == "0.098249557414708533273471906180643"


def test_m7_is_kahan_li_minimal_length():
    """Morales et al.: the m = 7 solution is Kahan & Li's s15odr8, 15 steps."""
    assert M.m7().q == 15
