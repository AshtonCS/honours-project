"""Hand-checkable cases for pfmean's summaries."""

import math

import pytest

from pysimcount import pfmean


def test_geomean_is_symmetric_for_reciprocal_gains():
    # a 2x gain on one seed and a 2x loss on another: arithmetic mean 1.25, geometric mean 1
    assert pfmean.mean([2.0, 0.5]) == 1.25
    assert math.isclose(pfmean.geomean([2.0, 0.5]), 1.0)


def test_geomean_known_value():
    assert math.isclose(pfmean.geomean([1.0, 4.0, 16.0]), 4.0)


def test_geomean_rejects_non_positive():
    with pytest.raises(ValueError):
        pfmean.geomean([1.0, 0.0])
    with pytest.raises(ValueError):
        pfmean.mean([])


def test_summarise_count_and_ratio():
    # Lie-Trotter n = 4 style: one low seed pulls the mean below the median
    per = {1: 8016.0, 2: 39059.0, 3: 39059.0, 4: 52631.0}
    s = pfmean.summarise(per, "count")
    assert s["centre"] == (8016 + 39059 + 39059 + 52631) / 4
    assert (s["median"], s["min"], s["max"]) == (39059.0, 8016.0, 52631.0)
    r = pfmean.summarise({1: 2.0, 2: 0.5}, "ratio")
    assert math.isclose(r["centre"], 1.0) and r["median"] == 1.25
    with pytest.raises(ValueError):
        pfmean.summarise(per, "other")
