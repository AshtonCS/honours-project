"""Hold-out check of the power-law extrapolation (pffit.py), on synthetic data
whose true law is known."""

import json
import math

import pytest

from plsimcount import pffit


def _rows(law, sizes, scheme="s", gb="cz", order=2):
    """Measured rows as pfcompare writes them (merged and unmerged copies)."""
    out = []
    for n in sizes:
        for merged in (False, True):
            out.append({"instance": "simcount", "r_source": "measured", "scheme": scheme, "gatebase": gb,
                        "family": "suzuki", "order": order, "n": n, "r": law(n), "merged": merged})
    return out


def test_exact_power_law_is_predicted_exactly():
    """Measured r = n^2: the fit recovers (a, b) = (1, 2) and predicts 100 at
    n = 10 (101 only if a*10^b lands a rounding error above 100). For the
    unrounded law r = 3 n^1.7 the fit is exact and the prediction is
    ceil(3 * 10^1.7) = ceil(150.356...) = 151."""
    rows = _rows(lambda n: n ** 2, [4, 5, 6, 7, 8, 10])
    [c] = pffit.run_checks(rows, [{"fit": [5, 6, 7, 8], "predict": 10}])
    assert c.a == pytest.approx(1.0, rel=1e-12)
    assert c.b == pytest.approx(2.0, rel=1e-12)
    # ceil of a float that is 100 up to rounding: allow 100 or 101, then check by hand
    assert c.r_predicted in (100, 101)
    assert c.r_measured == 100
    a, b, r = pffit.predict([5, 6, 7, 8], [3.0 * n ** 1.7 for n in (5, 6, 7, 8)], 10)
    assert (a, b) == (pytest.approx(3.0), pytest.approx(1.7))
    assert r == math.ceil(3.0 * 10 ** 1.7) == 151


def test_rounded_law_relative_error_is_small_and_signed():
    """Measured r = ceil(2 n^1.5). Hand check at n = 10: ceil(2 * 31.62) = 64.
    The fit on the rounded points is close to (2, 1.5), so the prediction is 64
    or 65, i.e. relative error 0 or +1/64."""
    law = lambda n: math.ceil(2 * n ** 1.5)  # noqa: E731
    rows = _rows(law, [4, 5, 6, 7, 8, 10])
    [c] = pffit.run_checks(rows, [{"fit": [6, 7, 8], "predict": 10}])
    assert c.r_measured == 64
    assert c.r_predicted in (64, 65)
    assert c.rel_error == pytest.approx((c.r_predicted - 64) / 64)


def test_deviation_from_power_law_shows_up_as_error():
    """If r jumps at the held-out size the check must see it: r = n^2 on the
    fit sizes but r(10) = 80 measured -> prediction 100, error +25%."""
    law = lambda n: 80 if n == 10 else n ** 2  # noqa: E731
    [c] = pffit.run_checks(_rows(law, [5, 6, 7, 8, 10]), [{"fit": [5, 6, 7, 8], "predict": 10}])
    assert c.r_predicted in (100, 101)
    assert c.rel_error == pytest.approx((c.r_predicted - 80) / 80)
    assert c.rel_error >= 0.25


def test_missing_size_and_bad_checks_raise():
    rows = _rows(lambda n: n ** 2, [5, 6, 7, 8])
    with pytest.raises(ValueError, match="no measured r"):
        pffit.run_checks(rows, [{"fit": [5, 6, 7, 8], "predict": 10}])
    with pytest.raises(ValueError, match="not a hold-out"):
        pffit.predict([5, 6, 7], [25, 36, 49], 7)
    with pytest.raises(ValueError, match="two sizes"):
        pffit.predict([5], [25], 7)


def test_inconsistent_merged_rows_raise():
    rows = _rows(lambda n: n ** 2, [5, 6])
    rows[1]["r"] = 999
    with pytest.raises(ValueError, match="two measured r"):
        pffit.measured_r(rows)


def test_ignores_fit_and_bound_rows():
    rows = _rows(lambda n: n ** 2, [5, 6, 7, 8, 10])
    rows.append({"instance": "simcount", "r_source": "fit", "scheme": "s", "gatebase": "cz",
                 "family": "suzuki", "order": 2, "n": 10, "r": 1, "merged": True})
    rows.append({"instance": "simcount", "r_source": "com", "scheme": "s", "gatebase": "cz",
                 "family": "suzuki", "order": 2, "n": 10, "r": 7, "merged": True})
    assert pffit.measured_r(rows)[("s", "cz")][10] == 100


def test_main_writes_outputs(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rows = _rows(lambda n: n ** 2, [4, 5, 6, 7, 8, 10])
    (src / "results.json").write_text(json.dumps({"meta": {"config": {"schemes": ["s"]}, "provenance": {}},
                                                  "rows": rows}))
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_text(json.dumps({"source": "../src/results.json",
                                                 "checks": [{"fit": [4, 5, 6, 7], "predict": 8}]}))
    pffit.main([str(out / "config.json")])
    for f in ("results.json", "results.csv", "table.md", "environment.txt"):
        assert (out / f).exists()
    res = json.loads((out / "results.json").read_text())
    assert res["rows"][0]["r_measured"] == 64
