"""pffitwindow applies pfchilds's extrapolation rule per fit window, by hand, and reproduces the record."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from pysimcount import pffitwindow


def test_labels():
    assert pffitwindow.label([5, 6, 7, 8, 9, 10, 11]) == "5-11"
    assert pffitwindow.label([5, 6, 7, 8, 10]) == "5-10 (no 9)"
    assert pffitwindow._tex_label([5, 6, 7, 8, 10]) == "$n = 5$--$10$ (no 9)"


def test_window_counts_by_hand():
    # r = 2n on both seeds for "a" (a = 2, b = 1), r = n^2 for "b" (a = 1, b = 2); per step 10 n and 3 n
    rs = {"a": {n: {1: 2 * n, 2: 2 * n} for n in (4, 6, 8)}, "b": {n: {1: n * n, 2: n * n} for n in (4, 6, 8)}}
    alpha = {"a": (Fraction(10), Fraction(0)), "b": (Fraction(3), Fraction(0))}
    got = pffitwindow.window_counts(rs, alpha, [4, 6, 8], [100], [1, 2])
    assert got["b"] == pytest.approx({"a": 1.0, "b": 2.0})
    # ceil guards against 200.0000001 by tolerance only through the fit; check the counts are near the exact values
    assert got["count"][100]["a"] in (10 * 100 * 200, 10 * 100 * 201)
    assert got["count"][100]["b"] in (3 * 100 * 10000, 3 * 100 * 10001)
    r = pffitwindow.ratios({"a": 400, "b": 100})
    assert r == {"a": 4.0, "b": 1.0}


def test_missing_seed_raises():
    rs = {"a": {4: {1: 8}, 6: {1: 12, 2: 12}}}
    with pytest.raises(ValueError, match="seeds"):
        pffitwindow.window_counts(rs, {"a": (Fraction(1), Fraction(0))}, [4, 6], [10], [1, 2])


def test_refuses_to_overwrite_its_config(tmp_path):
    cfg = tmp_path / "x.json"
    cfg.write_text(json.dumps({"out": "x"}))
    with pytest.raises(ValueError, match="would overwrite the config"):
        pffitwindow.main([str(cfg)])


def test_real_record_window_reproduces_and_tables_written(tmp_path):
    here = Path(__file__).resolve().parents[2] / "results" / "np-childs-format"
    if not (here / "fit-windows.json").exists():
        pytest.skip("results not present")
    cfg = json.loads((here / "fit-windows.json").read_text())
    cfg["out"] = str(tmp_path / "t")  # absolute: Path join keeps it, so the real outputs are untouched
    res = pffitwindow.run(cfg, here)  # raises unless the record's own window reproduces the record
    by = {w["label"]: w for w in res["windows"]}
    assert by["5-10 (no 9)"]["cheapest"]["500"] == "morales-m8"
    assert by["6-11"]["cheapest"]["500"] == "mo-q14"
    assert all(w["ratio"]["500"]["suzuki4"] > 4 for w in res["windows"])
    assert (tmp_path / "t.tex").exists() and (tmp_path / "t.md").exists()
