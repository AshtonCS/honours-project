"""Tests for the bound / measured r figure (pysimcount.pfboundratio)."""

import json
from pathlib import Path

import pytest

from pysimcount import figdrop, pfboundratio

RESULTS = Path(__file__).resolve().parents[2] / "results"


def _run(seed, rows, dirty=False):
    """A minimal results.json: what pfseeds.bound_ratios and load_set read."""
    return {"meta": {"config": {"seed": seed, "instances": "numpy"}, "provenance": {"git_dirty": dirty}},
            "rows": [{"instance": "numpy", "gatebase": "cz", "merged": True, "scheme": "suzuki4", "n": n,
                      "r_source": src, "r": r} for n, src, r in rows]}


def _write(tmp_path, name, seeds, rows_by_seed, dirty=False):
    paths = []
    for s in seeds:
        p = tmp_path / f"{name}-seed{s}.json"
        p.write_text(json.dumps(_run(s, rows_by_seed[s], dirty)))
        paths.append(p)
    return paths


def test_geometric_mean_by_hand(tmp_path):
    # Seed 1: com/measured = 400/10 = 40; seed 2: 1000/10 = 100. Geometric
    # mean sqrt(40 * 100) = 63.25 (the arithmetic mean would be 70).
    rows = {1: [(4, "measured", 10), (4, "com", 400), (13, "fit", 50), (13, "com", 900)],
            2: [(4, "measured", 10), (4, "com", 1000), (13, "fit", 50), (13, "com", 900)]}
    runs = pfboundratio.load_set(_write(tmp_path, "a", [1, 2], rows), None)
    recs = pfboundratio.ratios({"a": runs}, ["suzuki4"], ["com"], [1, 2])
    assert len(recs) == 1  # the n = 13 row divides by a fitted r and is dropped
    x = recs[0]
    assert (x["n"], x["min"], x["max"]) == (4, 40.0, 100.0)
    assert x["centre"] == pytest.approx(4000 ** 0.5)


def test_dirty_run_set_needs_a_reason(tmp_path):
    rows = {1: [(4, "measured", 10), (4, "com", 400)]}
    paths = _write(tmp_path, "d", [1], rows, dirty=True)
    with pytest.raises(ValueError, match="accept_dirty"):
        pfboundratio.load_set(paths, None)
    assert sorted(pfboundratio.load_set(paths, "reason given")) == [1]


def test_a_size_in_two_run_sets_raises(tmp_path):
    rows = {1: [(4, "measured", 10), (4, "com", 400)]}
    a = pfboundratio.load_set(_write(tmp_path, "a", [1], rows), None)
    b = pfboundratio.load_set(_write(tmp_path, "b", [1], rows), None)
    with pytest.raises(ValueError, match="run sets"):
        pfboundratio.ratios({"a": a, "b": b}, ["suzuki4"], ["com"], [1])


def test_figure_has_one_series_per_bound(tmp_path):
    rows = {1: [(4, "measured", 10), (4, "com", 400), (4, "tte", 60), (5, "measured", 20), (5, "com", 900),
                (5, "tte", 150)]}
    runs = pfboundratio.load_set(_write(tmp_path, "a", [1], rows), None)
    recs = pfboundratio.ratios({"a": runs}, ["suzuki4"], ["com", "tte"], [1])
    tex = pfboundratio.figure_tex(recs, "suzuki4", ["com", "tte"], ["test"])
    series, _ = figdrop.parse_series(tex)
    assert [s.legend for s in series] == ["Commutator bound (2018)", "Commutator bound (2021)"]


@pytest.mark.skipif(not (RESULTS / "np-main/seed10/results.json").exists(), reason="np-main runs not on disk")
def test_matches_np_mean_bounds_table():
    # np-mean/main-bounds.md (pfmean, same pairing): S4 tte n=4 6.031, com n=10 84.075.
    runs = pfboundratio.load_set([RESULTS / f"np-main/seed{s}/results.json" for s in range(1, 11)], None)
    recs = {(x["bound"], x["n"]): x["centre"]
            for x in pfboundratio.ratios({"np-main": runs}, ["suzuki4"], ["tte", "com"], list(range(1, 11)))}
    assert recs[("tte", 4)] == pytest.approx(6.031, abs=5e-4)
    assert recs[("com", 10)] == pytest.approx(84.075, abs=5e-4)
