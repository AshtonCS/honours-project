"""Aggregation of runs over seeds (pfseeds.py), on small hand-made runs."""

import json

import pytest

from plsimcount import pfseeds as S


def _row(scheme, order, n, r, cnot_per_step, gb="cz", merged=True, src="measured"):
    return {"instance": "simcount", "scheme": scheme, "order": order, "n": n, "gatebase": gb, "merged": merged,
            "r_source": src, "r": r, "cnot_total": cnot_per_step * r, "gates_total": 3 * cnot_per_step * r,
            "t_total": 0, "search_seconds": 1.0}


def _run(seed, rows, dirty=False):
    return {"meta": {"config": {"seed": seed, "schemes": ["suzuki2", "suzuki4", "mo-q6"], "name": f"s{seed}"},
                     "provenance": {"git_commit": "abc", "git_dirty": dirty}}, "rows": rows}


# r per seed for (scheme, n); CNOT per step: suzuki2 10, suzuki4 50, mo-q6 60
R = {
    1: {("suzuki2", 4): 100, ("suzuki4", 4): 10, ("mo-q6", 4): 8},
    2: {("suzuki2", 4): 120, ("suzuki4", 4): 12, ("mo-q6", 4): 11},
    3: {("suzuki2", 4): 90, ("suzuki4", 4): 14, ("mo-q6", 4): 9},
}
ORDER = {"suzuki2": 2, "suzuki4": 4, "mo-q6": 4}
CNOT = {"suzuki2": 10, "suzuki4": 50, "mo-q6": 60}


def _runs():
    return {s: _run(s, [_row(sc, ORDER[sc], n, r, CNOT[sc]) for (sc, n), r in R[s].items()]) for s in R}


def test_median_min_max_by_hand():
    """suzuki2 r over seeds 1,2,3 = 100,120,90 -> median 100, min 90, max 120, ratio 4/3.
    CNOT totals = 10 r -> 1000, 1200, 900."""
    aggs = {(x.scheme, x.n): x for x in S.aggregate(_runs(), "cnot", "cnot_total", "cz")}
    x = aggs[("suzuki2", 4)]
    assert (x.median, x.lo, x.hi) == (1000, 900, 1200)
    assert x.ratio == pytest.approx(4 / 3)
    assert x.values == {1: 1000, 2: 1200, 3: 900}


def test_ranking_flip_detected():
    """CNOT totals at n=4: suzuki4 = 50 r -> 500, 600, 700; mo-q6 = 60 r -> 480, 660, 540.
    Seed 1: mo-q6 (480) < suzuki4 (500). Seed 2: 660 > 600 -> flip. Seed 3: 540 < 700 -> no flip.
    suzuki2 (1000, 1200, 900) is above both on every seed -> no flips involving it."""
    aggs = S.aggregate(_runs(), "cnot", "cnot_total", "cz")
    flips = S.ranking_flips(aggs, reference=1)
    assert [(f.a, f.b, f.seed) for f in flips] == [("mo-q6", "suzuki4", 2)]
    f = flips[0]
    assert (f.ref_a, f.ref_b, f.seed_a, f.seed_b) == (480, 500, 660, 600)
    assert f.order_a == f.order_b == 4
    assert S.best_formula(aggs, 1)[(4, "measured")] == "mo-q6"
    assert S.best_formula(aggs, 2)[(4, "measured")] == "suzuki4"


def test_ties_are_not_flips():
    runs = {1: _run(1, [_row("suzuki4", 4, 4, 10, 50), _row("mo-q6", 4, 4, 10, 50)]),
            2: _run(2, [_row("suzuki4", 4, 4, 10, 50), _row("mo-q6", 4, 4, 11, 50)])}
    assert S.ranking_flips(S.aggregate(runs, "cnot", "cnot_total", "cz"), 1) == []


def test_missing_seed_point_raises():
    runs = _runs()
    runs[3]["rows"] = [r for r in runs[3]["rows"] if r["scheme"] != "mo-q6"]
    with pytest.raises(ValueError, match="expected"):
        S.aggregate(runs, "r", "r", "cz")


def test_load_runs_checks_configs_and_dirty(tmp_path):
    paths = []
    for s in (1, 2):
        p = tmp_path / f"s{s}.json"
        p.write_text(json.dumps(_run(s, [])))
        paths.append(p)
    assert sorted(S.load_runs(paths)) == [1, 2]
    bad = _run(3, [])
    bad["meta"]["config"]["schemes"] = ["suzuki2"]
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="more than the seed"):
        S.load_runs(paths + [tmp_path / "bad.json"])
    (tmp_path / "dirty.json").write_text(json.dumps(_run(4, [], dirty=True)))
    with pytest.raises(ValueError, match="uncommitted"):
        S.load_runs(paths + [tmp_path / "dirty.json"])


def test_compare_rows_ignores_only_timing():
    a = [_row("suzuki2", 2, 4, 100, 10)]
    b = [dict(a[0], search_seconds=99.0)]
    assert S.compare_rows(a, b, ("measured",)) == (1, [])
    c = [dict(a[0], r=101)]
    n, diffs = S.compare_rows(a, c, ("measured",))
    assert n == 1 and len(diffs) == 1 and "r 100 != 101" in diffs[0]
    n, diffs = S.compare_rows(a, [], ("measured",))
    assert n == 0 and "only in first" in diffs[0]


def test_figures_have_median_and_error_bars(tmp_path):
    """The A/B figures carry the median as the point and min/max as the bar:
    suzuki2 CNOT at n = 4 is (4, 1000) += (0, 200) -= (0, 100)."""
    root = S.write_figures(_runs(), tmp_path)
    tex = (root / "cnot" / "A-measured-by-order.tex").read_text()
    assert "(4,1000) += (0,200) -= (0,100)" in tex
    assert (root / "cnot" / "B-order4-by-formula.tex").exists()
    assert "\\input{cnot/A-measured-by-order.tex}" in (root / "all.tex").read_text()
