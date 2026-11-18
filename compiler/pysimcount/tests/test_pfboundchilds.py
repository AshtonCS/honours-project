"""pfboundchilds: bound r per seed, the record format figleader reads, and the relabelled key."""

import json

import pytest

from pysimcount import bounds, figinset, figleader, pfboundchilds as B, pfchilds
from pysimcount.compare import GATEBASES
from pysimcount.pfcompare import formula_registry, instance_hamiltonian

BOUND = {"kind": "min", "budget": "count", "constant": "childs"}


@pytest.mark.parametrize("key,order", [("lie-trotter", 1), ("suzuki2", 2), ("suzuki4", 4), ("suzuki8", 8)])
def test_bound_r_is_simcount_min_for_suzuki(key, order):
    for n in (4, 11, 50):
        H = instance_hamiltonian({"instances": "numpy", "seed": 3}, n)
        got = B.bound_r(key, 3, n, BOUND)
        for g in ("cz", "ct"):
            assert got[g] == bounds.minimised(order, H, float(n), GATEBASES[g][0])


def test_count_bound_is_seed_independent():
    # m = 4n, Lambda = 1 (couplings; |h_i| < 1), t = n: nothing depends on the fields
    assert len({B.bound_r("mo-q14", s, 20, BOUND)["cz"] for s in (1, 2, 3)}) == 1


def _record(points_by_scheme):
    series = {k: {"measured": [B.point(n, per, {s: 1 for s in per}) for n, per in pts], "extrapolated": []}
              for k, pts in points_by_scheme.items()}
    cfg = {"bound": BOUND, "seeds": [1, 2], "sizes": [4, 500], "line_style": "suzuki-dashed"}
    return B.make_record(cfg, {"id": "x", "title": "t"}, "cnot", "Clifford+Rz (cz)", series,
                         list(series), [], ["h"])


def test_record_feeds_figleader_and_figinset():
    rec = _record({"suzuki4": [(4, {1: 100, 2: 100}), (500, {1: 9000, 2: 9000})],
                   "bm-q10": [(4, {1: 50, 2: 50}), (500, {1: 7000, 2: 8000})]})
    rows = figleader.leaderboard(rec, {}, 500)
    assert [r["scheme"] for r in rows] == ["bm-q10", "suzuki4"]
    assert rows[0]["value"] == 7500 and rows[0]["min"] == 7000 and rows[0]["wins"] == 2
    tex = B.figure_tex(rec, {"ylabel": "CNOT"}, formula_registry())
    _, plots, trailing = figinset.parse_figure(tex)
    assert len(plots) == 2 and plots[0].points() == [(4.0, 100.0), (500.0, 9000.0)]
    assert any("computed at every" in line for line in trailing)


def test_key_replaced_exactly_once():
    rec = {"line_style": "suzuki-dashed", "fit_sizes": [4, 500]}
    old = "\n".join(pfchilds.legend_key("suzuki-dashed", "measured $r$", "extrapolated (fit on $n = 4$--$500$)"))
    out = B._replace_key("a\n" + old + "\nb", rec)
    assert "measured" not in out and "computed at every" in out
    with pytest.raises(ValueError):
        B._replace_key("no key here", rec)


def test_r_table_cache_refuses_other_settings(tmp_path):
    cfg = {"bound": BOUND, "seeds": [1], "sizes": [4], "schemes": ["suzuki2"], "workers": 1}
    t = B.r_table(cfg, tmp_path, log=lambda *_: None)
    assert t["cz"]["suzuki2"][4][1] == B.bound_r("suzuki2", 1, 4, BOUND)["cz"]
    assert json.loads((tmp_path / "r-table.json").read_text())["settings"]["sizes"] == [4]
    with pytest.raises(ValueError, match="move it aside"):
        B.r_table({**cfg, "sizes": [5]}, tmp_path)
