"""r vs n by grouping figure (pfgroupfigure.py), on small hand-made runs."""

import json

import pytest

from pysimcount import pfgroupfigure as F

VARIANTS = ["xyz", "twogroup-even-out"]


def _row(seed, scheme, n, variant, r, merged=True):
    return {"seed": seed, "scheme": scheme, "n": n, "variant": variant, "merged": merged, "r": r}


def _run(seed, rows):
    cfg = {"seed": seed, "name": f"s{seed}", "schemes": ["suzuki4"], "variants": VARIANTS}
    return {"meta": {"config": cfg, "provenance": {"git_commit": "abc", "git_dirty": False}}, "rows": rows}


# seed -> {(n, variant): r}
R = {1: {(4, "xyz"): 44, (4, "twogroup-even-out"): 39, (6, "xyz"): 66, (6, "twogroup-even-out"): 60},
     2: {(4, "xyz"): 50, (4, "twogroup-even-out"): 100, (6, "xyz"): 70, (6, "twogroup-even-out"): 62},
     3: {(4, "xyz"): 40, (4, "twogroup-even-out"): 40, (6, "xyz"): 64, (6, "twogroup-even-out"): 58}}


def _runs():
    runs = {}
    for s, rs in R.items():
        rows = [_row(s, "suzuki4", n, v, r) for (n, v), r in rs.items()]
        rows += [_row(s, "suzuki4", n, v, 1, merged=False) for (n, v) in rs]  # unmerged rows ignored
        runs[s] = _run(s, rows)
    return runs


def test_median_min_max_by_hand():
    """n=4 xyz: r = 44, 50, 40 -> median 44, min 40, max 50.
    n=4 two-group: 39, 100, 40 -> median 40, min 39, max 100."""
    st = F.r_stats(_runs(), ["suzuki4"], VARIANTS)
    assert st[("suzuki4", "xyz")][0] == (4, 44, 40, 50)
    assert st[("suzuki4", "twogroup-even-out")][0] == (4, 40, 39, 100)
    assert st[("suzuki4", "xyz")][1] == (6, 66, 64, 70)


def test_missing_row_raises():
    runs = _runs()
    runs[2]["rows"] = [r for r in runs[2]["rows"] if not (r["n"] == 6 and r["variant"] == "xyz")]
    with pytest.raises(ValueError, match="seed 2"):
        F.r_stats(runs, ["suzuki4"], VARIANTS)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="no rows"):
        F.r_stats(_runs(), ["suzuki6"], VARIANTS)


def test_figure_coordinates_and_legend():
    st = F.r_stats(_runs(), ["suzuki4"], VARIANTS)
    tex = F.figure_tex(st, ["suzuki4"], "twogroup-even-out", {"suzuki4": "Suzuki S4"}, ["hdr"])
    # xyz at n / 1.025, two-group at n * 1.025; error bars are offsets from the median
    assert f"({4 / 1.025:.4f},44) += (0,6) -= (0,4)" in tex
    assert f"({4 * 1.025:.4f},40) += (0,60) -= (0,1)" in tex
    assert tex.count("\\addlegendentry{Suzuki S4}") == 1
    assert "forget plot" in tex                       # two-group line has no legend entry of its own
    assert "\\begin{loglogaxis}" in tex and tex.startswith("% hdr")


def test_main_writes_figure(tmp_path):
    names = []
    for s, run in _runs().items():
        p = tmp_path / f"seed{s}.json"
        p.write_text(json.dumps(run))
        names.append(p.name)
    cfg = tmp_path / "aggregate.json"
    cfg.write_text(json.dumps({"runs": names, "out": "aggregate"}))
    F.main([str(cfg), "--schemes", "suzuki4"])
    out = tmp_path / "aggregate" / "figures"
    assert (out / "r-vs-n-twogroup-even-out.tex").exists()
    assert "\\input{r-vs-n-twogroup-even-out.tex}" in (out / "all.tex").read_text()
