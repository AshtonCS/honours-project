"""figgroupline summarises both groupings from the same runs, by hand, and agrees with the records."""

import json
from pathlib import Path

import pytest

from pysimcount import figdrop, figgroupline, figuniform
from pysimcount.pfcompare import formula_registry

# One formula, two seeds, n = 4 and 6. (r, CNOT per step) per grouping, seed, n:
CASES = {("xyz", 1, 4): (10, 400), ("xyz", 2, 4): (12, 400), ("xyz", 1, 6): (20, 600), ("xyz", 2, 6): (22, 600),
         ("twogroup-even-out", 1, 4): (9, 150), ("twogroup-even-out", 2, 4): (11, 150),
         ("twogroup-even-out", 1, 6): (20, 220), ("twogroup-even-out", 2, 6): (20, 220)}


def _runs() -> dict[int, dict]:
    out = {1: {"rows": []}, 2: {"rows": []}}
    for (g, s, n), (r, step) in CASES.items():
        for merged in (True, False):
            out[s]["rows"].append({"variant": g, "merged": merged, "scheme": "bm-q10", "n": n, "r": r,
                                   "cnot_per_step": step if merged else step + 7,
                                   "cnot_total": (step if merged else step + 7) * r})
    out[1]["rows"].append({"variant": "twogroup-odd-out", "merged": True, "scheme": "bm-q10", "n": 4, "r": 1,
                           "cnot_per_step": 1, "cnot_total": 1})  # other variant: ignored
    return out


def test_series_by_hand():
    d = figgroupline.series(_runs(), ["bm-q10"], [4, 6])
    x4 = d["xyz"]["bm-q10"][0]
    assert x4["r"] == {"mean": 11, "min": 10, "max": 12}
    assert x4["cnot"] == {"mean": 4400, "min": 4000, "max": 4800}
    t6 = d["twogroup-even-out"]["bm-q10"][1]
    assert t6["cnot"] == {"mean": 4400, "min": 4400, "max": 4400}


def test_ratio_range_by_hand():
    d = figgroupline.series(_runs(), ["bm-q10"], [4, 6])
    # n = 6: r 21 / 20; CNOT (600 x 21) / (220 x 20) = 12600 / 4400
    assert figgroupline.ratio_range(d, "r", 6) == pytest.approx((1.05, 1.05))
    assert figgroupline.ratio_range(d, "cnot", 6) == pytest.approx((12600 / 4400,) * 2)


def test_missing_seed_and_bad_total_raise():
    runs = _runs()
    runs[2]["rows"] = [r for r in runs[2]["rows"] if not (r["variant"] == "xyz" and r["n"] == 6)]
    with pytest.raises(ValueError, match="seeds"):
        figgroupline.series(runs, ["bm-q10"], [4, 6])
    runs = _runs()
    runs[1]["rows"][0]["cnot_total"] += 1
    with pytest.raises(ValueError, match="per step x r"):
        figgroupline.series(runs, ["bm-q10"], [4, 6])


def test_record_check():
    d = figgroupline.series(_runs(), ["bm-q10"], [4])
    rec = {"series": {"bm-q10": {"measured": [{"n": 4, "mean": 4400, "min": 4000, "max": 4800, "mean_r": 11}]}}}
    figgroupline.check_against_record(d["xyz"], rec, "rec")
    rec["series"]["bm-q10"]["measured"][0]["mean_r"] = 11.5
    with pytest.raises(ValueError, match="differs from the record"):
        figgroupline.check_against_record(d["xyz"], rec, "rec")


CFG = {"sizes": [4, 6], "xmin": 3.6, "xmax": 7, "width_cm": 6, "height_cm": 5, "gap_cm": 2, "mark_size": "1pt",
       "legend_columns": 4, "labels": {"bm-q10": "BM"},
       "r_panel": {"title": "a", "ylabel": "r", "ymin": 5, "ymax": 50, "ytick": [10, 20]},
       "cnot_panel": {"title": "b", "ylabel": "c", "ymin": 1000, "ymax": 2e4, "ytick": [3000]}}


def test_figure_styles_legend_and_range():
    d = figgroupline.series(_runs(), ["bm-q10"], [4, 6])
    tex = figgroupline.figure_tex(d, CFG, ["% derived: x"], formula_registry())
    assert tex.count("\\begin{loglogaxis}") == 2
    assert tex.count("\\addlegendentry{BM}") == 1  # one legend, in panel (b)
    # no open markers: they mean "extrapolated" in the RQ1 figures
    assert not any("fill=white" in l for l in tex.splitlines() if "\\addplot" in l)
    assert "densely dotted" in tex and "1.05--1.05$\\times$" in tex
    assert "at={(-0.1667,-0.2)}" in tex  # legend centred under both panels: ((12 + 2)/2 - 8) / 6
    assert not figuniform.in_scope(tex) and figdrop.has_fixed_y(tex) and figdrop.is_derived(tex)
    with pytest.raises(ValueError, match="outside y-range"):
        figgroupline.figure_tex(d, {**CFG, "r_panel": {**CFG["r_panel"], "ymin": 11}}, [], formula_registry())


def test_real_runs_reproduce_both_records():
    here = Path(__file__).resolve().parents[2] / "results" / "np-childs-format"
    if not (here / "grouping-lines.json").exists():
        pytest.skip("results not present")
    cfg = json.loads((here / "grouping-lines.json").read_text())
    from pysimcount.pfseeds import load_runs
    agg = here / cfg["runs"]
    runs = load_runs([agg.parent / p for p in json.loads(agg.read_text())["runs"]])
    d = figgroupline.series(runs, cfg["schemes"], cfg["sizes"])
    figgroupline.check_against_record(d["xyz"], json.loads((here / cfg["xyz_record"]).read_text()), "xyz")
    figgroupline.check_against_record(d["twogroup-even-out"],
                                      json.loads((here / cfg["twogroup_record"]).read_text()), "twogroup")
    lo, hi = figgroupline.ratio_range(d, "cnot", 10)
    assert 2.6 < lo < hi < 3.0
