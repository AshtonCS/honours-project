"""figratio's ratios are paired per seed and reproduce hand values; bad inputs raise."""

import json
import math
from pathlib import Path

import pytest

from pysimcount import figdrop, figratio, figuniform, pfchilds
from pysimcount.pfcompare import formula_registry

# Two formulas, two seeds, n = 4 measured; n = 100 extrapolated.
# Per step: suzuki4 42 n, bm-q10 82 n. Per-seed fits r = a n^b, ceil at n = 100:
#   suzuki4: seed 1 r = 1 * 100^1 = 100, seed 2 r = 2 * 100^1 = 200
#   bm-q10:  seed 1 r = 1 * 100^1 = 100, seed 2 r = 1 * 100^1 = 100
# Measured r at n = 4: suzuki4 10, 40; bm-q10 10, 5.


def _record() -> dict:
    def series(alpha: int, rs: dict[int, int], fits: dict[str, list[float]], extrap_rs: list[int]) -> dict:
        totals = [alpha * 4 * r for r in rs.values()]
        values = [alpha * 100 * r for r in extrap_rs]
        return {"measured": [{"n": 4, "mean": sum(totals) / 2, "min": min(totals), "max": max(totals),
                              "mean_r": sum(rs.values()) / 2}],
                "per_seed_fit": fits,
                "per_step_model": {"alpha": str(alpha), "beta": "0"},
                "extrapolated": [{"n": 100, "value": sum(values) / 2, "min": min(values), "max": max(values)}]}
    return {"id": "X", "kind": "xyz", "metric": "cnot", "seeds": [1, 2], "fit_sizes": [5, 10],
            "shown": ["suzuki4", "bm-q10"], "removed": ["suzuki8"],
            "series": {"suzuki4": series(42, {1: 10, 2: 40}, {"1": [1.0, 1.0], "2": [2.0, 1.0]}, [100, 200]),
                       "bm-q10": series(82, {1: 10, 2: 5}, {"1": [1.0, 1.0], "2": [1.0, 1.0]}, [100, 100])}}


def _runs() -> dict[int, dict]:
    def row(scheme: str, r: int, step: int) -> dict:
        return {"r_source": "measured", "gatebase": "cz", "merged": True, "scheme": scheme, "n": 4, "r": r,
                "cnot_per_step": step, "cnot_total": r * step}
    return {1: {"rows": [row("suzuki4", 10, 168), row("bm-q10", 10, 328)]},
            2: {"rows": [row("suzuki4", 40, 168), row("bm-q10", 5, 328)]}}


RATIO = {"width_cm": 10, "height_cm": 6, "xmin": 3, "xmax": 200, "ymin": 0.1, "ymax": 16, "ytick": [1, 5],
         "ylabel": "ratio", "mark_size": "1pt", "legend_columns": 2, "bar_sizes": [4, 100], "bar_note": "bars"}
PANELS = {"columns": 1, "width_cm": 5, "height_cm": 4, "gap_x_cm": 0.3, "gap_y_cm": 0.9,
          "xmin": 3, "xmax": 200, "ymin": 0.1, "ymax": 16, "ytick": [1, 5], "ylabel": "ratio",
          "mark_size": "1pt", "legend_pos": "north west", "note": "fit on {fit}, $q{=}1$",
          "panels": [{"title": "Suzuki", "schemes": ["suzuki4"], "labels": {"suzuki4": "short"}}]}


def test_paired_ratio_by_hand():
    # gains of 2 and 1/2 have geometric mean 1 (pfmean's reason for using it), not 1.25
    s = figratio.paired_summary({1: 2, 2: 1}, {1: 1, 2: 2})
    assert s == {"value": pytest.approx(1.0), "min": 0.5, "max": 2.0}


def test_measured_and_extrapolated_ratios_by_hand():
    r = figratio.ratio_points(_record(), _runs(), "bm-q10")["suzuki4"]
    # n = 4: seed 1 = (168 * 10) / (328 * 10) = 21/41; seed 2 = (168 * 40) / (328 * 5) = 168/41
    m = r["measured"][0]
    assert m["min"] == pytest.approx(21 / 41) and m["max"] == pytest.approx(168 / 41)
    assert m["value"] == pytest.approx(math.sqrt(21 * 168) / 41)
    # means do not pair: mean suzuki4 total / mean bm-q10 total = 4200 / 2460
    assert m["ratio_of_plotted"] == pytest.approx(4200 / 2460)
    # n = 100: seed 1 = 42/82, seed 2 = 2 * 42/82
    e = r["extrapolated"][0]
    assert e["min"] == pytest.approx(42 / 82) and e["max"] == pytest.approx(84 / 82)
    assert e["value"] == pytest.approx(math.sqrt(2) * 42 / 82)


def test_reference_is_exactly_one():
    r = figratio.ratio_points(_record(), _runs(), "bm-q10")["bm-q10"]
    for p in r["measured"] + r["extrapolated"]:
        assert (p["value"], p["min"], p["max"]) == (1.0, 1.0, 1.0)


def test_extrapolated_ratio_is_a_power_law_with_the_mean_exponent_difference():
    # With no ceiling effect, log ratio is linear in log n with slope mean_s(b_F,s - b_ref,s).
    rec = _record()
    rec["series"]["suzuki4"]["per_seed_fit"] = {"1": [1e6, 1.25], "2": [1e6, 1.75]}  # mean b 1.5
    rec["series"]["bm-q10"]["per_seed_fit"] = {"1": [1e6, 1.0], "2": [1e6, 1.0]}
    def centre(n: int) -> float:
        return figratio.paired_summary(figratio.extrapolated_per_seed(rec, "suzuki4", n),
                                       figratio.extrapolated_per_seed(rec, "bm-q10", n))["value"]
    assert math.log(centre(1000) / centre(100)) / math.log(10) == pytest.approx(0.5, abs=1e-4)


def test_absolute_points_are_the_records_values():
    a = figratio.absolute_points(_record())["suzuki4"]
    assert a["measured"][0]["value"] == 168 * 25 and a["extrapolated"][0]["value"] == 4200 * 150


def test_record_mismatch_raises():
    rec = _record()
    rec["series"]["suzuki4"]["measured"][0]["max"] += 1
    with pytest.raises(ValueError, match="do not reproduce"):
        figratio.ratio_points(rec, _runs(), "bm-q10")
    rec = _record()
    rec["series"]["suzuki4"]["extrapolated"][0]["min"] += 1
    with pytest.raises(ValueError, match="do not reproduce"):
        figratio.ratio_points(rec, _runs(), "bm-q10")


def test_unsupported_inputs_raise():
    with pytest.raises(ValueError, match="not in the figure"):
        figratio.ratio_points(_record(), _runs(), "suzuki8")
    rec = _record()
    rec["kind"] = "optimised"
    with pytest.raises(ValueError, match="optimised"):
        figratio.measured_per_seed(rec, _runs())
    with pytest.raises(ValueError, match="seeds differ"):
        figratio.paired_summary({1: 1}, {2: 1})


def test_clipped_bar_raises():
    points = figratio.ratio_points(_record(), _runs(), "bm-q10")
    with pytest.raises(ValueError, match="outside y-range"):
        figratio.check_range(points, 0.6, 16)


def test_panels_must_list_each_formula_once():
    shown = ["suzuki4", "suzuki6", "bm-q10"]
    figratio.check_panels([{"schemes": ["suzuki4", "suzuki6"]}], shown, "bm-q10")
    with pytest.raises(ValueError, match="expected"):
        figratio.check_panels([{"schemes": ["suzuki4"]}], shown, "bm-q10")
    with pytest.raises(ValueError, match="expected"):
        figratio.check_panels([{"schemes": ["suzuki4", "suzuki6", "suzuki6"]}], shown, "bm-q10")


def _ratio_figure() -> str:
    return figratio.ratio_tex(_record(), figratio.ratio_points(_record(), _runs(), "bm-q10"), "bm-q10", RATIO,
                              ["% Generated by pysimcount.figratio", "% derived: X.json"], formula_registry())


def _panel_figure() -> str:
    return figratio.panels_tex(_record(), figratio.ratio_points(_record(), _runs(), "bm-q10"), "bm-q10", PANELS,
                               ["% Generated by pysimcount.figratio", "% derived: X.json"], formula_registry())


def test_ratio_figure_draws_reference_at_one_and_bars_where_asked():
    tex = _ratio_figure()
    assert "(3,1) (200,1)" in tex and "Blanes \\& Moan q=10 (reference)" in tex
    assert tex.count("+=") == 2  # suzuki4 at n = 4 and n = 100; the joining point has none
    small = {**RATIO, "bar_sizes": [4]}
    tex = figratio.ratio_tex(_record(), figratio.ratio_points(_record(), _runs(), "bm-q10"), "bm-q10", small,
                             ["% derived: X.json"], formula_registry())
    assert tex.count("+=") == 1


def test_panel_figure_uses_short_label_and_keeps_literal_braces_in_note():
    tex = _panel_figure()
    assert "\\addlegendentry{short}" in tex and "Suzuki S4" not in tex
    assert "fit on $n = 5$--$10$, $q{=}1$" in tex
    assert tex.count("\\begin{loglogaxis}") == 1


def test_stays_out_of_the_other_derivers_scope():
    for tex in (_ratio_figure(), _panel_figure()):
        assert not figuniform.in_scope(tex)
        assert figdrop.has_fixed_y(tex)
        assert figdrop.is_derived(tex)


def test_real_record_reproduces_from_the_runs():
    # The committed record and runs, end to end: all checks in ratio_points must pass.
    here = Path(__file__).resolve().parents[2] / "results" / "np-childs-format"
    if not (here / "main-cnot-all-formulas.json").exists():
        pytest.skip("results not present")
    rec = json.loads((here / "main-cnot-all-formulas.json").read_text())
    spec = next(f["runs"] for f in json.loads((here / "config.json").read_text())["figures"]
                if f["id"] == "main-cnot-all-formulas")
    runs = pfchilds.load_run_set(here, spec)
    r = figratio.ratio_points(rec, runs, "bm-q10")
    assert sorted(r) == sorted(rec["shown"])
    # every paired value stays within 2% of the ratio of the plotted means
    for pts in r.values():
        for p in pts["measured"] + pts["extrapolated"]:
            assert p["value"] == pytest.approx(p["ratio_of_plotted"], rel=0.02)
