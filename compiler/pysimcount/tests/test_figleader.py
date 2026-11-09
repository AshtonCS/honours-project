"""figleader's ranks and seed counts reproduce hand values; ties and bad inputs raise."""

import json
from pathlib import Path

import pytest

from pysimcount import figdrop, figleader, figratio, figuniform, pfchilds
from pysimcount.pfcompare import formula_registry

# Three formulas, two seeds, n = 4 measured, n = 100 extrapolated.
# Per step: suzuki4 42 n, bm-q10 82 n, mo-q6 50 n.
# Measured r at n = 4 (seed 1, seed 2): suzuki4 10, 40; bm-q10 10, 5; mo-q6 4, 20
#   totals: suzuki4 1680, 6720; bm-q10 3280, 1640; mo-q6 800, 4000
#   ranks:  seed 1 mo-q6 < suzuki4 < bm-q10; seed 2 bm-q10 < mo-q6 < suzuki4
# Per-seed fits at n = 100, r = ceil(a n^b):
#   suzuki4 (1,1), (2,1) -> r 100, 200 -> 420000, 840000
#   bm-q10  (1,1), (1,1) -> r 100, 100 -> 820000, 820000
#   mo-q6   (3,1), (0.5,1) -> r 300, 50 -> 1500000, 250000
#   ranks:  seed 1 suzuki4 < bm-q10 < mo-q6; seed 2 mo-q6 < bm-q10 < suzuki4
STEP = {"suzuki4": 42, "bm-q10": 82, "mo-q6": 50}
R4 = {"suzuki4": (10, 40), "bm-q10": (10, 5), "mo-q6": (4, 20)}
FITS = {"suzuki4": [[1, 1], [2, 1]], "bm-q10": [[1, 1], [1, 1]], "mo-q6": [[3, 1], [0.5, 1]]}
AT100 = {"suzuki4": (420000, 840000), "bm-q10": (820000, 820000), "mo-q6": (1500000, 250000)}


def _record() -> dict:
    series = {}
    for k in STEP:
        totals = [STEP[k] * 4 * r for r in R4[k]]
        series[k] = {"measured": [{"n": 4, "mean": sum(totals) / 2, "min": min(totals), "max": max(totals)}],
                     "per_seed_fit": {"1": FITS[k][0], "2": FITS[k][1]},
                     "per_step_model": {"alpha": str(STEP[k]), "beta": "0"},
                     "extrapolated": [{"n": 100, "value": sum(AT100[k]) / 2, "min": min(AT100[k]),
                                       "max": max(AT100[k])}]}
    return {"id": "X", "kind": "xyz", "metric": "cnot", "seeds": [1, 2], "fit_sizes": [5, 10],
            "extrapolate_sizes": [100], "shown": list(STEP), "removed": ["suzuki8"], "series": series}


def _runs() -> dict[int, dict]:
    def row(k: str, seed: int) -> dict:
        r = R4[k][seed - 1]
        return {"r_source": "measured", "gatebase": "cz", "merged": True, "scheme": k, "n": 4, "r": r,
                "cnot_per_step": STEP[k] * 4, "cnot_total": r * STEP[k] * 4}
    return {s: {"rows": [row(k, s) for k in STEP]} for s in (1, 2)}


def _measured(rec: dict | None = None) -> dict:
    return figratio.measured_per_seed(rec or _record(), _runs())


def test_seed_ranks_by_hand_and_ties_share_the_mean_rank():
    values = figleader.per_seed_at(_record(), _measured(), 4)
    assert values["suzuki4"] == {1: 1680, 2: 6720}
    ranks = figleader.seed_ranks(values)
    assert ranks == {"suzuki4": {1: 2, 2: 3}, "bm-q10": {1: 3, 2: 1}, "mo-q6": {1: 1, 2: 2}}
    tied = figleader.seed_ranks({"a": {1: 5}, "b": {1: 5}, "c": {1: 9}})
    assert tied == {"a": {1: 1.5}, "b": {1: 1.5}, "c": {1: 3}}


def test_leaderboard_at_an_extrapolated_size_by_hand():
    rows = figleader.leaderboard(_record(), _measured(), 100)
    # sorted by the record's plotted value: 630000, 820000, 875000
    assert [r["scheme"] for r in rows] == ["suzuki4", "bm-q10", "mo-q6"]
    by = {r["scheme"]: r for r in rows}
    assert (by["suzuki4"]["wins"], by["bm-q10"]["wins"], by["mo-q6"]["wins"]) == (1, 0, 1)
    assert (by["suzuki4"]["rank_min"], by["suzuki4"]["rank_max"]) == (1, 3)
    assert (by["bm-q10"]["rank_min"], by["bm-q10"]["rank_max"], by["bm-q10"]["mean_rank"]) == (2, 2, 2)
    assert by["mo-q6"]["seed_mean"] == 875000 and not by["mo-q6"]["measured"]


def test_leaderboard_at_a_measured_size_uses_the_runs():
    by = {r["scheme"]: r for r in figleader.leaderboard(_record(), _measured(), 4)}
    assert by["mo-q6"]["measured"] and by["mo-q6"]["value"] == 2400
    assert (by["mo-q6"]["wins"], by["bm-q10"]["wins"], by["suzuki4"]["wins"]) == (1, 1, 0)


def test_tie_for_cheapest_raises():
    rec = _record()
    rec["series"]["mo-q6"]["per_seed_fit"]["1"] = [84, 0]  # 50 * 100 * 84 = 420000, suzuki4's seed 1
    with pytest.raises(ValueError, match="tie for cheapest"):
        figleader.leaderboard(rec, _measured(rec), 100)


def test_record_mismatch_and_unknown_size_raise():
    rec = _record()
    rec["series"]["bm-q10"]["extrapolated"][0]["min"] = 1
    with pytest.raises(ValueError, match="do not reproduce"):
        figleader.leaderboard(rec, _measured(rec), 100)
    with pytest.raises(ValueError, match="neither measured nor extrapolated"):
        figleader.per_seed_at(_record(), _measured(), 50)


def test_mean_rank_series_by_hand():
    s = figleader.mean_rank_series(_record(), _measured())
    assert s["suzuki4"] == [{"n": 4, "mean_rank": 2.5, "measured": True},
                            {"n": 100, "mean_rank": 2.0, "measured": False}]
    assert s["bm-q10"][0]["mean_rank"] == 2.0


def test_spread_labels():
    assert figleader.spread_labels({"a": 1, "b": 1, "c": 3}, 0.5, 0.5, 8) == {"a": 1, "b": 1.5, "c": 3}
    assert figleader.spread_labels({"a": 7.9, "b": 8}, 0.5, 0.5, 8) == pytest.approx({"a": 7.5, "b": 8})
    with pytest.raises(ValueError, match="do not fit"):
        figleader.spread_labels({"a": 1, "b": 1, "c": 1}, 0.5, 1, 1.5)


DOTS = {"width_cm": 6, "height_cm": 4, "title": "t", "xlabel": "x", "xmin": 1e5, "xmax": 2e6, "x_unit": 1e5,
        "xtick": [2e5, 1e6], "mark_size": "2pt", "columns_cm": [0.5, 1.4], "labels": {"mo-q6": "short"},
        "note_width_cm": 8, "note": "fit {fit}, n {n}, {seeds} seeds"}


def test_dots_figure_rows_columns_and_range():
    rows = figleader.leaderboard(_record(), _measured(), 100)
    tex = figleader.dots_tex(_record(), rows, 100, DOTS, ["% derived: X.json"], formula_registry())
    assert "yticklabels={{Suzuki S4},{Blanes \\& Moan q=10},{short}}" in tex
    assert "{1/2}" in tex and "{0/2}" in tex and "{1--3}" in tex
    assert "(420000,1) +=" not in tex and "(630000,1) += (210000,0) -= (210000,0)" in tex
    assert "fit $n = 5$--$10$, n 100, 2 seeds" in tex
    with pytest.raises(ValueError, match="outside x-range"):
        figleader.dots_tex(_record(), rows, 100, {**DOTS, "xmin": 3e5}, [], formula_registry())


SOURCE = r"""% Generated by pysimcount.pfchilds -- do not edit; regenerate instead.
% figure: X -- test
\begin{tikzpicture}
  \begin{loglogaxis}[
    xtick={10,100}, xticklabels={10,100},
    xmin=1, xmax=1000,
  ]
    % suzuki4: measured n, mean over seeds, bars min..max over seeds
    \addplot[red, solid, mark=square*, mark size=2.2pt, mark options={solid, fill=red}, error bars/.cd, y dir=both, y explicit, error bar style={solid}] coordinates {
      (4,4200) += (0,2520) -= (0,2520)
    };
    \addlegendentry{Suzuki S4}
    % suzuki4: extrapolated, r = ceil(1 n^1), per step = 42 n + 0; bars min..max of per-seed fits
    \addplot[red, dashed, mark=square*, mark size=2.2pt, mark options={solid, fill=white}, forget plot, error bars/.cd, y dir=both, y explicit, error bar style={solid}] coordinates {
      (4,4200)
      (100,630000) += (0,210000) -= (0,210000)
    };
    \addlegendimage{black, solid, mark=*, mark options={solid, fill=black}}
    \addlegendentry{measured $r$ (mean over seeds)}
    \addlegendimage{black, dashed, mark=*, mark options={solid, fill=white}}
    \addlegendentry{$r$ from power-law fit to mean measured $r$ (n = 5--10)}
  \end{loglogaxis}
\end{tikzpicture}
"""

SIDE = {"main": {"width_cm": 6, "height_cm": 5, "xmin": 3, "xmax": 200, "ymin": 1e3, "ymax": 1e7, "ylabel": "y",
                 "mark_size": "1pt", "legend_columns": 2, "legend_y": -0.2, "band_factor": 1.1},
        "dots": {**DOTS, "gap_cm": 3, "labels": {"suzuki4": "S4 short"}}}


def _side() -> str:
    rows = figleader.leaderboard(_record(), _measured(), 100)
    return figleader.side_by_side_tex(SOURCE, _record(), rows, 100, SIDE, ["% derived: X.tex"], formula_registry())


def test_side_by_side_copies_coordinates_without_bars_and_uses_short_labels():
    left = _side().split("\\end{loglogaxis}")[0]
    assert "      (4,4200)\n    };" in left and "      (100,630000)\n    };" in left and "+=" not in left
    assert "\\addlegendentry{S4 short}" in left and "Suzuki S4" not in left
    assert "\\addlegendentry{extrapolated (fit on $n = 5$--$10$)}" in left
    assert "at={(9cm,0cm)}" in _side()  # dot plot starts gap_cm right of the main axis


def test_side_by_side_rejects_a_plot_it_cannot_name():
    rows = figleader.leaderboard(_record(), _measured(), 100)
    with pytest.raises(ValueError, match="names no shown formula"):
        figleader.side_by_side_tex(SOURCE.replace("% suzuki4:", "% suzuki8:"), _record(), rows, 100, SIDE, [],
                                   formula_registry())


def test_stays_out_of_the_other_derivers_scope():
    rec, m = _record(), _measured()
    rows = figleader.leaderboard(rec, m, 100)
    cfg = {"width_cm": 6, "height_cm": 4, "xmin": 3, "xmax": 200, "ylabel": "rank", "mark_size": "1pt",
           "note_width_cm": 8, "note": "n"}
    texts = [figleader.dots_tex(rec, rows, 100, DOTS, ["% derived: X.json"], formula_registry()), _side(),
             figleader.mean_rank_tex(rec, figleader.mean_rank_series(rec, m), cfg, ["% derived: X.json"],
                                     formula_registry())]
    for tex in texts:
        assert not figuniform.in_scope(tex)
        assert figdrop.has_fixed_y(tex)
        assert figdrop.is_derived(tex)


def test_real_record_n500():
    here = Path(__file__).resolve().parents[2] / "results" / "np-childs-format"
    if not (here / "main-cnot-all-formulas.json").exists():
        pytest.skip("results not present")
    rec = json.loads((here / "main-cnot-all-formulas.json").read_text())
    spec = next(f["runs"] for f in json.loads((here / "config.json").read_text())["figures"]
                if f["id"] == "main-cnot-all-formulas")
    runs = pfchilds.load_run_set(here, spec)
    rows = figleader.leaderboard(rec, figratio.measured_per_seed(rec, runs), 500)
    assert sum(r["wins"] for r in rows) == len(rec["seeds"])
    # fit n = 5-11 (2026-11-08): three tiers that no seed crosses -- top three ranks 1-3, middle three 4-6
    by = {r["scheme"]: r for r in rows}
    assert {k for k, r in by.items() if r["rank_max"] <= 3} == {"bm-q10", "morales-m8", "mo-q14"}
    assert all(by[k]["rank_min"] >= 4 and by[k]["rank_max"] <= 6 for k in ("morales-y8m10", "mo-q6", "mo-o8-q21"))
    assert (by["bm-q10"]["wins"], by["morales-m8"]["wins"], by["mo-q14"]["wins"]) == (4, 5, 1)
    assert [r["scheme"] for r in rows[-2:]] == ["suzuki6", "suzuki4"]


ZOOM = {"main": {"width_cm": 6, "height_cm": 5, "xmin": 3, "xmax": 200, "ymin": 1e3, "ymax": 1e7, "ylabel": "y",
                 "mark_size": "1pt", "legend_columns": 2, "legend_y": -0.2},
        "panel": {"width_cm": 4, "height_cm": 5, "gap_cm": 1.5, "title": "z", "xmin": 50, "xmax": 150,
                  "ymin": 1e5, "ymax": 1e6, "xtick": [100], "ytick": [3e5], "label_gap_cm": 0.3,
                  "labels": {"suzuki4": "S4 short"}}}


def test_zoom_copies_coordinates_and_bars_and_labels_each_line():
    tex = figleader.side_by_side_zoom_tex(SOURCE, _record(), ZOOM, ["% derived: X.tex"], formula_registry())
    left, right = tex.split("\\end{loglogaxis}")[:2]
    assert "+=" not in left and "(axis cs:50,100000) rectangle (axis cs:150,1e+06)" in left
    # the right axis holds the extrapolated plot only, bars and all, exactly as in the source
    assert "      (100,630000) += (0,210000) -= (0,210000)" in right and "(4,4200) +=" not in right
    assert "clip mode=individual" in right and right.count("{S4 short}") == 1
    assert "$3{\\times}10^{5}$" in right


def test_zoom_refuses_a_point_it_would_clip():
    bad = {**ZOOM, "panel": {**ZOOM["panel"], "ymax": 5e5}}
    with pytest.raises(ValueError, match="outside its y-range"):
        figleader.side_by_side_zoom_tex(SOURCE, _record(), bad, [], formula_registry())
