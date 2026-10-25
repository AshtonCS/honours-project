"""pffigure: figures contain exactly the rows of the run, and the figure set
follows from the rows (so new formulas appear without code changes)."""

import pytest

from pysimcount import pffigure
from pysimcount.pfcompare import formula_registry
from pysimcount.weighted import FormulaSpec

REG = formula_registry()


def _row(scheme, n, source, gatebase="cz", merged=True, cnot=0, instance="simcount"):
    return {"instance": instance, "scheme": scheme, "n": n, "r_source": source, "gatebase": gatebase,
            "merged": merged, "cnot_total": cnot, "t_total": 0, "gates_total": 0}


ROWS = [
    _row("suzuki4", 4, "measured", cnot=100),
    _row("suzuki4", 10, "measured", cnot=900),
    _row("suzuki4", 20, "fit", cnot=5000),
    _row("suzuki4", 4, "com", cnot=7000),
    _row("suzuki2", 4, "measured", cnot=300),
    _row("suzuki2", 4, "com", cnot=9000),
    _row("mo-q6", 4, "measured", cnot=80),
    _row("suzuki4", 4, "measured", merged=False, cnot=999),    # other merge setting
    _row("suzuki4", 4, "measured", gatebase="ct", cnot=777),   # other gatebase
    _row("suzuki4", 4, "measured", instance="paper", cnot=1),  # other problem
]


def test_collect_filters_and_splits_fit_from_computed():
    d = pffigure.collect(ROWS, "cnot_total", "cz", merged=True)
    assert d[("suzuki4", "measured")] == {"measured": [(4, 100.0), (10, 900.0)], "fit": [(20, 5000.0)]}
    assert d[("suzuki4", "com")] == {"measured": [(4, 7000.0)], "fit": []}
    assert set(d) == {("suzuki4", "measured"), ("suzuki4", "com"), ("suzuki2", "measured"),
                      ("suzuki2", "com"), ("mo-q6", "measured")}


def test_duplicate_rows_raise():
    with pytest.raises(ValueError, match="two rows"):
        pffigure.collect(ROWS + [_row("mo-q6", 4, "measured")], "cnot_total", "cz", merged=True)


def test_standard_figures_follow_from_the_rows():
    d = pffigure.collect(ROWS, "cnot_total", "cz", merged=True)
    figs = {f.name: f.members for f in pffigure.standard_figures(set(d), REG)}
    assert figs == {
        "A-com-by-order": [("suzuki2", "com"), ("suzuki4", "com")],
        "A-measured-by-order": [("suzuki2", "measured"), ("suzuki4", "measured")],
        "B-order4-by-formula": [("suzuki4", "measured"), ("mo-q6", "measured")],
        "B-all-by-formula": [("suzuki2", "measured"), ("suzuki4", "measured"), ("mo-q6", "measured")],
        "C-suzuki2-by-bound": [("suzuki2", "com"), ("suzuki2", "measured")],
        "C-suzuki4-by-bound": [("suzuki4", "com"), ("suzuki4", "measured")],
    }


def test_new_formula_is_styled_from_family_and_order():
    reg = dict(REG)
    reg["new-o4"] = FormulaSpec("New & improved", "paper", 4, 7, lambda H, d: [])
    rows = ROWS + [_row("new-o4", 4, "measured", cnot=50)]
    d = pffigure.collect(rows, "cnot_total", "cz", merged=True)
    spec = next(f for f in pffigure.standard_figures(set(d), reg) if f.name == "B-order4-by-formula")
    assert [m[0] for m in spec.members] == ["suzuki4", "mo-q6", "new-o4"]
    # same family and order as mo-q6 -> identical look -> must raise, naming the fix
    with pytest.raises(ValueError, match="COLOUR_OVERRIDE"):
        pffigure.series_for(spec, d, reg)


def test_fit_points_are_open_and_others_filled():
    d = pffigure.collect(ROWS, "cnot_total", "cz", merged=True)
    spec = next(f for f in pffigure.standard_figures(set(d), REG) if f.name == "C-suzuki4-by-bound")
    tex = pffigure.figure_tex(pffigure.series_for(spec, d, REG), "CNOT count", ["test"])
    computed, fit = tex.split("% r from power-law fit")
    assert "(10,900)" in computed and "(4,7000)" in computed
    assert "(20,5000)" in fit and "fill=white}, forget plot" in fit
    assert tex.count("\\addlegendentry") == 2 + 2  # two methods + computed/fitted key


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="formula_registry"):
        pffigure.standard_figures({("mystery", "measured")}, REG)
