"""pfexacttables: the LaTeX tables carry the numbers in the committed np-exact results."""

import json

import pytest

from pysimcount import pfexacttables as T
from pysimcount.pfcompare import HERE

RES = HERE.parent / "results"
EX, LOC = RES / "np-exact" / "results.json", RES / "np-exact-local" / "results.json"
pytestmark = pytest.mark.skipif(not (EX.exists() and LOC.exists()), reason="np-exact results not present")


@pytest.fixture(scope="module")
def tex():
    return T.tables_tex(json.loads(EX.read_text()), json.loads(LOC.read_text()))


def test_sci():
    assert T.sci(3.71e-15) == "$3.7\\times10^{-15}$"
    assert T.sci(1e-15, 0) == "$10^{-15}$"
    assert T.sci(9.96e-3) == "$1.0\\times10^{-2}$"
    assert T.thousands(7000) == "7{,}000"


def test_all_tables_present_and_braces_balance(tex):
    for label in ("exact-summary", "exact-commuting", "exact-order", "exact-s8", "exact-controls"):
        assert f"\\label{{tab:{label}}}" in tex
    assert tex.count("{") == tex.count("}")
    assert tex.count("\\begin{table}") == tex.count("\\end{table}") == 5


def test_numbers_come_from_the_results(tex):
    loc = json.loads(LOC.read_text())
    s8 = {f["n"]: f for f in loc["fits"] if f["scheme"] == "suzuki8"}
    assert f"{s8[4]['fitted_order']:.2f}$^\\dagger$" in tex
    ex = json.loads(EX.read_text())
    passed = sum(x["pass"] for x in ex["commuting"])
    assert f"& {len(ex['commuting'])} & {passed} \\\\" in tex


def test_compact_table():
    tex = "\n".join(T.compact_table(json.loads(EX.read_text()), json.loads(LOC.read_text())))
    assert "\\label{tab:exact-compact}" in tex and tex.count("{") == tex.count("}")
    assert "S8 $n=4$" in tex and "M\\&O $q=14$ $n=6$" in tex
