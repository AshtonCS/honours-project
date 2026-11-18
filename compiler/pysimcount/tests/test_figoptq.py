"""figoptq: the (4q+1)/4q layer-count model and its check against the RQ4 record."""

import copy
import json
from pathlib import Path

import pytest

from pysimcount import figoptq
from pysimcount.pfcompare import formula_registry

RECORD = Path(__file__).resolve().parents[2] / "results" / "np-mean" / "rq4-optimiser-table.json"


def test_model_by_hand():
    # S2: 6 layers -> 5 merged -> 4 optimised, factor 5/4.
    assert figoptq.optimiser_factor(1) == 5 / 4
    # S4, q = 5: 30 -> 21 -> 20, factor 21/20.
    assert figoptq.optimiser_factor(5) == 21 / 20
    with pytest.raises(ValueError):
        figoptq.optimiser_factor(0)


@pytest.mark.skipif(not RECORD.exists(), reason="RQ4 record not present")
def test_record_matches_model_at_every_n():
    rec = json.loads(RECORD.read_text())
    reg = formula_registry()
    for n in rec["sizes"]:
        assert figoptq.check(rec["rows"], reg, n)


@pytest.mark.skipif(not RECORD.exists(), reason="RQ4 record not present")
def test_check_raises_on_mismatch():
    rec = json.loads(RECORD.read_text())
    rows = copy.deepcopy(rec["rows"])
    row = next(r for r in rows if r["scheme"] == "suzuki4" and r["n"] == 11)
    row["opt_cnot"]["per_seed"]["1"] *= 1.001
    with pytest.raises(ArithmeticError):
        figoptq.check(rows, formula_registry(), 11)
