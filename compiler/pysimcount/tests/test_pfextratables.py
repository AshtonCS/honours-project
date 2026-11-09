"""pfextratables: exponents and bound winners by hand; committed tables cover the agreed ranges."""

import json
from pathlib import Path

import pytest

from pysimcount import pfextratables

RESULTS = Path(__file__).resolve().parents[2] / "results" / "np-mean"


def test_exponents_by_hand():
    rec = {"series": {"suzuki4": {"fit": {"b": 1.6}, "per_seed_fit": {"1": [3.0, 1.5], "2": [3.0, 1.7]}},
                      "mo-q6": {"fit": {"b": 1.3}, "per_seed_fit": {"1": [9.0, 1.2], "2": [9.0, 1.4]}}}}
    s4, q6 = pfextratables.exponents(rec)
    assert (s4["scheme"], s4["b_min"], s4["b_max"], s4["theory"]) == ("suzuki4", 1.5, 1.7, 1.5)
    assert s4["eq147"] == pytest.approx(1.554526451252546)  # simcount's Eq. 147 constant
    assert q6["eq147"] is None and q6["theory"] == 1.5


def _row(scheme, n, src, cnot):
    return {"instance": "numpy", "gatebase": "cz", "merged": True, "scheme": scheme, "n": n, "r_source": src,
            "cnot_total": cnot}


def test_bound_winners_by_hand():
    # measured: S4 cheaper on both seeds; ana: S2 on seed 1, S4 on seed 2; tte has no S6 row (orders <= 4)
    runs = {s: {"meta": {"config": {"instances": "numpy"}}, "rows": [
        _row("suzuki2", 6, "measured", 100), _row("suzuki4", 6, "measured", 50), _row("suzuki6", 6, "measured", 70),
        _row("suzuki2", 6, "ana", 1000), _row("suzuki4", 6, "ana", 900 if s == 1 else 1100),
        _row("suzuki2", 6, "tte", 300), _row("suzuki4", 6, "tte", 200)]} for s in (1, 2)}
    rows = {r["source"]: r for r in pfextratables.bound_winners(runs, [6])}
    assert rows["measured"]["wins"] == {"suzuki4": 2}
    assert rows["ana"]["wins"] == {"suzuki4": 1, "suzuki2": 1}
    assert rows["tte"]["candidates"] == ["suzuki2", "suzuki4"]


def test_committed_tables():
    exp = json.loads((RESULTS / "rq3-exponents-table.json").read_text())
    assert len(exp["rows"]) == 11
    bw = json.loads((RESULTS / "rq2-bound-winners-table.json").read_text())
    assert bw["sizes"] == list(range(4, 12)) and bw["seeds"] == list(range(1, 11))
    par = json.loads((RESULTS / "validation-parity-table.json").read_text())
    status = [r["status"] for r in par["rows"]]
    assert status.count("matched") == 117 and "MISMATCH" not in status and len(status) == 119
