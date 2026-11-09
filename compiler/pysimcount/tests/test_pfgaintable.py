"""pfgaintable: paired gains by hand, the step x r decomposition, and the committed table."""

import json
from pathlib import Path

import pytest

from pysimcount import pfgaintable


def _row(scheme, family, order, n, gb, r, rz_step, cnot_step, t_total=0):
    return {"instance": "numpy", "gatebase": gb, "merged": True, "r_source": "measured", "scheme": scheme,
            "family": family, "order": order, "n": n, "r": r, "rz_per_step": rz_step,
            "rz_total": rz_step * r, "cnot_total": cnot_step * r, "t_total": t_total}


def _runs(r_alt_by_seed):
    runs = {}
    for s, r_alt in r_alt_by_seed.items():
        rows = []
        for gb in ("cz", "ct"):
            rows.append(_row("suzuki4", "suzuki", 4, 6, gb, 60, 100, 160, t_total=8000 if gb == "ct" else 0))
            rows.append(_row("mo-q6", "mo", 4, 6, gb, r_alt, 125, 200, t_total=4000 if gb == "ct" else 0))
        runs[s] = {"meta": {"config": {"instances": "numpy", "seed": s}}, "rows": rows}
    return runs


def test_gains_by_hand():
    # per-step ratio 100/125 = 0.8; r ratio 60/30 = 2 and 60/40 = 1.5 -> exponential gains 1.6 and 1.2
    recs = pfgaintable.gains(_runs({1: 30, 2: 40}), [6])
    [g] = recs
    assert (g["base"], g["alt"], g["n"]) == ("suzuki4", "mo-q6", 6)
    assert g["step"]["geomean"] == pytest.approx(0.8)
    assert g["exp"]["per_seed"] == pytest.approx({1: 1.6, 2: 1.2})
    assert g["exp"]["geomean"] == pytest.approx((1.6 * 1.2) ** 0.5)
    assert g["cnot"]["geomean"] == pytest.approx(g["exp"]["geomean"])  # same CNOT/Rz per step ratio here
    assert g["t"]["per_seed"] == pytest.approx({1: 2.0, 2: 2.0})


def test_missing_row_raises():
    runs = _runs({1: 30})
    runs[1]["rows"] = [r for r in runs[1]["rows"] if not (r["scheme"] == "mo-q6" and r["gatebase"] == "ct")]
    with pytest.raises(ValueError, match="no measured t row"):
        pfgaintable.gains(runs, [6])


def test_committed_table_covers_n4_to_11_and_decomposes():
    p = Path(__file__).resolve().parents[2] / "results" / "np-mean" / "rq1-gains-table.json"
    d = json.loads(p.read_text())
    assert d["sizes"] == list(range(4, 12)) and d["table_n"] == 11 and d["seeds"] == list(range(1, 11))
    assert len(d["rows"]) == 6 * 8
    for g in d["rows"]:
        assert g["exp"]["geomean"] == pytest.approx(g["step"]["geomean"] * g["r"]["geomean"], rel=1e-12)
