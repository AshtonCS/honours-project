"""Paired two-group vs X-Y-Z aggregation over seeds (pfgroupseeds.py), on small hand-made runs."""

import json

import pytest

from pysimcount import pfgroupseeds as G


def _row(seed, scheme, n, variant, r, cnot_per_step, merged=True):
    return {"seed": seed, "scheme": scheme, "n": n, "variant": variant, "merged": merged, "r": r,
            "cnot_per_step": cnot_per_step, "cnot_total": cnot_per_step * r,
            "gates_per_step": 7 * cnot_per_step, "gates_total": 7 * cnot_per_step * r}


def _run(seed, rows, dirty=False):
    cfg = {"seed": seed, "name": f"s{seed}", "schemes": ["suzuki4"],
           "variants": ["xyz", "twogroup-even-out", "twogroup-odd-out"]}
    return {"meta": {"config": cfg, "provenance": {"git_commit": "abc", "git_dirty": dirty}}, "rows": rows}


# (seed) -> r for (variant); CNOT per step: xyz 168, two-group 66 (the n=4 S4 merged counts)
R = {1: {"xyz": 44, "twogroup-even-out": 39, "twogroup-odd-out": 40},
     2: {"xyz": 50, "twogroup-even-out": 100, "twogroup-odd-out": 60},
     3: {"xyz": 40, "twogroup-even-out": 40, "twogroup-odd-out": 44}}
CNOT = {"xyz": 168, "twogroup-even-out": 66, "twogroup-odd-out": 66}


def _runs():
    runs = {}
    for s, rs in R.items():
        rows = [_row(s, "suzuki4", 4, v, r, CNOT[v]) for v, r in rs.items()]
        rows += [_row(s, "suzuki4", 4, v, 1, 1, merged=False) for v in rs]  # unmerged rows must be ignored
        runs[s] = _run(s, rows)
    return runs


def test_gain_factorises_by_hand():
    """Seed 1, even-out: step ratio 168/66 = 2.545, r ratio 44/39 = 1.128, gain = 7392/2574 = 2.872.
    Seed 2: r ratio 0.5, gain 1.273. Seed 3: r ratio 1, gain 2.545. Median gain = 2.545 (seed 3)."""
    g = {(x["variant"], x["metric"]): x for x in G.paired_gains(_runs())}
    x = g[("twogroup-even-out", "cnot")]
    assert x["gain"][1] == pytest.approx((168 * 44) / (66 * 39))
    for s in R:
        assert x["gain"][s] == pytest.approx(x["step_ratio"][s] * x["r_ratio"][s])
    assert x["r_ratio"][2] == pytest.approx(0.5)
    assert x["gain_median"] == pytest.approx(168 / 66)
    assert (x["gain_min"], x["gain_max"]) == (pytest.approx(168 * 50 / (66 * 100)), pytest.approx(168 * 44 / (66 * 39)))
    assert x["seeds_two_group_cheaper"] == 3


def test_seed_where_two_group_loses_is_not_counted():
    """Odd-out seed 2: r ratio 50/60, gain 2.545 * 0.833 = 2.12 (> 1). Make it lose by needing 3x the steps."""
    runs = _runs()
    for row in runs[2]["rows"]:
        if row["variant"] == "twogroup-odd-out" and row["merged"]:
            row["r"] = 150
            row["cnot_total"] = 66 * 150
            row["gates_total"] = 7 * 66 * 150
    x = {(g["variant"], g["metric"]): g for g in G.paired_gains(runs)}[("twogroup-odd-out", "total")]
    assert x["gain"][2] == pytest.approx(168 * 50 / (66 * 150))
    assert x["gain"][2] < 1
    assert x["seeds_two_group_cheaper"] == 2


def test_missing_row_raises():
    runs = _runs()
    runs[3]["rows"] = [r for r in runs[3]["rows"] if r["variant"] != "twogroup-odd-out"]
    with pytest.raises(ValueError, match="different set"):
        G.paired_gains(runs)


def test_main_writes_outputs(tmp_path):
    for s, run in _runs().items():
        (tmp_path / f"seed{s}.json").write_text(json.dumps(run))
    cfg = {"runs": [f"seed{s}.json" for s in R], "out": "agg"}
    (tmp_path / "aggregate.json").write_text(json.dumps(cfg))
    G.main([str(tmp_path / "aggregate.json")])
    md = (tmp_path / "agg" / "gains.md").read_text()
    assert "| twogroup-even-out | suzuki4 | 4 | 2.55× |" in md
    assert len(json.loads((tmp_path / "agg" / "paired.json").read_text())["gains"]) == 4
