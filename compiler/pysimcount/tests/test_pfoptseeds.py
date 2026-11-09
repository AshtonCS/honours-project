"""Paired unoptimised vs Nam-optimised aggregation over seeds (pfoptseeds.py), on small hand-made runs."""

import pytest

from pysimcount import pfoptseeds as P


def _rows(scheme, n, r, per_step):
    """per_step: {metric: (unopt, marginal)} -> pfoptimised-style rows."""
    return [{"scheme": scheme, "n": n, "r": r, "metric": m,
             "unopt_per_step": u, "marginal_per_step": o, "unopt_total": u * r, "marginal_total": o * r}
            for m, (u, o) in per_step.items()]


def _run(seed, rows, dirty=False):
    cfg = {"seed": seed, "name": f"s{seed}", "schemes": ["suzuki2"], "gatebase": "cz"}
    return {"meta": {"config": cfg, "provenance": {"git_commit": "abc", "git_dirty": dirty}}, "rows": rows}


# The n=4 S2 merged step: CNOT 40 -> 32, Rz 24 -> 20, total 112 -> 49 (hand-made total), r per seed below
STEP = {"CNOT": (40, 32), "Rz": (24, 20), "total": (112, 49)}
R = {1: 632, 2: 700, 3: 650}


def _runs():
    return {s: _run(s, _rows("suzuki2", 4, r, STEP)) for s, r in R.items()}


def test_factor_is_per_step_ratio_by_hand():
    """Same step on every seed, so the factor is 40/32 = 1.25 on each, whatever r is.
    Median r = 650 (seed 3); median unoptimised CNOT = 40 x 650 = 26,000, optimised 32 x 650 = 20,800."""
    x = {(g["metric"]): g for g in P.paired_factors(_runs())}["CNOT"]
    for s in R:
        assert x["factor"][s] == pytest.approx(1.25)
    assert (x["factor_min"], x["factor_max"]) == (pytest.approx(1.25), pytest.approx(1.25))
    assert x["r_median"] == 650
    assert (x["unopt_total_median"], x["optimised_total_median"]) == (26_000, 20_800)


def test_factor_varies_when_a_seed_optimises_differently():
    """Seed 2 finds one more CNOT cancellation (marginal 31): factor 40/31 = 1.290 there; median stays 1.25."""
    runs = _runs()
    for row in runs[2]["rows"]:
        if row["metric"] == "CNOT":
            row["marginal_per_step"] = 31
            row["marginal_total"] = 31 * row["r"]
    x = {g["metric"]: g for g in P.paired_factors(runs)}["CNOT"]
    assert x["factor"][2] == pytest.approx(40 / 31)
    assert x["factor_median"] == pytest.approx(1.25)
    assert x["factor_max"] == pytest.approx(40 / 31)


def test_missing_row_raises():
    runs = _runs()
    runs[3]["rows"] = [row for row in runs[3]["rows"] if row["metric"] != "Rz"]
    with pytest.raises(ValueError, match="different set"):
        P.paired_factors(runs)


def test_zero_optimised_total_raises():
    runs = _runs()
    for row in runs[1]["rows"]:
        if row["metric"] == "Rz":
            row["marginal_total"] = 0
    with pytest.raises(ValueError, match="optimised total"):
        P.paired_factors(runs)


def test_dirty_run_is_rejected(tmp_path):
    import json
    paths = []
    for s, r in R.items():
        p = tmp_path / f"seed{s}.json"
        p.write_text(json.dumps(_run(s, _rows("suzuki2", 4, r, STEP), dirty=(s == 2))))
        paths.append(p)
    with pytest.raises(ValueError, match="git_dirty"):
        P.load_runs(paths)


def test_per_seed_source_is_allowed_only_when_declared(tmp_path):
    import json
    paths = []
    for s, r in R.items():
        run = _run(s, _rows("suzuki2", 4, r, STEP))
        run["meta"]["config"]["source"] = f"../../np-main/seed{s}/results.json"
        p = tmp_path / f"seed{s}.json"
        p.write_text(json.dumps(run))
        paths.append(p)
    with pytest.raises(ValueError, match="config differs"):
        P.load_runs(paths)
    assert sorted(P.load_runs(paths, per_seed_keys=("source",))) == [1, 2, 3]
