"""pfopttable: merging and optimiser factors by hand, ranks, and the consistency checks."""

import pytest

from pysimcount import pfopttable


def _main(seed, scheme, n, merged, r, rz_step, cnot_step, gates_step):
    return {"instance": "numpy", "gatebase": "cz", "r_source": "measured", "scheme": scheme, "n": n, "merged": merged,
            "r": r, "rz_per_step": rz_step, "cnot_per_step": cnot_step, "gates_per_step": gates_step,
            "rz_total": rz_step * r, "cnot_total": cnot_step * r, "gates_total": gates_step * r}


def _opt(scheme, n, r, metric, unopt, marginal):
    return {"scheme": scheme, "n": n, "r": r, "metric": metric, "unopt_per_step": unopt,
            "unopt_total": unopt * r, "marginal_total": marginal * r, "_fields": None}


def _setup():
    # "a": unmerged 12 Rz / 24 CNOT per step, merged 10 / 20, optimised CNOT 16, Rz 9, total 50 (from 100); r = 10
    # "b": no merging (6 / 12), optimised CNOT 12 (no saving), Rz 6, total 40 (from 60); r = 30
    runs = {1: {"meta": {"config": {"instances": "numpy", "seed": 1}}, "rows": [
        _main(1, "a", 6, False, 10, 12, 24, 120), _main(1, "a", 6, True, 10, 10, 20, 100),
        _main(1, "b", 6, False, 30, 6, 12, 60), _main(1, "b", 6, True, 30, 6, 12, 60)]}}
    opt = {1: [_opt("a", 6, 10, "CNOT", 20, 16), _opt("a", 6, 10, "Rz", 10, 9), _opt("a", 6, 10, "total", 100, 50),
               _opt("b", 6, 30, "CNOT", 12, 12), _opt("b", 6, 30, "Rz", 6, 6), _opt("b", 6, 30, "total", 60, 40)]}
    return runs, opt


def test_factors_and_ranks_by_hand():
    runs, opt = _setup()
    a, b = pfopttable.factors(runs, opt, [6])
    assert a["merge_exp"]["geomean"] == pytest.approx(1.2) and a["merge_cnot"]["geomean"] == pytest.approx(1.2)
    assert a["opt_cnot"]["geomean"] == pytest.approx(1.25) and a["opt_total"]["geomean"] == pytest.approx(2.0)
    assert a["combined_cnot"]["geomean"] == pytest.approx(1.5)
    # CNOT totals: a 200 -> 160, b 360 -> 360: a is cheaper before and after
    assert (a["rank_before"], a["rank_after"], b["rank_before"], b["rank_after"]) == (1, 1, 2, 2)


def test_optimiser_input_must_equal_merged_step():
    runs, opt = _setup()
    opt[1][0]["unopt_per_step"] = 21
    with pytest.raises(ValueError, match="optimiser input"):
        pfopttable.factors(runs, opt, [6])


def test_optimiser_r_must_equal_measured_r():
    runs, opt = _setup()
    opt[1][3]["r"] = 31
    with pytest.raises(ValueError, match="optimiser r"):
        pfopttable.factors(runs, opt, [6])
