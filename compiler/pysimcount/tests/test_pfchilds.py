"""Hand-checkable cases for pfchilds: exact affine counts, the r rule, and pruning."""

import json
from fractions import Fraction

import pytest

from pysimcount import pfchilds


def test_affine_exact_recovers_line():
    # S8 merged step: 10,020 / 20,040 / 40,080 CNOTs measured at n = 10 / 20 / 40, i.e. 1002 n
    pts = [(10, 10020), (20, 20040), (40, 40080)]
    alpha, beta = pfchilds.affine_exact(pts)
    assert (alpha, beta) == (Fraction(1002), Fraction(0))
    assert pfchilds.affine_eval(alpha, beta, 500) == 501000


def test_affine_exact_with_offset():
    # optimised S4 total gates: 115 n - 40 at n = 10, 13, 16 (1110, 1455, 1800)
    alpha, beta = pfchilds.affine_exact([(10, 1110), (13, 1455), (16, 1800)])
    assert (alpha, beta) == (115, -40)
    assert pfchilds.affine_eval(alpha, beta, 100) == 11460


def test_affine_exact_rejects_curvature_and_two_points():
    with pytest.raises(ValueError):
        pfchilds.affine_exact([(10, 100), (20, 200), (40, 401)])
    with pytest.raises(ValueError):
        pfchilds.affine_exact([(10, 100), (20, 200)])


def test_affine_eval_rejects_non_integer():
    with pytest.raises(ValueError):
        pfchilds.affine_eval(Fraction(1, 2), Fraction(0), 3)


def test_r_fit_ceil_matches_simcount_emp():
    # Childs et al. 2018 Eq. 147, order 4: r = 4.0345 n^1.5545 -> 145 at n = 10 (bounds.py emp row)
    assert pfchilds.r_fit_ceil(4.034530593241404, 1.554526451252546, 10) == 145


def test_prune_removes_worst_by_geomean():
    curves = {k: {10: v, 100: v * 10} for k, v in [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]}
    kept, rec = pfchilds.prune(curves, remove=3, protect_top=2)
    assert kept == ["a", "b"]
    assert [r["series"] for r in rec if r["removed"]] == ["c", "d", "e"]


def test_prune_protects_a_formula_best_at_one_size():
    # e is worst on average but cheapest at n = 500, so it must stay;
    # the next-worst unprotected formulas go instead
    curves = {
        "a": {10: 1, 100: 10, 500: 100},
        "b": {10: 2, 100: 11, 500: 101},
        "c": {10: 3, 100: 12, 500: 102},
        "d": {10: 4, 100: 13, 500: 103},
        "e": {10: 1000, 100: 1000, 500: 50},
        "f": {10: 5, 100: 14, 500: 104},
    }
    kept, rec = pfchilds.prune(curves, remove=3, protect_top=2)
    assert "e" in kept and "a" in kept
    assert sorted(k for k in curves if k not in kept) == ["c", "d", "f"]
    note = {r["series"]: r["note"] for r in rec}
    assert note["e"].startswith("kept: rank 1")


def test_prune_never_below_protect_top():
    curves = {"a": {10: 1}, "b": {10: 2}, "c": {10: 3}}
    kept, _ = pfchilds.prune(curves, remove=3, protect_top=2)
    assert kept == ["a", "b"]


def test_prune_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        pfchilds.prune({"a": {10: 1}, "b": {20: 1}}, remove=1, protect_top=1)


# ------------------------------------------------------------------ load_run_set

def _write_run(path, seed: int, sizes: list[int], dirty: bool = False, commit: str = "aaaa111", template: str = "simcount",
               extra_rows: list[dict] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = [{"scheme": "bm-q10", "n": n, "gatebase": "cz", "merged": True, "r_source": "measured", "r": n} for n in sizes]
    (path / "results.json").write_text(json.dumps({
        "meta": {"config": {"seed": seed, "name": "x", "instances": "numpy", "template": template,
                            "synth_backend": "pygridsynth", "gatebases": ["cz"], "sizes": sizes},
                 "provenance": {"git_commit": commit, "git_dirty": dirty},
                 "fields": {str(n): [0.1 * seed] * n for n in sizes}},
        "rows": rows + (extra_rows or [])}))


def _run_set(tmp_path):
    for s in (1, 2):
        _write_run(tmp_path / "main" / f"seed{s}", s, [4, 5])
        _write_run(tmp_path / "n9" / f"seed{s}", s, [9], dirty=True, commit="e42ff29abc")
    (tmp_path / "main" / "aggregate.json").write_text(json.dumps({"runs": ["seed1/results.json", "seed2/results.json"]}))
    return {"aggregate": "main/aggregate.json", "extra": ["n9/seed{seed}"],
            "dirty_exception": {"commit": "e42ff29", "reason": "test"}}


def test_load_run_set_string_is_the_aggregate(tmp_path):
    _run_set(tmp_path)
    runs = pfchilds.load_run_set(tmp_path, "main/aggregate.json")
    assert sorted(runs) == [1, 2] and sorted(r["n"] for r in runs[1]["rows"]) == [4, 5]


def test_load_run_set_merges_extra_rows_and_fields(tmp_path):
    runs = pfchilds.load_run_set(tmp_path, _run_set(tmp_path))
    assert sorted(r["n"] for r in runs[2]["rows"]) == [4, 5, 9]
    assert runs[2]["meta"]["fields"]["9"] == [0.2] * 9
    assert runs[1]["extra_provenance"] == [{"path": "n9/seed1/results.json", "git_commit": "e42ff29abc", "git_dirty": True}]
    assert pfchilds._run_source(_run_set(tmp_path)) == "main/aggregate.json + n9/seed{seed}"


def test_load_run_set_refuses_unexcused_dirty_runs(tmp_path):
    spec = _run_set(tmp_path)
    with pytest.raises(ValueError, match="no exception for that commit"):
        pfchilds.load_run_set(tmp_path, {**spec, "dirty_exception": {"commit": "fffffff", "reason": "other"}})
    del spec["dirty_exception"]
    with pytest.raises(ValueError, match="no exception for that commit"):
        pfchilds.load_run_set(tmp_path, spec)


def test_load_run_set_refuses_other_settings_and_repeated_rows(tmp_path):
    spec = _run_set(tmp_path)
    _write_run(tmp_path / "n9" / "seed2", 2, [9], dirty=True, commit="e42ff29abc", template="other")
    with pytest.raises(ValueError, match="template"):
        pfchilds.load_run_set(tmp_path, spec)
    spec = _run_set(tmp_path)
    _write_run(tmp_path / "n9" / "seed1", 1, [5], dirty=True, commit="e42ff29abc")  # n = 5 is already in the main run
    with pytest.raises(ValueError, match="already in seed 1"):
        pfchilds.load_run_set(tmp_path, spec)
