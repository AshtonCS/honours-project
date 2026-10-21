"""eps sweep: config handling in pfcompare and the slope analysis in pfeps."""

import json
import math
import warnings

import pytest

warnings.filterwarnings("ignore")

from plsimcount import pfcompare as P, pfeps  # noqa: E402

THESIS = P.HERE.parent / "results" / "pf-thesis" / "results.json"


def _run(tmp_path, cfg):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    P.main([str(tmp_path / "config.json")])
    return json.loads((tmp_path / "results.json").read_text())["rows"]


def _key(r):
    return (r["instance"], r["n"], r["scheme"], r["gatebase"], r["r_source"], r["merged"])


def test_config_without_eps_reproduces_committed_thesis_rows(tmp_path):
    """Existing behaviour unchanged: a config without eps_trotter gives, field
    for field (except wall-clock time), the committed pf-thesis rows."""
    rows = _run(tmp_path, {"name": "t", "schemes": ["suzuki2", "mo-q6"], "sizes": [4], "seed": 1,
                           "gatebases": ["cz", "ct"]})
    ref = {_key(r): r for r in json.loads(THESIS.read_text())["rows"]}
    assert len(rows) == 8
    for r in rows:
        want = ref[_key(r)]
        for f, v in r.items():
            if f != "search_seconds":
                assert v == want[f], (f, v, want[f])


def test_eps_override_at_default_equals_default(tmp_path):
    """eps_trotter = [1e-3] on cz is the cz default: identical rows."""
    a = _run(tmp_path, {"name": "t", "schemes": ["suzuki4"], "sizes": [4], "seed": 1, "gatebases": ["cz"]})
    b = _run(tmp_path, {"name": "t", "schemes": ["suzuki4"], "sizes": [4], "seed": 1, "gatebases": ["cz"],
                        "eps_trotter": [0.001]})
    strip = lambda rows: [{k: v for k, v in r.items() if k != "search_seconds"} for r in rows]  # noqa: E731
    assert strip(a) == strip(b)


def test_eps_sweep_rows_and_monotone_r(tmp_path):
    rows = _run(tmp_path, {"name": "t", "schemes": ["suzuki2"], "sizes": [4], "seed": 1, "gatebases": ["cz"],
                           "eps_trotter": [0.01, 0.001, 0.0001]})
    rs = {r["eps_trotter"]: r["r"] for r in rows if r["merged"]}
    assert sorted(rs) == [0.0001, 0.001, 0.01]
    assert rs[0.01] < rs[0.001] < rs[0.0001]
    assert rs[0.001] == 699  # pf-thesis S2, n = 4, cz
    for r in rows:
        assert r["err_at_r"] <= r["eps_trotter"] < r["err_at_r_minus_1"]


@pytest.mark.parametrize("extra,match", [
    ({"gatebases": ["ct"]}, "without synthesis"),
    ({"extrapolate": {"sizes": [13], "fit_sizes": [4, 5]}}, "cannot be combined"),
    ({"eps_trotter": [0.01, 0.01]}, "distinct"),
    ({"eps_trotter": [2.0]}, "in \\(0, 1\\)"),
])
def test_eps_sweep_rejects_unsupported_configs(tmp_path, extra, match):
    cfg = {"name": "t", "schemes": ["suzuki2"], "sizes": [4], "seed": 1, "gatebases": ["cz"],
           "eps_trotter": [0.01]}
    cfg.update(extra)
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, cfg)


def _rows(law, eps_list, scheme="suzuki4", order=4, n=6):
    return [{"instance": "simcount", "r_source": "measured", "gatebase": "cz", "scheme": scheme, "family": "suzuki",
             "order": order, "n": n, "eps_trotter": e, "r": law(e), "merged": m}
            for e in eps_list for m in (False, True)]


def test_slope_of_exact_power_law():
    """r = eps^(-1/4) at eps = 1e-4, 1e-8, 1e-12 gives the integers 10, 100,
    1000 (by hand): slope -0.25 exactly, ratio 1."""
    table = {1e-4: 10, 1e-8: 100, 1e-12: 1000}
    [s] = pfeps.slopes(_rows(lambda e: table[e], sorted(table)))
    assert s.r == [1000, 100, 10]
    assert s.fit_slope == pytest.approx(-0.25)
    assert s.pair_slopes == [pytest.approx(-0.25), pytest.approx(-0.25)]
    assert s.ratio == pytest.approx(1.0)
    assert s.small_r is False


def test_slope_by_hand_and_small_r_flag():
    """r = 2, 3, 5 at eps = 1e-2, 1e-3, 1e-4: pairwise slopes -log10(3/2) = -0.176
    and -log10(5/3) = -0.222; LS slope over equally spaced log eps = (log 5 - log 2)
    / (2 log 10) * -1 = -0.199. r <= 3 -> flagged."""
    table = {1e-2: 2, 1e-3: 3, 1e-4: 5}
    [s] = pfeps.slopes(_rows(lambda e: table[e], sorted(table)))
    assert s.eps == [1e-4, 1e-3, 1e-2]
    assert s.r == [5, 3, 2]
    assert s.pair_slopes[0] == pytest.approx(-math.log10(5 / 3))
    assert s.pair_slopes[1] == pytest.approx(-math.log10(3 / 2))
    assert s.fit_slope == pytest.approx(-math.log10(5 / 2) / 2)
    assert s.small_r is True


def test_inconsistent_r_raises():
    rows = _rows(lambda e: 10, [1e-2, 1e-3])
    rows[1]["r"] = 11
    with pytest.raises(ValueError, match="two r"):
        pfeps.slopes(rows)
