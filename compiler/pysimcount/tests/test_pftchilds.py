"""pftchilds reproduces a run's T total from scratch, and its record feeds figleader."""

import json
import math
from pathlib import Path

import pytest

from pysimcount import figdrop, figinset, figleader, figuniform, pfchilds, pftchilds
from pysimcount.pfcompare import formula_registry

SCHEMES = ["suzuki4", "bm-q10"]
# measured ct r at n = 5, 6 per seed, and T per step (any integers; T total = T per step x r)
R = {("suzuki4", 1): (10, 12), ("suzuki4", 2): (11, 14), ("bm-q10", 1): (6, 8), ("bm-q10", 2): (5, 7)}
T_STEP = {"suzuki4": 1000, "bm-q10": 3000}


def _runs() -> dict[int, dict]:
    def row(k: str, s: int, n: int, merged: bool = True, gb: str = "ct") -> dict:
        r = R[(k, s)][n - 5]
        return {"gatebase": gb, "r_source": "measured", "merged": merged, "scheme": k, "n": n, "r": r,
                "t_per_step": T_STEP[k] * n, "t_total": T_STEP[k] * n * r}
    out = {}
    for s in (1, 2):
        rows = [row(k, s, n) for k in SCHEMES for n in (5, 6)]
        rows += [row("suzuki4", s, 5, merged=False), row("suzuki4", s, 5, gb="cz")]  # filtered out
        out[s] = {"rows": rows}
    return out


def test_ct_rows_keep_measured_merged_ct_only_and_check_totals():
    rows = pftchilds.ct_rows(_runs(), SCHEMES)
    assert sorted(rows) == [("bm-q10", 5), ("bm-q10", 6), ("suzuki4", 5), ("suzuki4", 6)]
    assert rows[("suzuki4", 6)][2]["r"] == 14
    bad = _runs()
    bad[1]["rows"][0]["t_total"] += 1
    with pytest.raises(ValueError, match="T total != T per step x r"):
        pftchilds.ct_rows(bad, SCHEMES)


def test_per_seed_fit_by_hand():
    # r = 10, 12 at n = 5, 6: b = log(12/10) / log(6/5) = 1 exactly, a = 2
    fits = pftchilds.per_seed_fits(pftchilds.ct_rows(_runs(), SCHEMES), "suzuki4", [1, 2], [5, 6])
    assert fits[1] == pytest.approx((2.0, 1.0))
    with pytest.raises(ValueError, match="no measured ct r at n=7"):
        pftchilds.per_seed_fits(pftchilds.ct_rows(_runs(), SCHEMES), "suzuki4", [1, 2], [5, 7])


def _setup(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.setattr(pftchilds.pfchilds, "load_run_set", lambda here, spec: _runs())
    (tmp_path / "agg.json").write_text(json.dumps({"runs": []}))
    (tmp_path / "like.json").write_text(json.dumps({"shown": SCHEMES, "seeds": [1, 2], "removed": ["suzuki8"]}))
    cfg = {"id": "T", "title": "t", "runs": "agg.json", "like": "like.json", "fit_sizes": [5, 6],
           "extrapolate_sizes": [10], "workers": 1, "cache": "cache.json"}
    rows = pftchilds.ct_rows(_runs(), SCHEMES)
    entries = {}
    for k in SCHEMES:
        fits = pftchilds.per_seed_fits(rows, k, [1, 2], [5, 6])
        for s in (1, 2):
            r = pfchilds.r_fit_ceil(*fits[s], 10)
            entries[f"{k}|{s}|10"] = {"r": r, "t_per_step": T_STEP[k] * 10, "rotations_per_step": 10,
                                      "t_total": T_STEP[k] * 10 * r, "eps_rot": 1e-9}
    (tmp_path / "cache.json").write_text(json.dumps({"checked": True, "entries": entries}))
    return cfg


def test_record_values_by_hand(tmp_path, monkeypatch):
    rec = pftchilds.build_record(_setup(tmp_path, monkeypatch), tmp_path)
    m = rec["series"]["suzuki4"]["measured"][1]  # n = 6: 6000 x 12, 6000 x 14
    assert (m["mean"], m["min"], m["max"]) == (78000, 72000, 84000) and m["per_seed"] == {"1": 72000, "2": 84000}
    e = rec["series"]["bm-q10"]["extrapolated"][0]
    v = list(e["per_seed"].values())
    assert e["value"] == sum(v) / 2 and (e["min"], e["max"]) == (min(v), max(v))
    assert e["t_per_rotation_mean"] == 3000
    assert rec["per_seed_stored"] and rec["removed"] == ["suzuki8"]


def test_record_refuses_unchecked_or_stale_cache(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch)
    cache = json.loads((tmp_path / "cache.json").read_text())
    cache["entries"]["suzuki4|1|10"]["r"] += 1
    (tmp_path / "cache.json").write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="does not match the current fit"):
        pftchilds.build_record(cfg, tmp_path)
    cache["checked"] = False
    (tmp_path / "cache.json").write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="has not passed the check"):
        pftchilds.build_record(cfg, tmp_path)


def test_figure_parses_and_feeds_figleader(tmp_path, monkeypatch):
    rec = pftchilds.build_record(_setup(tmp_path, monkeypatch), tmp_path)
    tex = pftchilds.figure_tex(rec, formula_registry())
    _, plots, trailing = figinset.parse_figure(tex)
    assert len(plots) == 4 and len(trailing) == 4
    assert not figuniform.in_scope(tex)
    rows = figleader.leaderboard(rec, {}, 10)
    assert sum(r["wins"] for r in rows) == 2
    by = {r["scheme"]: r for r in rows}
    # the dot sits at the stored mean, and the ranks come from the stored per-seed values
    assert by["bm-q10"]["value"] == rec["series"]["bm-q10"]["extrapolated"][0]["value"]
    assert figleader.per_seed_at(rec, {}, 6)["suzuki4"] == {1: 72000, 2: 84000}
    side = figleader.side_by_side_tex(tex, rec, rows, 10, {
        "main": {"width_cm": 6, "height_cm": 5, "xmin": 3, "xmax": 20, "ymin": 1e4, "ymax": 1e7, "ylabel": "T",
                 "mark_size": "1pt", "legend_columns": 2, "legend_y": -0.2, "band_factor": 1.1},
        "dots": {"width_cm": 4, "height_cm": 5, "gap_cm": 3, "title": "t", "xlabel": "x", "xmin": 1e5, "xmax": 1e7,
                 "x_unit": 1e5, "xtick": [1e6], "mark_size": "2pt", "columns_cm": [0.5, 1.4]}},
        ["% derived: T.tex"], formula_registry())
    assert figdrop.has_fixed_y(side) and "\\begin{semilogxaxis}" in side
    assert "\\addlegendentry{Suzuki S4}" in side and "+=" not in side.split("\\end{loglogaxis}")[0]


def test_synthesised_step_reproduces_a_run():
    # From scratch: the merged ct step of bm-q10 on NumPy seed 1 at n = 4, measured r, as np-main recorded it.
    here = Path(__file__).resolve().parents[2] / "results" / "np-main" / "seed1" / "results.json"
    if not here.exists():
        pytest.skip("results not present")
    rows = [r for r in json.loads(here.read_text())["rows"] if r["gatebase"] == "ct" and r["merged"]
            and r["r_source"] == "measured" and r["scheme"] == "bm-q10" and r["n"] == 4]
    assert len(rows) == 1
    got = pftchilds.synthesised_step("bm-q10", 1, 4, int(rows[0]["r"]), "simcount", "pygridsynth")
    assert got["t_total"] == int(rows[0]["t_total"])
    assert got["eps_rot"] == pytest.approx(float(rows[0]["eps_rot"]), rel=1e-12)
    assert math.isclose(got["eps_rot"] * got["rotations_per_step"] * got["r"], 5e-4)
