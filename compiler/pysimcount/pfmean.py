"""Mean / geometric-mean summaries of the NumPy seed studies, beside the medians.

    python -m pysimcount.pfmean results/np-mean/config.json

The existing aggregations (``pfseeds``, ``pfgroupseeds``, ``pfoptseeds``)
summarise per-seed values by their median. This module reads the *same* runs
through the *same* paired functions and summarises each per-seed list by:

* the arithmetic mean, for counts and Trotter numbers (r, gate totals) --
  the average cost over random field instances, as in Childs et al. (2018),
  SI F.3 and App. I (mean over five random instances);
* the geometric mean, for per-seed ratios (gains, step ratios, r ratios,
  optimiser factors, bound / measured r). An arithmetic mean of ratios is not
  symmetric under swapping numerator and denominator: gains of 2 and 1/2
  average to 1.25, their geometric mean is 1.

Every table also shows the median (as in the existing aggregate files) and
the min-max over seeds, so a reader can see what the change of summary does.
Nothing in the existing aggregate directories is read except their configs
(the run lists), and nothing there is written.

What it writes (into the config's ``out`` directory):

* ``main-spread.md`` / ``.csv``: np-main, per (metric, gate set, scheme, n,
  r source) mean, median, min, max over seeds (r and gate totals);
* ``main-gains.md``: np-main, gain over the same-order Suzuki formula,
  geometric mean;
* ``main-bounds.md``: np-main, bound r / measured r, geometric mean;
* ``grouping-gains.md``: np-grouping, X-Y-Z / two-group gain, step ratio and
  r ratio, geometric mean;
* ``optimised.md``: np-optimised-seeds, mean r and totals, geometric-mean
  factor;
* ``summary.json`` (every number above), ``meta.json``, ``environment.txt``.

Seed win counts (``wins.md``) are counts, not averages, and are unchanged by
this choice; they are not rewritten here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from . import pfgroupseeds, pfoptseeds, pfseeds
from .pfcompare import provenance, write_environment


def mean(values: list[float]) -> float:
    """Arithmetic mean. Raises on an empty list."""
    if not values:
        raise ValueError("mean of no values")
    return math.fsum(values) / len(values)


def geomean(values: list[float]) -> float:
    """Geometric mean exp(mean(log x)) of positive ratios. Raises on a
    non-positive value (a ratio <= 0 means a zero or negative cost)."""
    if not values:
        raise ValueError("geometric mean of no values")
    for v in values:
        if not v > 0:
            raise ValueError(f"geometric mean needs positive values, got {v}")
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def summarise(per_seed: dict[int, float], kind: str) -> dict[str, float]:
    """{centre, median, min, max} of one per-seed list; centre is the mean for
    kind 'count' and the geometric mean for kind 'ratio'."""
    vals = [per_seed[s] for s in sorted(per_seed)]
    if kind == "count":
        centre = mean(vals)
    elif kind == "ratio":
        centre = geomean(vals)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return {"centre": centre, "median": statistics.median(vals), "min": min(vals), "max": max(vals)}


def _runs(aggregate_config: Path, per_seed_keys: tuple[str, ...] = ()) -> dict[int, dict]:
    cfg = json.loads(aggregate_config.read_text())
    return pfseeds.load_runs([aggregate_config.parent / p for p in cfg["runs"]], per_seed_keys)


def _num(x: float) -> str:
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.1f}"


# ------------------------------------------------------------------ np-main

def main_spread(runs: dict[int, dict]) -> list[dict]:
    out = []
    for metric, column, gb in pfseeds.SPREAD_METRICS:
        for x in pfseeds.aggregate(runs, metric, column, gb, merged=True):
            out.append({"metric": metric, "gatebase": gb, "scheme": x.scheme, "order": x.order, "n": x.n,
                        "r_source": x.r_source, "per_seed": x.values, **summarise(x.values, "count")})
    return out


def main_gains(runs: dict[int, dict]) -> list[dict]:
    return [{k: g[k] for k in ("metric", "base", "alt", "n", "r_source", "per_seed")}
            | summarise(g["per_seed"], "ratio") for g in pfseeds.paired_gains(runs)]


def main_bounds(runs: dict[int, dict]) -> list[dict]:
    return [{k: b[k] for k in ("scheme", "bound", "n", "denominator", "per_seed")}
            | summarise(b["per_seed"], "ratio") for b in pfseeds.bound_ratios(runs)]


# ------------------------------------------------------------------ np-grouping

def grouping_gains(runs: dict[int, dict]) -> list[dict]:
    out = []
    for g in pfgroupseeds.paired_gains(runs):
        rec = {k: g[k] for k in ("variant", "scheme", "n", "metric", "seeds_two_group_cheaper")}
        for name in ("gain", "step_ratio", "r_ratio"):
            rec[name] = summarise(g[name], "ratio") | {"per_seed": g[name]}
        out.append(rec)
    return out


# ------------------------------------------------------------------ np-optimised-seeds

def optimised(runs: dict[int, dict]) -> list[dict]:
    out = []
    for x in pfoptseeds.paired_factors(runs):
        rec = {k: x[k] for k in ("metric", "scheme", "n")}
        for name in ("r", "unopt_total", "optimised_total"):
            rec[name] = summarise(x[name], "count") | {"per_seed": x[name]}
        rec["factor"] = summarise(x["factor"], "ratio") | {"per_seed": x["factor"]}
        out.append(rec)
    return out


# ------------------------------------------------------------------ markdown

def spread_md(recs: list[dict]) -> str:
    lines = ["| metric | gate set | formula | n | r from | mean | median | min | max | mean / median |",
             "|---|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for x in recs:
        lines.append(f"| {x['metric']} | {x['gatebase']} | {x['scheme']} | {x['n']} | {x['r_source']} | "
                     f"{x['centre']:,.1f} | {_num(x['median'])} | {_num(x['min'])} | {_num(x['max'])} | "
                     f"{x['centre'] / x['median']:.3f} |")
    return "\n".join(lines)


def gains_md(recs: list[dict]) -> str:
    lines = ["| metric | pair | n | r from | geometric-mean gain | median | min | max |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for g in recs:
        lines.append(f"| {g['metric']} | {g['alt']} vs {g['base']} | {g['n']} | {g['r_source']} | "
                     f"{g['centre']:.3f}× | {g['median']:.3f}× | {g['min']:.3f}× | {g['max']:.3f}× |")
    return "\n".join(lines)


def bounds_md(recs: list[dict]) -> str:
    lines = ["| formula | bound | n | denominator | geometric-mean bound/measured | median | min | max |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for b in recs:
        lines.append(f"| {b['scheme']} | {b['bound']} | {b['n']} | {b['denominator']} | {b['centre']:,.3f} | "
                     f"{b['median']:,.3f} | {b['min']:,.3f} | {b['max']:,.3f} |")
    return "\n".join(lines)


def grouping_md(recs: list[dict], metric: str, n_seeds: int) -> str:
    lines = ["| variant | formula | n | geo-mean gain | median gain | min | max | geo-mean step ratio "
             f"| geo-mean r ratio | seeds two-group cheaper (of {n_seeds}) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in recs:
        if g["metric"] != metric:
            continue
        lines.append(f"| {g['variant']} | {g['scheme']} | {g['n']} | {g['gain']['centre']:.3f}× | "
                     f"{g['gain']['median']:.3f}× | {g['gain']['min']:.3f}× | {g['gain']['max']:.3f}× | "
                     f"{g['step_ratio']['centre']:.3f} | {g['r_ratio']['centre']:.3f} | "
                     f"{g['seeds_two_group_cheaper']} |")
    return "\n".join(lines)


def optimised_md(recs: list[dict], metric: str) -> str:
    lines = ["| formula | n | mean r | unoptimised (mean) | Nam Light (mean) | geo-mean factor | median factor "
             "| min | max |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in recs:
        if x["metric"] != metric:
            continue
        lines.append(f"| {x['scheme']} | {x['n']} | {x['r']['centre']:,.1f} | {x['unopt_total']['centre']:,.1f} | "
                     f"{x['optimised_total']['centre']:,.1f} | {x['factor']['centre']:.3f}× | "
                     f"{x['factor']['median']:.3f}× | {x['factor']['min']:.3f}× | {x['factor']['max']:.3f}× |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    here = args.config.parent
    out = here / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)

    main_runs = _runs(here / cfg["main"])
    group_runs = _runs(here / cfg["grouping"])
    opt_runs = _runs(here / cfg["optimised"], per_seed_keys=("source",))
    seeds = {"main": sorted(main_runs), "grouping": sorted(group_runs), "optimised": sorted(opt_runs)}

    spread = main_spread(main_runs)
    gains = main_gains(main_runs)
    bounds = main_bounds(main_runs)
    ggains = grouping_gains(group_runs)
    opt = optimised(opt_runs)

    with open(out / "main-spread.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "gatebase", "scheme", "order", "n", "r_source", "mean", "median", "min", "max"]
                   + [f"seed{s}" for s in seeds["main"]])
        for x in spread:
            w.writerow([x["metric"], x["gatebase"], x["scheme"], x["order"], x["n"], x["r_source"], x["centre"],
                        x["median"], x["min"], x["max"]] + [x["per_seed"][s] for s in seeds["main"]])
    intro = "Mean for counts and r, geometric mean for ratios; median, min and max over seeds alongside."
    (out / "main-spread.md").write_text("\n".join([
        f"# np-main: mean over seeds {seeds['main']} (merged counts)", "", intro, "", spread_md(spread), ""]))
    (out / "main-gains.md").write_text("\n".join([
        f"# np-main: gain over the same-order Suzuki formula, geometric mean over seeds {seeds['main']}", "",
        "gain = Suzuki cost / other cost on the same instance; > 1 favours the non-Suzuki formula.", "",
        gains_md(gains), ""]))
    (out / "main-bounds.md").write_text("\n".join([
        f"# np-main: bound r / measured r, geometric mean over seeds {seeds['main']} (cz)", "",
        "Denominator 'fit' marks extrapolated measured r.", "", bounds_md(bounds), ""]))
    body = [f"# np-grouping: X-Y-Z / two-group, geometric mean over seeds {seeds['grouping']} (merged)", "",
            "gain = step ratio × r ratio holds per seed, and therefore also for the geometric means "
            "(it does not hold for medians).", ""]
    for metric in pfgroupseeds.GAIN_METRICS:
        body += [f"## {metric}", "", grouping_md(ggains, metric, len(seeds["grouping"])), ""]
    (out / "grouping-gains.md").write_text("\n".join(body))
    body = [f"# np-optimised-seeds: mean totals, geometric-mean factor over seeds {seeds['optimised']}", "",
            "factor = unoptimised total / optimised total on the same seed (> 1: the optimiser saves).", ""]
    for metric in pfoptseeds.METRICS:
        body += [f"## {metric}", "", optimised_md(opt, metric), ""]
    (out / "optimised.md").write_text("\n".join(body))

    (out / "summary.json").write_text(json.dumps(
        {"main_spread": spread, "main_gains": gains, "main_bounds": bounds,
         "grouping_gains": ggains, "optimised": opt}, indent=1))
    meta = {"config": cfg, "provenance": provenance(), "seeds": seeds,
            "runs": {name: {s: r["meta"]["provenance"] for s, r in runs.items()}
                     for name, runs in (("main", main_runs), ("grouping", group_runs), ("optimised", opt_runs))}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    write_environment(out)
    print(f"{len(spread)} spread, {len(gains)} gain, {len(bounds)} bound, {len(ggains)} grouping, "
          f"{len(opt)} optimiser records; outputs in {out}")


if __name__ == "__main__":
    main()
