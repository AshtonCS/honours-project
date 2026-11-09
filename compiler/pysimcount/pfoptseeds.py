"""Aggregate ``pfoptimised`` runs that differ only in the seed.

    python -m pysimcount.pfoptseeds results/np-optimised-seeds/aggregate.json

Each seed is a different random-field instance (see ``pfseeds``), with r the
measured cz r of that seed's ``np-main`` run. The comparison is *paired*: on
each seed, for each formula, n and metric (CNOT, Rz, total gates), the
unoptimised total and the Nam Light total are taken on the same instance,

    factor = unopt_total / optimised_total      (> 1: the optimiser saves),

with optimised_total = marginal per step x r (see ``pfoptimised``), and only
then summarised by median, min and max over seeds. Both totals use the same r,
so the factor is the per-step ratio unopt / marginal; it varies over seeds
only if the optimiser finds different cancellations on different field
values.

What it writes (into the config's ``out`` directory):

* ``side-by-side.md`` / ``side-by-side.csv``: per (metric, scheme, n) the
  median r, the median unoptimised and optimised totals next to each other,
  and the median, min and max of the paired factor;
* ``paired.json``: the same with every per-seed value;
* ``meta.json`` / ``environment.txt``: provenance of this aggregation and of
  every run.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from .pfcompare import provenance, write_environment
from .pfoptimised import METRICS
from .pfseeds import load_runs


def _index(run: dict) -> dict[tuple[str, int, str], dict]:
    """{(scheme, n, metric): row} for one pfoptimised run. Raises on duplicates."""
    out: dict[tuple[str, int, str], dict] = {}
    for row in run["rows"]:
        k = (row["scheme"], int(row["n"]), row["metric"])
        if k in out:
            raise ValueError(f"duplicate row {k} in seed {run['meta']['config']['seed']}")
        out[k] = row
    return out


def paired_factors(runs: dict[int, dict]) -> list[dict]:
    """One record per (metric, scheme, n) with per-seed r, unoptimised and
    optimised totals and factor, and their median/min/max over seeds.
    Raises if a seed lacks a row another seed has."""
    seeds = sorted(runs)
    idx = {s: _index(runs[s]) for s in seeds}
    keys = sorted(idx[seeds[0]])
    for s in seeds[1:]:
        if sorted(idx[s]) != keys:
            raise ValueError(f"seed {s} has a different set of (scheme, n, metric) rows from seed {seeds[0]}")
    order = runs[seeds[0]]["meta"]["config"]["schemes"]
    sizes = sorted({k[1] for k in keys})

    out: list[dict] = []
    for metric in METRICS:
        for scheme in order:
            for n in sizes:
                r: dict[int, int] = {}
                unopt: dict[int, int] = {}
                opt: dict[int, int] = {}
                factor: dict[int, float] = {}
                for s in seeds:
                    row = idx[s][(scheme, n, metric)]
                    if row["marginal_total"] <= 0:
                        raise ValueError(f"seed {s} {scheme} n={n} {metric}: optimised total is {row['marginal_total']}")
                    r[s] = int(row["r"])
                    unopt[s] = int(row["unopt_total"])
                    opt[s] = int(row["marginal_total"])
                    factor[s] = unopt[s] / opt[s]
                rec = {"metric": metric, "scheme": scheme, "n": n,
                       "r": r, "unopt_total": unopt, "optimised_total": opt, "factor": factor}
                for name in ("r", "unopt_total", "optimised_total", "factor"):
                    vals = list(rec[name].values())
                    rec[f"{name}_median"] = statistics.median(vals)
                    rec[f"{name}_min"] = min(vals)
                    rec[f"{name}_max"] = max(vals)
                out.append(rec)
    return out


def _num(x: float) -> str:
    """Counts as integers with thousands separators; a median of an even
    number of seeds can end in .5, which is kept."""
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.1f}"


def side_by_side_markdown(recs: list[dict], metric: str) -> str:
    lines = ["| formula | n | median r | unoptimised (median) | Nam Light (median) | median factor | min | max |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in recs:
        if x["metric"] != metric:
            continue
        lines.append(f"| {x['scheme']} | {x['n']} | {_num(x['r_median'])} | {_num(x['unopt_total_median'])} "
                     f"| {_num(x['optimised_total_median'])} | {x['factor_median']:.3f}× "
                     f"| {x['factor_min']:.3f}× | {x['factor_max']:.3f}× |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    here = args.config.parent
    out = here / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    # each seed's r comes from its own np-main run, so ``source`` differs by seed
    runs = load_runs([here / p for p in cfg["runs"]], per_seed_keys=("source",))
    seeds = sorted(runs)
    for s in seeds:
        src = runs[s]["meta"]["config"]["source"]
        if Path(src).parent.name != f"seed{s}":
            raise ValueError(f"seed {s} run reads r from {src}, not from a seed{s} run")
    recs = paired_factors(runs)

    with open(out / "side-by-side.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "scheme", "n", "r_median", "unopt_total_median", "optimised_total_median",
                    "factor_median", "factor_min", "factor_max"] + [f"factor_seed{s}" for s in seeds])
        for x in recs:
            w.writerow([x["metric"], x["scheme"], x["n"], x["r_median"], x["unopt_total_median"],
                        x["optimised_total_median"], x["factor_median"], x["factor_min"], x["factor_max"]]
                       + [x["factor"][s] for s in seeds])
    body = [f"# Unoptimised vs Nam Light optimised cz totals, per seed (seeds {seeds}, merged steps, measured r)", "",
            "Each seed's r is the measured cz r of np-main/seed<s>. Optimised total = Light marginal per step × r",
            "(pfoptimised). factor = unoptimised total / optimised total on the same seed (> 1: the optimiser saves);",
            "medians of the totals and of the factor are taken separately, so median unopt / median opt need not equal",
            "the median factor.", ""]
    for metric in METRICS:
        body += [f"## {metric}", "", side_by_side_markdown(recs, metric), ""]
    (out / "side-by-side.md").write_text("\n".join(body))
    (out / "paired.json").write_text(json.dumps({"records": recs}, indent=1))
    meta = {"config": cfg, "provenance": provenance(),
            "runs": {s: runs[s]["meta"]["provenance"] for s in seeds}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    write_environment(out)
    print(f"{len(recs)} paired records; outputs in {out}")


if __name__ == "__main__":
    main()
