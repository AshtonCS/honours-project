"""Aggregate ``pfgrouping`` schemes runs that differ only in the seed.

    python -m plsimcount.pfgroupseeds results/np-grouping/aggregate.json

Each seed is a different random-field instance (see ``pfseeds``). The
comparison is *paired*: on each seed, for each formula and n, the X-Y-Z
grouping and a two-group variant are compared on the same instance, and only
then summarised by median, min and max over seeds.

For a cost column c (merged counts):

    gain = c(xyz) / c(two-group)        (> 1 favours two-group)

and since c = (cost per step) * r, the gain factors exactly into

    gain = step_ratio * r_ratio,
    step_ratio = c_per_step(xyz) / c_per_step(two-group),
    r_ratio    = r(xyz) / r(two-group).

A step ratio > 1 means the two-group step is cheaper; an r ratio < 1 means
two-group needs more steps (its splitting error is larger).

What it writes (into the config's ``out`` directory):

* ``gains.md`` / ``gains.csv``: per (variant, scheme, n, metric) the median,
  min and max of gain, step ratio and r ratio over seeds, and the number of
  seeds on which two-group is strictly cheaper;
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
from .pfseeds import load_runs

REFERENCE_VARIANT = "xyz"

# metric name -> (total column, per-step column)
GAIN_METRICS: dict[str, tuple[str, str]] = {
    "cnot": ("cnot_total", "cnot_per_step"),
    "total": ("gates_total", "gates_per_step"),
}


def _index(run: dict) -> dict[tuple[str, int, str], dict]:
    """{(scheme, n, variant): merged row} for one run. Raises on duplicates."""
    out: dict[tuple[str, int, str], dict] = {}
    for row in run["rows"]:
        if not row["merged"]:
            continue
        k = (row["scheme"], int(row["n"]), row["variant"])
        if k in out:
            raise ValueError(f"duplicate merged row {k} for seed {row['seed']}")
        out[k] = row
    return out


def paired_gains(runs: dict[int, dict]) -> list[dict]:
    """One record per (variant, scheme, n, metric) with per-seed gain, step
    ratio and r ratio of X-Y-Z over the variant, and their median/min/max.
    Raises if a seed lacks a row another seed has."""
    seeds = sorted(runs)
    idx = {s: _index(runs[s]) for s in seeds}
    keys = sorted(idx[seeds[0]])
    for s in seeds[1:]:
        if sorted(idx[s]) != keys:
            raise ValueError(f"seed {s} has a different set of (scheme, n, variant) rows from seed {seeds[0]}")
    order = runs[seeds[0]]["meta"]["config"]["schemes"]
    variants = [v for v in runs[seeds[0]]["meta"]["config"]["variants"] if v != REFERENCE_VARIANT]
    sizes = sorted({k[1] for k in keys})

    out: list[dict] = []
    for variant in variants:
        for scheme in order:
            for n in sizes:
                if (scheme, n, REFERENCE_VARIANT) not in idx[seeds[0]]:
                    continue
                for metric, (col, col_step) in GAIN_METRICS.items():
                    gain: dict[int, float] = {}
                    step: dict[int, float] = {}
                    rr: dict[int, float] = {}
                    for s in seeds:
                        ref = idx[s][(scheme, n, REFERENCE_VARIANT)]
                        alt = idx[s][(scheme, n, variant)]
                        gain[s] = ref[col] / alt[col]
                        step[s] = ref[col_step] / alt[col_step]
                        rr[s] = ref["r"] / alt["r"]
                    rec = {"variant": variant, "scheme": scheme, "n": n, "metric": metric,
                           "gain": gain, "step_ratio": step, "r_ratio": rr,
                           "seeds_two_group_cheaper": sum(1 for s in seeds if gain[s] > 1)}
                    for name in ("gain", "step_ratio", "r_ratio"):
                        vals = list(rec[name].values())
                        rec[f"{name}_median"] = statistics.median(vals)
                        rec[f"{name}_min"] = min(vals)
                        rec[f"{name}_max"] = max(vals)
                    out.append(rec)
    return out


def gains_markdown(gains: list[dict], n_seeds: int, metric: str) -> str:
    lines = ["| variant | formula | n | median gain | min | max | median step ratio | median r ratio "
             f"| seeds two-group cheaper (of {n_seeds}) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in gains:
        if g["metric"] != metric:
            continue
        lines.append(f"| {g['variant']} | {g['scheme']} | {g['n']} | {g['gain_median']:.2f}× | {g['gain_min']:.2f}× "
                     f"| {g['gain_max']:.2f}× | {g['step_ratio_median']:.2f} | {g['r_ratio_median']:.2f} "
                     f"| {g['seeds_two_group_cheaper']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    here = args.config.parent
    out = here / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs([here / p for p in cfg["runs"]])
    seeds = sorted(runs)
    gains = paired_gains(runs)

    with open(out / "gains.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "scheme", "n", "metric"] + [f"gain_seed{s}" for s in seeds]
                   + ["gain_median", "gain_min", "gain_max", "step_ratio_median", "r_ratio_median",
                      "r_ratio_min", "r_ratio_max", "seeds_two_group_cheaper"])
        for g in gains:
            w.writerow([g["variant"], g["scheme"], g["n"], g["metric"]] + [g["gain"][s] for s in seeds]
                       + [g["gain_median"], g["gain_min"], g["gain_max"], g["step_ratio_median"],
                          g["r_ratio_median"], g["r_ratio_min"], g["r_ratio_max"], g["seeds_two_group_cheaper"]])
    body = [f"# Two-group vs X-Y-Z grouping, per seed (seeds {seeds}, merged counts)", "",
            "gain = X-Y-Z cost / two-group cost on the same instance (> 1 favours two-group);",
            "gain = step ratio × r ratio, step ratio = X-Y-Z cost per step / two-group cost per step,",
            "r ratio = r(X-Y-Z) / r(two-group). All r measured (no fitted sizes).", ""]
    for metric in GAIN_METRICS:
        body += [f"## {metric}", "", gains_markdown(gains, len(seeds), metric), ""]
    (out / "gains.md").write_text("\n".join(body))
    (out / "paired.json").write_text(json.dumps({"gains": gains}, indent=1))
    meta = {"config": cfg, "provenance": provenance(),
            "runs": {s: runs[s]["meta"]["provenance"] for s in seeds}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    write_environment(out)
    print(f"{len(gains)} paired records; outputs in {out}")


if __name__ == "__main__":
    main()
