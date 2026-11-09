"""RQ4 table: merging and the Nam Light optimiser per formula, measured n, paired per seed.

    python -m pysimcount.pfopttable results/np-mean/rq4-optimiser.json

For each formula and measured n, on each seed (merged steps from np-main and
np-main-n9/-n11 via ``pfchilds.load_run_set``; optimiser runs from
``pfoptimised``, one per seed and size set):

    merging, exponentials = Rz_total(unmerged) / Rz_total(merged)      cz, measured r
    merging, CNOT         = CNOT_total(unmerged) / CNOT_total(merged)
    optimiser, CNOT / Rz / total = unoptimised total / Light total      (pfoptimised's
        reduction_factor: the merged step against the marginal optimised step, both x r)
    combined, CNOT        = merging CNOT x optimiser CNOT               (unmerged -> optimised)

Per-seed values are summarised by the geometric mean with min-max (``pfmean``'s
ratio rule). Rank: the formula's position by mean CNOT total over seeds among
all formulas at that n, merged (before) and optimised (after) -- the RQ4
question of whether the optimiser changes the ranking.

Checks (each raises): every optimiser run is clean (git_dirty False); its seed
and instance fields equal the main run's; its r equals the main run's measured
cz r; its unoptimised per-step CNOT and Rz equal the main run's merged per-step
counts; merged and unmerged rows use the same r.

Writes ``<out>.md`` (every n), ``<out>.tex`` (one tabular at ``table_n``) and
``<out>.json`` beside the config.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import pfchilds, pffigure
from .pfcompare import formula_registry, provenance
from .pfmean import geomean

FACTORS = ("merge_exp", "merge_cnot", "opt_cnot", "opt_rz", "opt_total", "combined_cnot")


def load_optimiser_runs(here: Path, patterns: list[str], seeds: list[int]) -> dict[int, list[dict]]:
    """{seed: [rows]} over every pattern; raises on a dirty run, a seed mismatch or a duplicate row."""
    out: dict[int, list[dict]] = {s: [] for s in seeds}
    seen = set()
    for pat in patterns:
        for s in seeds:
            d = json.loads((here / pat.format(seed=s) / "results.json").read_text())
            if d["meta"]["provenance"]["git_dirty"]:
                raise ValueError(f"{pat.format(seed=s)} was made from uncommitted code")
            if int(d["meta"]["config"]["seed"]) != s:
                raise ValueError(f"{pat.format(seed=s)} is seed {d['meta']['config']['seed']}")
            for row in d["rows"]:
                k = (s, row["scheme"], int(row["n"]), row["metric"])
                if k in seen:
                    raise ValueError(f"duplicate optimiser row {k}")
                seen.add(k)
                out[s].append(row | {"_fields": d["meta"].get("fields", {}).get(str(row["n"]))})
    return out


def _main_rows(runs: dict[int, dict]) -> dict[tuple[int, str, int, bool], dict]:
    out = {}
    for s, run in runs.items():
        inst = run["meta"]["config"].get("instances", "simcount")
        for row in run["rows"]:
            if row["instance"] == inst and row["gatebase"] == "cz" and row["r_source"] == "measured":
                out[(s, row["scheme"], int(row["n"]), bool(row["merged"]))] = row
    return out


def factors(runs: dict[int, dict], opt: dict[int, list[dict]], sizes: list[int]) -> list[dict]:
    """One record per (scheme, n) with per-seed factors and mean totals before/after."""
    seeds = sorted(runs)
    main = _main_rows(runs)
    by = {}
    for s in seeds:
        fields = runs[s]["meta"].get("fields", {})
        for row in opt[s]:
            n, k = int(row["n"]), row["scheme"]
            if n not in sizes:
                continue
            merged = main.get((s, k, n, True))
            if merged is None:
                raise ValueError(f"no merged measured cz row for {k} n={n} seed {s}")
            if row["_fields"] is not None and fields.get(str(n)) is not None and row["_fields"] != fields[str(n)]:
                raise ValueError(f"{k} n={n} seed {s}: optimiser instance fields differ from the main run's")
            if int(row["r"]) != int(merged["r"]):
                raise ValueError(f"{k} n={n} seed {s}: optimiser r {row['r']} != measured r {merged['r']}")
            col = {"CNOT": "cnot_per_step", "Rz": "rz_per_step", "total": "gates_per_step"}[row["metric"]]
            if int(row["unopt_per_step"]) != int(merged[col]):
                raise ValueError(f"{k} n={n} seed {s} {row['metric']}: optimiser input {row['unopt_per_step']} != "
                                 f"merged step {merged[col]}")
            by[(k, n, s, row["metric"])] = row
    out = []
    schemes = list(dict.fromkeys(row["scheme"] for row in opt[seeds[0]]))
    for n in sizes:
        recs = []
        for k in schemes:
            per = {f: {} for f in FACTORS}
            before, after = [], []
            for s in seeds:
                m, u = main[(s, k, n, True)], main.get((s, k, n, False))
                if u is None or int(u["r"]) != int(m["r"]):
                    raise ValueError(f"{k} n={n} seed {s}: no unmerged row with the merged row's r")
                o = {met: by.get((k, n, s, met)) for met in ("CNOT", "Rz", "total")}
                if any(v is None for v in o.values()):
                    raise ValueError(f"{k} n={n} seed {s}: optimiser rows missing {[m_ for m_, v in o.items() if v is None]}")
                per["merge_exp"][s] = u["rz_total"] / m["rz_total"]
                per["merge_cnot"][s] = u["cnot_total"] / m["cnot_total"]
                per["opt_cnot"][s] = o["CNOT"]["unopt_total"] / o["CNOT"]["marginal_total"]
                per["opt_rz"][s] = o["Rz"]["unopt_total"] / o["Rz"]["marginal_total"]
                per["opt_total"][s] = o["total"]["unopt_total"] / o["total"]["marginal_total"]
                per["combined_cnot"][s] = per["merge_cnot"][s] * per["opt_cnot"][s]
                before.append(m["cnot_total"])
                after.append(o["CNOT"]["marginal_total"])
            rec = {"scheme": k, "n": n, "cnot_before_mean": math.fsum(before) / len(before),
                   "cnot_after_mean": math.fsum(after) / len(after)}
            for f in FACTORS:
                vals = list(per[f].values())
                rec[f] = {"geomean": geomean(vals), "min": min(vals), "max": max(vals), "per_seed": per[f]}
            recs.append(rec)
        for key, rank in (("cnot_before_mean", "rank_before"), ("cnot_after_mean", "rank_after")):
            for i, rec in enumerate(sorted(recs, key=lambda r: r[key])):
                rec[rank] = i + 1
        out += recs
    return out


def tables(recs: list[dict], names: dict[str, str], table_n: int, seeds: list[int]) -> tuple[str, str]:
    g = lambda x: f"{x['geomean']:.2f}"
    rng = lambda x: f"{x['min']:.2f}–{x['max']:.2f}" if f"{x['min']:.2f}" != f"{x['max']:.2f}" else ""
    md = ["| formula | n | merging: exponentials | merging: CNOT | optimiser: CNOT | optimiser: Rz | optimiser: total "
          "| combined CNOT | CNOT rank before → after |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in recs:
        cells = [f"{g(r[f])}" + (f" ({rng(r[f])})" if rng(r[f]) else "") for f in FACTORS]
        md.append(f"| {names[r['scheme']]} | {r['n']} | " + " | ".join(cells) + f" | {r['rank_before']} → {r['rank_after']} |")
    tex = [f"% merging and Nam Light optimiser factors at n = {table_n}; geometric mean over seeds {seeds}; "
           "CNOT rank by mean CNOT total over seeds", "\\begin{tabular}{lrrrrrrc}", "\\hline",
           "Formula & \\multicolumn{2}{c}{merging} & \\multicolumn{3}{c}{optimiser} & combined & CNOT rank \\\\",
           " & exponentials & CNOT & CNOT & $R_z$ & total & CNOT & before $\\to$ after \\\\", "\\hline"]
    for r in recs:
        if r["n"] != table_n:
            continue
        tex.append(f"{pffigure.latex_escape(names[r['scheme']])} & "
                   + " & ".join(f"{r[f]['geomean']:.2f}" for f in FACTORS)
                   + f" & {r['rank_before']} $\\to$ {r['rank_after']} \\\\")
    tex += ["\\hline", "\\end{tabular}", ""]
    return "\n".join(md) + "\n", "\n".join(tex)


def run(cfg: dict, here: Path) -> dict:
    runs = pfchilds.load_run_set(here, cfg["runs"])
    seeds = sorted(runs)
    sizes = cfg["sizes"]
    if cfg["table_n"] not in sizes:
        raise ValueError(f"table_n {cfg['table_n']} is not among the sizes {sizes}")
    opt = load_optimiser_runs(here, cfg["optimiser_runs"], seeds)
    recs = factors(runs, opt, sizes)
    registry = formula_registry()
    names = {k: registry[k].name for k in registry} | cfg.get("names", {})
    md, tex = tables(recs, names, cfg["table_n"], seeds)
    prov = provenance()
    head = ("# Merging and the Nam Light optimiser per formula (RQ4), measured n\n\n"
            f"Written by `python -m pysimcount.pfopttable`. Seeds {seeds}, cz, measured r, paired per seed; "
            "geometric mean, min–max in brackets where it varies. Merging = unmerged / merged (np-main rows); "
            "optimiser = merged / Light-optimised (pfoptimised, marginal per step x r); combined = product. "
            "Rank = position by mean CNOT total over seeds among all formulas at that n (1 = cheapest). "
            "Every optimiser run checked against the main run: same seed, fields, r and merged per-step count. "
            f"Made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}.\n\n")
    out = here / cfg["out"]
    out.with_suffix(".md").write_text(head + md)
    out.with_suffix(".tex").write_text("% Generated by pysimcount.pfopttable; "
                                       f"git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}; "
                                       "do not edit, regenerate instead.\n" + tex)
    result = {"seeds": seeds, "sizes": sizes, "table_n": cfg["table_n"], "provenance": prov, "rows": recs}
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    print(f"  wrote {out.with_suffix('.md')}, .tex, .json ({len(recs)} rows)")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    if (args.config.parent / cfg["out"]).with_suffix(".json").resolve() == args.config.resolve():
        raise ValueError("the output would overwrite the config; choose another 'out'")
    run(cfg, args.config.parent)


if __name__ == "__main__":
    main()
