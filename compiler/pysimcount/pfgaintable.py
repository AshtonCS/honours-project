"""RQ1 table: gain over the same-order Suzuki formula at every measured n, paired per seed.

    python -m pysimcount.pfgaintable results/np-mean/gains-table.json

For each non-Suzuki formula X and the Suzuki formula S of the same order, on
each seed (same random-field instance for both), merged steps:

    gain (exponentials) = Rz_total(S) / Rz_total(X)            cz, measured r
    gain (CNOT)         = CNOT_total(S) / CNOT_total(X)        cz, measured r
    gain (T)            = T_total(S) / T_total(X)              ct, measured r
    per-step ratio      = Rz_per_step(S) / Rz_per_step(X)      exact, the same on every seed
    r ratio             = r(S) / r(X)                           cz

One exponential is one Rz after merging, so gain (exponentials) = per-step
ratio x r ratio holds exactly on each seed; this is checked, and it also holds
for the geometric means (not for medians). Each per-seed list is summarised by
its geometric mean with min-max (the ratio rule of ``pfmean``). Only measured
r is used: no fitted sizes.

Eff prediction: M&O's efficiency Eff = 1 / (q^p Err_p) (arXiv:2601.18756,
Eq. 22) implies (Eff_X / Eff_S)^(1/p) times fewer exponentials than S at the
same accuracy; computed by ``pfreport._eff_prediction``, unchanged. The 1/p
follows from the definition (cost per unit time ~ q Err^(1/p) at fixed error);
it has not been re-checked against the paper's text here.

Runs come from ``pfchilds.load_run_set`` (np-main plus np-main-n9/-n11 under
the documented dirty exception), every seed present at every n.

Writes ``<out>.md`` (every n), ``<out>.tex`` (one tabular at ``table_n``) and
``<out>.json`` beside the config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import pfchilds, pffigure, pfseeds
from .pfcompare import formula_registry, provenance
from .pfmean import geomean

ORDER = ["mo-q6", "mo-q14", "bm-q10", "mo-o8-q21", "morales-y8m10", "morales-m8"]


def _per_seed(runs: dict[int, dict], column: str, gatebase: str) -> dict[int, dict[tuple[str, int], float]]:
    """{seed: {(scheme, n): value}} from merged, measured rows."""
    out = {}
    for s, run in runs.items():
        v = pfseeds._values(run, column, gatebase, sources=("measured",))
        out[s] = {(k, n): x for (k, n, _), x in v.items()}
    return out


def _summary(per: dict[int, float]) -> dict:
    vals = list(per.values())
    return {"geomean": geomean(vals), "min": min(vals), "max": max(vals), "per_seed": per}


def gains(runs: dict[int, dict], sizes: list[int]) -> list[dict]:
    """One record per (pair, n). Raises if a seed lacks a row or the decomposition fails."""
    seeds = sorted(runs)
    rows0 = runs[seeds[0]]["rows"]
    pairs = sorted(pfseeds.same_order_pairs(rows0), key=lambda p: ORDER.index(p[1]))
    cols = {"exp": _per_seed(runs, "rz_total", "cz"), "cnot": _per_seed(runs, "cnot_total", "cz"),
            "t": _per_seed(runs, "t_total", "ct"), "r": _per_seed(runs, "r", "cz"),
            "step": _per_seed(runs, "rz_per_step", "cz")}
    out = []
    for base, alt in pairs:
        for n in sizes:
            rec = {"base": base, "alt": alt, "n": n}
            for name, col in cols.items():
                per = {}
                for s in seeds:
                    if (base, n) not in col[s] or (alt, n) not in col[s]:
                        raise ValueError(f"seed {s} has no measured {name} row for {base} or {alt} at n={n}")
                    per[s] = col[s][(base, n)] / col[s][(alt, n)]
                rec[name] = _summary(per)
            steps = set(rec["step"]["per_seed"].values())
            if len(steps) != 1:
                raise ValueError(f"{alt} vs {base} n={n}: per-step ratio differs across seeds {sorted(steps)}")
            for s in seeds:
                lhs, rhs = rec["exp"]["per_seed"][s], rec["step"]["per_seed"][s] * rec["r"]["per_seed"][s]
                if abs(lhs - rhs) > 1e-12 * lhs:
                    raise ValueError(f"{alt} vs {base} n={n} seed {s}: exponential gain {lhs} != step x r {rhs}")
            out.append(rec)
    return out


def _fmt(x: dict) -> str:
    return f"{x['geomean']:.2f}"


def _rng(x: dict) -> str:
    return f"{x['min']:.2f}–{x['max']:.2f}"


def tables(recs: list[dict], eff: dict[str, float], names: dict[str, str], table_n: int,
           seeds: list[int]) -> tuple[str, str]:
    md = ["| pair | n | per-step ratio | r ratio | gain: exponentials (min–max) | gain: CNOT (min–max) "
          "| gain: T (min–max) | Eff prediction |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in recs:
        md.append(f"| {names[g['alt']]} vs {names[g['base']]} | {g['n']} | {_fmt(g['step'])} | {_fmt(g['r'])} | "
                  f"{_fmt(g['exp'])} ({_rng(g['exp'])}) | {_fmt(g['cnot'])} ({_rng(g['cnot'])}) | "
                  f"{_fmt(g['t'])} ({_rng(g['t'])}) | {eff[g['alt']]:.2f} |")
    tex = [f"% gain over the same-order Suzuki formula at n = {table_n}; geometric mean over seeds {seeds}, "
           "min--max in brackets; merged; measured r",
           "\\begin{tabular}{lrrrrrr}", "\\hline",
           "Pair & per-step & $r$ & \\multicolumn{3}{c}{gain} & Eff \\\\",
           " & ratio & ratio & exponentials & CNOT & T & prediction \\\\", "\\hline"]
    for g in recs:
        if g["n"] != table_n:
            continue
        cell = lambda x: f"{x['geomean']:.2f} ({x['min']:.2f}--{x['max']:.2f})"
        tex.append(f"{pffigure.latex_escape(names[g['alt']])} vs {pffigure.latex_escape(names[g['base']])} & "
                   f"{g['step']['geomean']:.2f} & {g['r']['geomean']:.2f} & {cell(g['exp'])} & {cell(g['cnot'])} & "
                   f"{cell(g['t'])} & {eff[g['alt']]:.2f} \\\\")
    tex += ["\\hline", "\\end{tabular}", ""]
    return "\n".join(md) + "\n", "\n".join(tex)


def run(cfg: dict, here: Path) -> dict:
    from .pfreport import _eff_prediction
    runs = pfchilds.load_run_set(here, cfg["runs"])
    seeds = sorted(runs)
    sizes = cfg["sizes"]
    if cfg["table_n"] not in sizes:
        raise ValueError(f"table_n {cfg['table_n']} is not among the sizes {sizes}")
    recs = gains(runs, sizes)
    registry = formula_registry()
    names = cfg.get("names") or {k: registry[k].name for k in registry}
    orders = {k: registry[k].order for k in registry}
    eff = {g["alt"]: _eff_prediction(orders[g["alt"]], g["base"], g["alt"]) for g in recs}
    md, tex = tables(recs, eff, names, cfg["table_n"], seeds)
    prov = provenance()
    head = ("# Gain over the same-order Suzuki formula (RQ1), measured n only\n\n"
            f"Written by `python -m pysimcount.pfgaintable`. Seeds {seeds}, paired per seed, merged steps, measured r "
            f"(np-main + np-main-n9/-n11). Geometric mean over seeds, min–max in brackets. gain = Suzuki / other "
            "(> 1 favours the other formula). Exponentials and CNOT in cz (eps = 1e-3), T in ct (5e-4 Trotter + "
            "5e-4 synthesis). Checked on every seed: exponential gain = per-step ratio x r ratio. Eff prediction = "
            "(Eff_X / Eff_Suzuki)^(1/p), M&O Eq. 22, via pfreport (exponent not re-checked against the paper). "
            f"Made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}.\n\n")
    out = here / cfg["out"]
    out.with_suffix(".md").write_text(head + md)
    out.with_suffix(".tex").write_text(f"% Generated by pysimcount.pfgaintable from {cfg['runs']['aggregate'] if isinstance(cfg['runs'], dict) else cfg['runs']}; "
                                       f"git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}; "
                                       "do not edit, regenerate instead.\n" + tex)
    result = {"seeds": seeds, "sizes": sizes, "table_n": cfg["table_n"], "eff_prediction": eff,
              "provenance": prov, "rows": recs}
    out.with_suffix(".json").write_text(json.dumps(result, indent=1, default=str))
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
