"""How the extrapolated ranking depends on which measured sizes the power law of r is fitted to.

    python -m pysimcount.pffitwindow results/np-childs-format/fit-windows.json

For each fit window (a list of measured n) and each formula, this applies
pfchilds's extrapolation rule unchanged: mean measured r over seeds at each n
of the window, ``pfcompare.fit_power_law`` (least squares in log-log), then

    count(n) = c(n) ceil(a n^b),     c(n) = the exact affine per-step count of the record

and reports, at each configured n, every formula's count divided by the
cheapest formula's count under the same window (1.00 = predicted cheapest).
The fitted exponents b are reported too, since they decide the ranking far
from the data.

Measured r is read from the merged cz rows of every run directory listed in
``runs`` (np-main has n = 4-8, 10; np-main-n9 and np-main-n11 add n = 9, 11),
with every seed of the record present at every n. Raises unless the window
equal to the record's own ``fit_sizes`` reproduces the record's extrapolated
values exactly, so the table and the figures share one rule.

Writes ``<out>.md``, ``<out>.tex`` (one tabular per n, plain \\hline rules)
and ``<out>.json`` beside the config.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

from . import pfchilds, pffigure
from .pfcompare import fit_power_law, formula_registry


def measured_r(run_dirs: list[Path], schemes: list[str], seeds: list[int]) -> dict[str, dict[int, dict[int, int]]]:
    """{scheme: {n: {seed: r}}} from the merged, measured cz rows of every run."""
    out: dict[str, dict[int, dict[int, int]]] = {k: {} for k in schemes}
    for d in run_dirs:
        res = json.loads((d / "results.json").read_text())
        seed = int(res["meta"]["config"]["seed"])
        if seed not in seeds:
            continue
        for row in res["rows"]:
            if not (row["gatebase"] == "cz" and row["merged"] and row["r_source"] == "measured"
                    and row["scheme"] in out):
                continue
            per = out[row["scheme"]].setdefault(int(row["n"]), {})
            if seed in per:
                raise ValueError(f"seed {seed}: two measured rows for {row['scheme']} n={row['n']}")
            per[seed] = int(row["r"])
    return out


def window_counts(rs: dict[str, dict[int, dict[int, int]]], alpha: dict[str, tuple[Fraction, Fraction]],
                  window: list[int], sizes: list[int], seeds: list[int]) -> dict:
    """{"b": {scheme: b}, "count": {n: {scheme: count}}} for one fit window."""
    b_out, counts = {}, {n: {} for n in sizes}
    for k, per_n in rs.items():
        means = []
        for n in window:
            per = per_n.get(n, {})
            if sorted(per) != seeds:
                raise ValueError(f"{k} n={n}: seeds {sorted(per)}, expected {seeds}")
            means.append(math.fsum(per.values()) / len(seeds))
        a, b = fit_power_law(window, means)
        b_out[k] = float(b)
        for n in sizes:
            counts[n][k] = pfchilds.affine_eval(*alpha[k], n) * pfchilds.r_fit_ceil(a, b, n)
    return {"b": b_out, "count": counts}


def ratios(counts: dict[str, int]) -> dict[str, float]:
    best = min(counts.values())
    return {k: v / best for k, v in counts.items()}


def label(window: list[int]) -> str:
    """'5-11', or '5-10 (no 9)' when the window skips sizes inside its range."""
    missing = [n for n in range(window[0], window[-1] + 1) if n not in window]
    return f"{window[0]}-{window[-1]}" + (f" (no {', '.join(map(str, missing))})" if missing else "")


def _tex_label(window: list[int]) -> str:
    missing = [n for n in range(window[0], window[-1] + 1) if n not in window]
    return f"$n = {window[0]}$--${window[-1]}$" + (f" (no {', '.join(map(str, missing))})" if missing else "")


def tables(result: dict, schemes: list[str], names: dict[str, str], order_by: str) -> tuple[str, str]:
    windows = result["windows"]
    labels = [w["label"] for w in windows]
    md, tex = [], []
    for n in result["sizes"]:
        ref = next(w for w in windows if w["label"] == order_by)
        rows = sorted(schemes, key=lambda k: ref["ratio"][str(n)][k])
        md += [f"## n = {n}: count / cheapest count under the same fit window", "",
               "| formula | " + " | ".join(f"fit n = {l}" for l in labels) + " |",
               "|---|" + "---:|" * len(labels)]
        tex += [f"% n = {n}: count / cheapest count under the same fit window",
                "\\begin{tabular}{l" + "r" * len(labels) + "}", "\\hline",
                "Formula & " + " & ".join(_tex_label(w["sizes"]) for w in windows) + " \\\\", "\\hline"]
        for k in rows:
            cells = [w["ratio"][str(n)][k] for w in windows]
            md.append(f"| {names[k]} | " + " | ".join(f"**{c:.2f}**" if c == 1 else f"{c:.2f}" for c in cells) + " |")
            tex.append(f"{pffigure.latex_escape(names[k])} & "
                       + " & ".join(f"\\textbf{{{c:.2f}}}" if c == 1 else f"{c:.2f}" for c in cells) + " \\\\")
        md.append("")
        tex += ["\\hline", "\\end{tabular}", ""]
    md += ["## Fitted exponent b of r = a n^b", "",
           "| formula | " + " | ".join(f"fit n = {l}" for l in labels) + " |", "|---|" + "---:|" * len(labels)]
    for k in schemes:
        md.append(f"| {names[k]} | " + " | ".join(f"{w['b'][k]:.3f}" for w in windows) + " |")
    return "\n".join(md) + "\n", "\n".join(tex)


def run(cfg: dict, here: Path) -> dict:
    rec = json.loads((here / cfg["record"]).read_text())
    schemes, seeds = rec["shown"], rec["seeds"]
    alpha = {}
    for k in schemes:
        m = rec["series"][k]["per_step_model"]
        alpha[k] = (Fraction(m["alpha"]), Fraction(m["beta"]))
    run_dirs = sorted(p for pattern in cfg["runs"] for p in here.glob(pattern) if p.is_dir())
    rs = measured_r(run_dirs, schemes, seeds)
    sizes = cfg["sizes"]

    # the record's own window must reproduce the record's extrapolated values
    own = window_counts(rs, alpha, rec["fit_sizes"], rec["extrapolate_sizes"], seeds)
    for k in schemes:
        for p in rec["series"][k]["extrapolated"]:
            if own["count"][p["n"]][k] != p["value"]:
                raise ValueError(f"{k} n={p['n']}: window {rec['fit_sizes']} gives {own['count'][p['n']][k]}, "
                                 f"the record {p['value']}")

    windows = []
    for w in cfg["windows"]:
        r = window_counts(rs, alpha, w, sizes, seeds)
        windows.append({"sizes": w, "label": label(w), "b": r["b"],
                        "count": {str(n): r["count"][n] for n in sizes},
                        "ratio": {str(n): ratios(r["count"][n]) for n in sizes},
                        "cheapest": {str(n): min(r["count"][n], key=r["count"][n].get) for n in sizes}})
    result = {"record": cfg["record"], "seeds": seeds, "sizes": sizes, "record_fit_sizes": rec["fit_sizes"],
              "runs": [str(d.relative_to(here)) for d in run_dirs], "windows": windows}
    registry = formula_registry()
    names = cfg.get("names") or {k: registry[k].name for k in schemes}
    md, tex = tables(result, schemes, names, cfg["order_by"])
    head = (f"# Extrapolated ranking vs fit window\n\nWritten by `python -m pysimcount.pffitwindow`. Record "
            f"`{cfg['record']}` (its figures fit n = {rec['fit_sizes']}); seeds {seeds}; mean measured r over seeds, "
            "power law fitted per window, count = exact per-step count x ceil(a n^b) (pfchilds's rule; the record's "
            "own window reproduces its values exactly). Rows ordered by the fit-window column "
            f"`{cfg['order_by']}`; bold = predicted cheapest.\n\n")
    out = here / cfg["out"]
    out.with_suffix(".md").write_text(head + md)
    out.with_suffix(".tex").write_text(f"% Generated by pysimcount.pffitwindow from {cfg['record']}; "
                                       "do not edit, regenerate instead.\n" + tex)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    print(f"  wrote {out.with_suffix('.md')}, .tex, .json")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    if (args.config.parent / cfg["out"]).with_suffix(".json").resolve() == args.config.resolve():
        raise ValueError(f"out {cfg['out']!r} would overwrite the config {args.config}")
    run(cfg, args.config.parent)


if __name__ == "__main__":
    main()
