"""How loose each Trotter-number bound is, against n: bound r / measured r.

    python -m pysimcount.pfboundratio results/np-bounds-ratio/config.json [--render]

For RQ2. On each seed (Hamiltonian instance) and size, the ratio is

    ratio_s(n) = r_bound,s(n) / r_measured,s(n)

with r_measured the fastempirical window rule at eps = 1e-3, t = n (cz rows,
``r_source == "measured"``). The pairing is ``pfseeds.bound_ratios``, the same
function behind np-main/aggregate/bounds.md and np-mean/main-bounds.md, so the
numbers here equal those tables at their sizes. Plotted value = geometric mean
over seeds, bars = min-max over seeds (the summary chosen on 2026-10-27; see
``pfmean`` for why not the arithmetic mean of ratios).

Only *measured* denominators are used: np-main's n = 13 and 20 rows divide by
a fitted r (an extrapolation), so they are dropped rather than drawn as if
they were the same kind of point.

Run sets. The sizes come from several run directories (np-main: n = 4-8, 10;
np-main-n9; np-main-n11). ``pfseeds.load_runs`` refuses a run made from
uncommitted code, and the n = 9 and 11 runs were (git_dirty at e42ff29).
A run set may therefore name ``accept_dirty`` in the config: a written reason,
copied into every figure header. A dirty run set without one raises.

What it writes (into the config's ``out`` directory):

* ``figures/bound-ratio-<scheme>.tex``, one per configured scheme;
* ``figures/all.tex`` (``--render`` compiles it with tectonic);
* ``table.md`` and ``results.json`` (every per-seed ratio), ``meta.json``,
  ``environment.txt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import pffigure, pfseeds
from .pfcompare import formula_registry, provenance, write_environment
from .pfmean import summarise

BOUNDS = ("ana", "min", "com", "tte", "emp")  # pysimcount's r sources other than measured


def load_set(paths: list[Path], accept_dirty: str | None) -> dict[int, dict]:
    """{seed: results.json} for one run set, with ``pfseeds.load_runs``'s checks
    (one run per seed, configs equal but for seed/name/description). A run made
    from uncommitted code raises unless ``accept_dirty`` gives a reason."""
    runs: dict[int, dict] = {}
    base = None
    for p in paths:
        d = json.loads(Path(p).read_text())
        cfg = dict(d["meta"]["config"])
        seed = int(cfg.pop("seed"))
        for k in ("name", "description"):
            cfg.pop(k, None)
        if seed in runs:
            raise ValueError(f"two runs for seed {seed}")
        if base is None:
            base = cfg
        elif cfg != base:
            raise ValueError(f"{p}: config differs from the first run in more than the seed")
        if d["meta"]["provenance"].get("git_dirty") and not accept_dirty:
            raise ValueError(f"{p} was made from uncommitted code (git_dirty) and its run set gives no accept_dirty")
        runs[seed] = d
    return runs


def ratios(sets: dict[str, dict[int, dict]], schemes: list[str], bounds: list[str],
           seeds: list[int]) -> list[dict]:
    """Per (scheme, bound, n) over every run set: per-seed bound / measured r
    and its geometric-mean summary. Raises if a size appears in two run sets
    or a run set lacks one of the seeds."""
    out, seen = [], {}
    for label, runs in sets.items():
        if sorted(runs) != sorted(seeds):
            raise ValueError(f"run set {label}: seeds {sorted(runs)}, expected {sorted(seeds)}")
        for b in pfseeds.bound_ratios(runs):
            if b["denominator"] != "measured" or b["scheme"] not in schemes or b["bound"] not in bounds:
                continue
            key = (b["scheme"], b["bound"], b["n"])
            if key in seen:
                raise ValueError(f"{key} is in run sets {seen[key]} and {label}")
            seen[key] = label
            out.append({"scheme": b["scheme"], "bound": b["bound"], "n": b["n"], "run_set": label,
                        "per_seed": b["per_seed"], **summarise(b["per_seed"], "ratio")})
    out.sort(key=lambda x: (schemes.index(x["scheme"]), bounds.index(x["bound"]), x["n"]))
    return out


def table_md(recs: list[dict], scheme: str, bounds: list[str]) -> str:
    sizes = sorted({x["n"] for x in recs if x["scheme"] == scheme})
    cell = {(x["bound"], x["n"]): x for x in recs if x["scheme"] == scheme}
    lines = ["| bound | " + " | ".join(f"n = {n}" for n in sizes) + " |",
             "|---|" + "---:|" * len(sizes)]
    for b in bounds:
        row = []
        for n in sizes:
            x = cell.get((b, n))
            row.append(f"{x['centre']:,.3g} ({x['min']:,.3g}-{x['max']:,.3g})" if x else "--")
        lines.append(f"| {pffigure.METHOD_STYLE[b][0]} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def figure_tex(recs: list[dict], scheme: str, bounds: list[str], header: list[str]) -> str:
    """Log-y, linear-x: one line per bound (geometric mean, min-max bars), and
    a dashed line at 1 (the bound would give exactly the measured r)."""
    pts = [x for x in recs if x["scheme"] == scheme]
    if not pts:
        raise ValueError(f"no ratios for {scheme}")
    sizes = sorted({x["n"] for x in pts})
    out = [f"% {h}" for h in header] + [
        "\\begin{tikzpicture}", "  \\begin{semilogyaxis}[", "    width=10cm, height=8.3cm,",
        "    xlabel={System size $n$}, ylabel={bound $r$ / measured $r$},",
        f"    xmin={sizes[0] - 0.5:g}, xmax={sizes[-1] + 0.5:g},",
        "    xtick={" + ",".join(str(n) for n in sizes) + "},",
        "    ymajorgrids=true, yminorgrids=true,", "    legend pos=outer north east, legend cell align=left,",
        "    legend style={font=\\footnotesize},", "  ]",
        f"    \\addplot[dashed, black, domain={sizes[0] - 0.5:g}:{sizes[-1] + 0.5:g}, samples=2, forget plot] {{1}};"]
    for b in bounds:
        ser = sorted((x["n"], x["centre"], x["min"], x["max"]) for x in pts if x["bound"] == b)
        if not ser:
            continue
        label, colour, mark = pffigure.METHOD_STYLE[b]
        out.append(f"    \\addplot[solid, {colour}, mark={mark}, mark size=2pt, "
                   "error bars/.cd, y dir=both, y explicit] coordinates {")
        out += [f"      ({n},{c:.6g}) -= (0,{c - lo:.6g}) += (0,{hi - c:.6g})" for n, c, lo, hi in ser]
        out += ["    };", f"    \\addlegendentry{{{pffigure.latex_escape(label)}}}"]
    out += ["    \\addlegendimage{dashed, black}", "    \\addlegendentry{bound $=$ measured $r$}",
            "  \\end{semilogyaxis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    here = args.config.parent
    out = here / cfg["out"]
    seeds, schemes, bounds = cfg["seeds"], cfg["schemes"], cfg["bounds"]
    bad = [b for b in bounds if b not in BOUNDS] + [s for s in schemes if s not in formula_registry()]
    if bad:
        raise ValueError(f"unknown bounds or schemes: {bad}")

    sets, dirty = {}, {}
    for rs in cfg["run_sets"]:
        paths = [(here / rs["runs"].format(seed=s)).resolve() for s in seeds]
        sets[rs["label"]] = load_set(paths, rs.get("accept_dirty"))
        if any(r["meta"]["provenance"].get("git_dirty") for r in sets[rs["label"]].values()):
            dirty[rs["label"]] = rs["accept_dirty"]
    recs = ratios(sets, schemes, bounds, seeds)

    out.mkdir(parents=True, exist_ok=True)
    prov = provenance()
    meta = {"config": cfg, "provenance": prov, "dirty_run_sets": dirty,
            "runs": {label: {s: r["meta"]["provenance"] for s, r in runs.items()} for label, runs in sets.items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    (out / "results.json").write_text(json.dumps({"meta": meta, "ratios": recs}, indent=1) + "\n")
    write_environment(out)

    reg = formula_registry()
    sizes_by_set = {label: sorted({x["n"] for x in recs if x["run_set"] == label}) for label in sets}
    md = [f"# Bound r / measured r: geometric mean (min-max) over seeds {seeds}, cz, eps = 1e-3, t = n\n",
          f"Written by `python -m pysimcount.pfboundratio {args.config}`. Pairing = `pfseeds.bound_ratios`; "
          "only measured denominators.\n",
          "Sizes by run set: " + "; ".join(f"{k} n = {v}" for k, v in sizes_by_set.items()) + "\n"]
    md += [f"Run set {k} was made from uncommitted code (git_dirty), accepted because: {v}\n" for k, v in dirty.items()]
    for s in schemes:
        md += [f"## {reg[s].name}\n", table_md(recs, s, bounds), ""]
    (out / "table.md").write_text("\n".join(md))

    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    names = []
    for s in schemes:
        name = f"bound-ratio-{s}"
        header = ["Generated by pysimcount.pfboundratio -- do not edit; regenerate instead.",
                  f"figure: {name} -- {reg[s].name}: Trotter number from each bound / measured r, against n",
                  f"metric=r_ratio gatebase=cz; geometric mean over seeds {seeds}, bars min-max; "
                  "r_method=measured (denominator); bounds: "
                  + ", ".join(pffigure.METHOD_STYLE[b][0] for b in bounds),
                  "sizes: " + "; ".join(f"{k} n = {v}" for k, v in sizes_by_set.items()),
                  *[f"run set {k} made from uncommitted code (git_dirty), accepted: {v}" for k, v in dirty.items()],
                  f"formulas: {reg[s].name}",
                  f"figure made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}"]
        (figs / f"{name}.tex").write_text(figure_tex(recs, s, bounds, header))
        names.append(name)
    (figs / "all.tex").write_text(pffigure.PREAMBLE + "".join(
        f"\\section*{{\\texttt{{{n}.tex}}}}\n\\input{{{n}.tex}}\n\\clearpage\n" for n in names) + "\\end{document}\n")
    for s in schemes:
        cells = []
        for b in bounds:  # the commutator bounds exist for orders 1, 2, 4 only (bounds.commutator)
            v = {x["n"]: x["centre"] for x in recs if x["scheme"] == s and x["bound"] == b}
            if v:
                cells.append(f"{b} {v[min(v)]:.3g} (n={min(v)}) -> {v[max(v)]:.3g} (n={max(v)})")
        print(f"{s:12s} " + "   ".join(cells))
    if args.render:
        print(f"rendered {pffigure.render(figs)}")


if __name__ == "__main__":
    main()
