"""Measured Trotter number r against system size n, X-Y-Z vs two-group grouping.

    python -m pysimcount.pfgroupfigure results/np-grouping/aggregate.json --render

Reads the same seed runs as ``pfgroupseeds`` (the config's ``runs`` list;
``pfseeds.load_runs`` refuses runs made from uncommitted code or whose configs
differ in more than the seed). For each formula, grouping and n it takes the
measured r on every seed and plots the median, with error bars from the
minimum to the maximum over seeds. r is the same for merged and unmerged rows;
the merged row is read.

Writes into ``<config dir>/<out>/figures/``:

* ``r-vs-n-<variant>.tex``: one self-contained tikzpicture (needs only
  ``\\usepackage{pgfplots}``), coordinates inline;
* ``all.tex`` (one page per figure) and, with ``--render``, ``all.pdf``.

Style: one colour per formula; X-Y-Z is a solid line with filled markers, the
two-group variant a dashed line with open markers. Deviation from
``pffigure``: there, colour follows the family and marker the order, which
makes S2 and S4 the same colour; here the plotted formulas are few and must be
told apart at slide size, so each gets its own colour from ``PALETTE``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .pffigure import PREAMBLE, latex_escape, render
from .pfgroupseeds import REFERENCE_VARIANT, _index
from .pfseeds import load_runs

# Lie-Trotter is left out by default: its r is ~10^5 at n = 10 and stretches
# the log axis so far that the other formulas' grouping difference vanishes.
DEFAULT_SCHEMES = ["suzuki2", "suzuki4", "mo-q14"]
DEFAULT_VARIANT = "twogroup-even-out"
PALETTE = ["red", "blue", "green!50!black", "orange", "violet", "teal", "brown", "gray"]
MARKS = ["*", "square*", "triangle*", "diamond*", "pentagon*"]
VARIANT_LABEL = {"xyz": "X-Y-Z grouping",
                 "twogroup-even-out": "two-group ($E'$ outer)",
                 "twogroup-odd-out": "two-group ($O$ outer)"}
# Horizontal offset so the two groupings' error bars at one n do not overlap:
# X-Y-Z is drawn at n / (1 + DODGE), two-group at n * (1 + DODGE). The tick
# labels stay at the true n. Purely visual; the coordinates in the .tex file
# carry the dodged x, and a comment states it.
DODGE = 0.025


def r_stats(runs: dict[int, dict], schemes: list[str], variants: list[str]) -> dict:
    """``{(scheme, variant): [(n, median, min, max)]}`` of measured r over seeds.
    Raises if any seed lacks a (scheme, n, variant) row that another seed has."""
    seeds = sorted(runs)
    idx = {s: _index(runs[s]) for s in seeds}
    keys = sorted(idx[seeds[0]])
    for s in seeds[1:]:
        if sorted(idx[s]) != keys:
            raise ValueError(f"seed {s} has a different set of (scheme, n, variant) rows from seed {seeds[0]}")
    out: dict[tuple[str, str], list[tuple[int, float, float, float]]] = {}
    for scheme in schemes:
        for variant in variants:
            sizes = sorted(n for (sc, n, v) in keys if sc == scheme and v == variant)
            if not sizes:
                raise ValueError(f"no rows for {scheme} / {variant}")
            points = []
            for n in sizes:
                rs = [idx[s][(scheme, n, variant)]["r"] for s in seeds]
                points.append((n, statistics.median(rs), min(rs), max(rs)))
            out[(scheme, variant)] = points
    return out


def figure_tex(stats: dict, schemes: list[str], variant: str, names: dict[str, str],
               header: list[str]) -> str:
    """One tikzpicture: per formula, an X-Y-Z line and a two-group line."""
    if len(schemes) > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} formulas per figure")
    sizes = sorted({p[0] for pts in stats.values() for p in pts})
    ticks = ",".join(str(n) for n in sizes)
    out = [f"% {line}" for line in header]
    out += [
        f"% x is dodged for readability: X-Y-Z at n/{1 + DODGE}, two-group at n*{1 + DODGE}.",
        "\\begin{tikzpicture}",
        "  \\begin{loglogaxis}[",
        "    width=10cm,",
        "    xlabel={System size $n$},",
        "    ylabel={Measured Trotter number $r$},",
        f"    xtick={{{ticks}}},",
        f"    xticklabels={{{ticks}}},",
        f"    xmin={sizes[0] * 0.9:g}, xmax={sizes[-1] * 1.1:g},",
        "    ymajorgrids=true,",
        "    legend pos=outer north east,",
        "    legend cell align=left,",
        "    legend style={font=\\footnotesize},",
        "  ]",
    ]
    for i, scheme in enumerate(schemes):
        colour, mark = PALETTE[i], MARKS[i % len(MARKS)]
        for v in (REFERENCE_VARIANT, variant):
            two_group = v != REFERENCE_VARIANT
            fill = "white" if two_group else colour
            dash = "dashed" if two_group else "solid"
            scale = (1 + DODGE) if two_group else 1 / (1 + DODGE)
            # the two-group line shares the formula's legend entry (forget plot);
            # the line styles get their own black entries below
            extra = ", forget plot" if two_group else ""
            out.append(f"    % {scheme}, {v}: (n, median r) with error bars to min and max over seeds")
            out.append(f"    \\addplot[{colour}, {dash}, thick, mark={mark}, mark size=2.4pt, "
                       f"mark options={{solid, fill={fill}}}{extra}, "
                       "error bars/.cd, y dir=both, y explicit, error bar style={solid}] coordinates {")
            for n, med, lo, hi in stats[(scheme, v)]:
                out.append(f"      ({n * scale:.4f},{med:g}) += (0,{hi - med:g}) -= (0,{med - lo:g})")
            out.append("    };")
            if not two_group:
                out.append(f"    \\addlegendentry{{{latex_escape(names[scheme])}}}")
    out += [
        "    \\addlegendimage{black, solid, thick, mark=*, mark options={solid, fill=black}}",
        f"    \\addlegendentry{{{VARIANT_LABEL[REFERENCE_VARIANT]}}}",
        "    \\addlegendimage{black, dashed, thick, mark=*, mark options={solid, fill=white}}",
        f"    \\addlegendentry{{{VARIANT_LABEL[variant]}}}",
        "  \\end{loglogaxis}",
        "\\end{tikzpicture}",
        "",
    ]
    return "\n".join(out)


def header_lines(config: Path, runs: dict[int, dict], schemes: list[str], variant: str) -> list[str]:
    from .pfcompare import provenance

    prov = provenance()
    seeds = sorted(runs)
    commits = sorted({runs[s]["meta"]["provenance"]["git_commit"] for s in seeds})
    lines = [
        "Generated by pysimcount.pfgroupfigure -- do not edit; regenerate instead.",
        f"source: {config} (seeds {seeds}); run commits: {', '.join(commits)}",
        f"formulas: {', '.join(schemes)}; groupings: {REFERENCE_VARIANT} vs {variant}",
        "y: measured r, median over seeds; error bars: min and max over seeds",
        f"figure made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}",
    ]
    if prov["git_dirty"]:
        lines.append("WARNING: the figure was made from uncommitted code; the commit above does not reproduce it.")
    return lines


def main(argv: list[str] | None = None) -> None:
    from .pfcompare import formula_registry

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="the pfgroupseeds aggregate config (runs + out)")
    ap.add_argument("--schemes", nargs="+", default=DEFAULT_SCHEMES)
    ap.add_argument("--variant", default=DEFAULT_VARIANT, choices=[v for v in VARIANT_LABEL if v != REFERENCE_VARIANT])
    ap.add_argument("--render", action="store_true", help="compile all.pdf with tectonic")
    args = ap.parse_args(argv)

    cfg = json.loads(args.config.read_text())
    here = args.config.parent
    runs = load_runs([here / p for p in cfg["runs"]])
    registry = formula_registry()
    for s in args.schemes:
        if s not in registry:
            raise ValueError(f"unknown scheme {s!r}")
    names = {s: registry[s].name for s in args.schemes}
    stats = r_stats(runs, args.schemes, [REFERENCE_VARIANT, args.variant])

    root = here / cfg["out"] / "figures"
    root.mkdir(parents=True, exist_ok=True)
    name = f"r-vs-n-{args.variant}.tex"
    (root / name).write_text(figure_tex(stats, args.schemes, args.variant, names,
                                        header_lines(args.config, runs, args.schemes, args.variant)))
    page = (f"\\section*{{\\texttt{{{latex_escape(name)}}}}}\n"
            f"Measured $r$ vs $n$, {latex_escape(VARIANT_LABEL[REFERENCE_VARIANT])} vs "
            f"{VARIANT_LABEL[args.variant]}; median over seeds, bars min--max.\\par\\medskip\n"
            f"\\input{{{name}}}\n")
    (root / "all.tex").write_text(PREAMBLE + page + "\\end{document}\n")
    print(f"wrote {root / name}")
    if args.render:
        print(f"rendered {render(root)}")


if __name__ == "__main__":
    main()
