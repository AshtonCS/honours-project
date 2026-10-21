"""pgfplots figures from a ``pfcompare`` run, in the style of Childs et al. (2018).

    python -m plsimcount.pffigure results/pf-thesis/results.json --render

Writes every standard figure of the run, for each of the three gate metrics
(CNOT, T, total), to ``<results dir>/figures/<merged|unmerged>/<metric>/``:

* ``A-<method>-by-order.tex``  one way of getting r (ana, min, com, emp, tte,
  measured), Suzuki formulas of every order -- the layout of the 2018 paper;
* ``B-order<p>-by-formula.tex`` one order, every formula at measured r, and
  ``B-all-by-formula.tex`` with every formula;
* ``C-<scheme>-by-bound.tex``  one Suzuki formula, every way of getting r --
  how far each bound is from the measured r.

Each file is one self-contained tikzpicture with the coordinates inline (paste
into Overleaf; needs only ``\\usepackage{pgfplots}``). ``all.tex`` puts every
figure on its own page for checking, and ``INDEX.md`` lists them; ``--render``
compiles ``all.tex`` to ``all.pdf`` with tectonic.

The figure list is not hard-coded: it is derived from the rows present, so a
formula added to ``pfcompare.formula_registry`` and to the run's config
appears in every figure it belongs to on the next run. Its style follows
from its family (colour) and order (marker); two formulas that would look the
same in one figure raise an error naming the formula to add to
``COLOUR_OVERRIDE``.

Filled markers: r measured, or computed from a bound. Open markers: r from
the power-law fit (``r_source == "fit"``), i.e. extrapolations.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# metric -> (results column, gatebase, y-axis label)
# ASSUMPTION: "total" uses the Clifford+Rz basis (cz), as in Childs et al.
# 2018's total-gate-count figure; gates_total excludes simcount's omega
# (global phase) markers. CNOT also uses cz (eps = 1e-3 all to Trotter);
# T uses ct (5e-4 Trotter + 5e-4 synthesis). (unconfirmed)
METRICS: dict[str, tuple[str, str, str]] = {
    "cnot": ("cnot_total", "cz", "CNOT count"),
    "t": ("t_total", "ct", "$T$ count"),
    "total": ("gates_total", "cz", "Total gate count (Clifford+$R_z$)"),
}

# Ways of getting r, in legend order. "measured" also covers the "fit" rows.
METHODS = ["ana", "min", "com", "emp", "tte", "measured"]
METHOD_STYLE: dict[str, tuple[str, str, str]] = {  # method -> (label, colour, mark)
    "ana": ("Analytic bound", "gray", "square*"),
    "min": ("Minimised bound", "brown", "triangle*"),
    "com": ("Commutator bound (2018)", "orange", "diamond*"),
    "emp": ("Empirical fit (2018)", "teal", "pentagon*"),
    "tte": ("Commutator bound (2021)", "blue", "oplus*"),
    # measured needs a fillable mark: its fitted points are drawn open
    "measured": ("Measured $r$", "red", "*"),
}

FAMILY_COLOUR = {"suzuki": "red", "paper": "blue", "historical": "green!50!black"}
ORDER_MARK = {1: "*", 2: "triangle*", 4: "square*", 6: "pentagon*", 8: "diamond*"}
# Formulas that share a family and an order need their own colour.
COLOUR_OVERRIDE = {"morales-y8m10": "orange", "morales-m8": "violet"}


def latex_escape(text: str) -> str:
    for ch in "&%#_":
        text = text.replace(ch, "\\" + ch)
    return text


@dataclass
class Series:
    label: str
    colour: str
    mark: str
    measured: list[tuple[int, float]] = field(default_factory=list)  # filled markers
    fit: list[tuple[int, float]] = field(default_factory=list)       # open markers


@dataclass
class FigureSpec:
    name: str
    description: str
    members: list[tuple[str, str]]  # (scheme, method)
    legend_by: str                  # "scheme" or "method"


def collect(rows: list[dict], column: str, gatebase: str, merged: bool,
            instance: str = "simcount") -> dict[tuple[str, str], dict[str, list[tuple[int, float]]]]:
    """``{(scheme, method): {"measured": [(n, y)], "fit": [(n, y)]}}`` for one
    gate basis and merge setting. Raises on two rows for one point."""
    out: dict[tuple[str, str], dict[str, list[tuple[int, float]]]] = {}
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        if row["instance"] != instance or row["gatebase"] != gatebase or bool(row["merged"]) != merged:
            continue
        source = row["r_source"]
        method = "measured" if source in ("measured", "fit") else source
        if method not in METHODS:
            raise ValueError(f"unknown r_source {source!r}")
        part = "fit" if source == "fit" else "measured"
        n = int(row["n"])
        if (row["scheme"], method, n) in seen:
            raise ValueError(f"two rows for {row['scheme']} / {method} at n={n}")
        seen.add((row["scheme"], method, n))
        out.setdefault((row["scheme"], method), {"measured": [], "fit": []})[part].append((n, float(row[column])))
    for parts in out.values():
        parts["measured"].sort()
        parts["fit"].sort()
    return out


def standard_figures(keys: set[tuple[str, str]], registry: dict) -> list[FigureSpec]:
    """Every figure the rows support (see module docstring). Formulas appear
    in registry order, methods in METHODS order."""
    for scheme, _ in keys:
        if scheme not in registry:
            raise ValueError(f"scheme {scheme!r} is not in pfcompare.formula_registry()")
    schemes = [s for s in registry if any(k[0] == s for k in keys)]
    suzuki = [s for s in schemes if registry[s].family == "suzuki"]
    figs: list[FigureSpec] = []

    for method in METHODS:
        members = [(s, method) for s in suzuki if (s, method) in keys]
        if len(members) >= 2:
            figs.append(FigureSpec(f"A-{method}-by-order",
                                   f"Suzuki formulas of each order, r from: {METHOD_STYLE[method][0]}",
                                   members, "scheme"))

    measured = [s for s in schemes if (s, "measured") in keys]
    for order in sorted({registry[s].order for s in measured}):
        members = [(s, "measured") for s in measured if registry[s].order == order]
        if len(members) >= 2:
            figs.append(FigureSpec(f"B-order{order}-by-formula",
                                   f"Every order-{order} formula at measured r", members, "scheme"))
    if len(measured) >= 2:
        figs.append(FigureSpec("B-all-by-formula", "Every formula at measured r",
                               [(s, "measured") for s in measured], "scheme"))

    for s in suzuki:
        members = [(s, m) for m in METHODS if (s, m) in keys]
        if len(members) >= 2:
            figs.append(FigureSpec(f"C-{s}-by-bound", f"{registry[s].name}: r from each bound and measured",
                                   members, "method"))
    return figs


def series_for(spec: FigureSpec, data: dict, registry: dict) -> list[Series]:
    """Styled series of one figure; raises if two series would look the same."""
    out: list[Series] = []
    looks: dict[tuple[str, str], str] = {}
    for scheme, method in spec.members:
        if spec.legend_by == "method":
            label, colour, mark = METHOD_STYLE[method]
        else:
            f = registry[scheme]
            if f.order not in ORDER_MARK:
                raise ValueError(f"no marker for order {f.order}; add it to ORDER_MARK")
            if scheme not in COLOUR_OVERRIDE and f.family not in FAMILY_COLOUR:
                raise ValueError(f"no colour for family {f.family!r}; add it to FAMILY_COLOUR")
            label = latex_escape(f.name)
            colour = COLOUR_OVERRIDE.get(scheme, FAMILY_COLOUR.get(f.family))
            mark = ORDER_MARK[f.order]
        if (colour, mark) in looks:
            raise ValueError(f"{scheme!r} would look identical to {looks[(colour, mark)]!r} in {spec.name}; "
                             f"add one of them to COLOUR_OVERRIDE")
        looks[(colour, mark)] = scheme
        parts = data[(scheme, method)]
        out.append(Series(label, colour, mark, list(parts["measured"]), list(parts["fit"])))
    return out


def _coordinates(points: list[tuple[int, float]]) -> str:
    return "\n".join(f"      ({n},{y:.0f})" for n, y in points)


def figure_tex(series: list[Series], ylabel: str, header: list[str]) -> str:
    sizes = sorted({n for s in series for n, _ in s.measured + s.fit})
    ticks = ",".join(str(n) for n in sizes)
    out = [f"% {line}" for line in header]
    out += [
        "\\begin{tikzpicture}",
        "  \\begin{loglogaxis}[",
        "    width=10cm,",
        "    xlabel={System size $n$},",
        f"    ylabel={{{ylabel}}},",
        f"    xtick={{{ticks}}},",
        f"    xticklabels={{{ticks}}},",
        "    ymajorgrids=true,",
        "    legend pos=outer north east,",
        "    legend cell align=left,",
        "    legend style={font=\\footnotesize},",
        "  ]",
    ]
    for s in series:
        if s.measured:
            out += [
                f"    \\addplot[only marks, {s.colour}, mark={s.mark}, mark size=2.4pt, mark options={{solid, fill={s.colour}}}] coordinates {{",
                _coordinates(s.measured),
                "    };",
                f"    \\addlegendentry{{{s.label}}}",
            ]
        if s.fit:
            out += [
                "    % r from power-law fit (extrapolation); one legend entry per series",
                f"    \\addplot[only marks, {s.colour}, mark={s.mark}, mark size=2.4pt, mark options={{solid, fill=white}}"
                + (", forget plot" if s.measured else "") + "] coordinates {",
                _coordinates(s.fit),
                "    };",
            ]
            if not s.measured:
                out.append(f"    \\addlegendentry{{{s.label}}}")
    if any(s.fit for s in series):
        out += [
            "    \\addlegendimage{only marks, black, mark=*, mark options={fill=black}}",
            "    \\addlegendentry{computed $r$}",
            "    \\addlegendimage{only marks, black, mark=*, mark options={fill=white}}",
            "    \\addlegendentry{fitted $r$ (extrapolated)}",
        ]
    out += ["  \\end{loglogaxis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def header_lines(results: Path, meta: dict, spec: FigureSpec, metric: str, gatebase: str, merged: bool) -> list[str]:
    cfg, prov = meta["config"], meta["provenance"]
    lines = [
        "Generated by plsimcount.pffigure -- do not edit; regenerate instead.",
        f"figure: {spec.name} -- {spec.description}",
        f"source: {results}",
        f"metric={metric} gatebase={gatebase} {'merged' if merged else 'unmerged'}",
        f"seed={cfg['seed']} t=n template={cfg.get('template')} synth={cfg.get('synth_backend')}",
        f"git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}",
        f"source_sha256={prov['source_sha256']}",
    ]
    if prov["git_dirty"]:
        lines.append("WARNING: the run was made from uncommitted code; the commit above does not reproduce it.")
    return lines


PREAMBLE = """\\documentclass{article}
\\usepackage[a4paper,landscape,margin=1.5cm]{geometry}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}
\\pagestyle{empty}
\\begin{document}
"""


def write_all(results: Path, merged: bool = True, instance: str = "simcount") -> Path:
    """Write every standard figure for every metric; returns the output directory."""
    from .pfcompare import formula_registry

    data_json = json.loads(results.read_text())
    rows, meta = data_json["rows"], data_json["meta"]
    registry = formula_registry()
    root = results.parent / "figures" / ("merged" if merged else "unmerged")
    pages, index = [], ["# Figures", "", f"Generated from `{results}` by `plsimcount.pffigure`.", ""]

    for metric, (column, gatebase, ylabel) in METRICS.items():
        data = collect(rows, column, gatebase, merged, instance)
        if not data:
            raise ValueError(f"no {gatebase} rows for metric {metric}")
        (root / metric).mkdir(parents=True, exist_ok=True)
        index += [f"## {metric} ({gatebase})", ""]
        for spec in standard_figures(set(data), registry):
            tex = figure_tex(series_for(spec, data, registry), ylabel,
                             header_lines(results, meta, spec, metric, gatebase, merged))
            rel = f"{metric}/{spec.name}.tex"
            (root / rel).write_text(tex)
            index.append(f"- `{rel}` -- {spec.description}")
            pages.append(f"\\section*{{\\texttt{{{rel}}}}}\n{latex_escape(spec.description)}\\par\\medskip\n"
                         f"\\input{{{rel}}}\n\\clearpage\n")
        index.append("")

    (root / "all.tex").write_text(PREAMBLE + "\n".join(pages) + "\\end{document}\n")
    (root / "INDEX.md").write_text("\n".join(index))
    return root


def render(root: Path) -> Path:
    """Compile all.tex with tectonic. Raises if tectonic is missing or fails."""
    exe = shutil.which("tectonic")
    if exe is None:
        raise RuntimeError("tectonic not found on PATH; install it or drop --render")
    subprocess.run([exe, "all.tex"], cwd=root, check=True)
    return root / "all.pdf"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="results.json written by pfcompare")
    ap.add_argument("--unmerged", action="store_true", help="unmerged counts (default: merged)")
    ap.add_argument("--instance", default="simcount")
    ap.add_argument("--render", action="store_true", help="compile all.pdf with tectonic")
    args = ap.parse_args(argv)
    root = write_all(args.results, merged=not args.unmerged, instance=args.instance)
    print((root / "INDEX.md").read_text())
    if args.render:
        print(f"rendered {render(root)}")


if __name__ == "__main__":
    main()
