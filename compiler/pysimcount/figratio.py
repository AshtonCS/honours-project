"""Comparison figures from a pfchilds record: cost ratios, and one panel per family.

    python -m pysimcount.figratio results/np-childs-format/comparison-figures.json [--render]

Every formula in a pfchilds figure grows like n^(1 + b), b being the fitted
exponent of r, and that shared growth fills the y-axis. This module writes, for
one pfchilds record ``<id>.json`` and a chosen reference formula:

``<id>-ratio.tex``            cost of each formula / cost of the reference, against n
``<id>-families.tex``         absolute cost, one panel per family, other formulas in grey
``<id>-families-ratio.tex``   the same panels, y = the ratio
``<id>-ratio.json``           every ratio plotted, and the ratio of the plotted means

The ratios are paired: the two formulas are divided seed by seed, on the same
Hamiltonian instance. Plotted value = geometric mean over seeds, bars = min-max
over seeds (the summary chosen on 2026-10-27; see pfmean for why not the
arithmetic mean). Seed s contributes

    measured n      total_F,s(n) / total_ref,s(n)                  (from the runs)
    extrapolated n  c_F(n) ceil(a_F,s n^b_F,s)
                    / (c_ref(n) ceil(a_ref,s n^b_ref,s))           (per-seed fits, from <id>.json)

Consequences a reader has to know:

* A geometric mean of power laws is a power law, so apart from the ceilings each
  extrapolated ratio is a straight line on these axes with slope
  mean_s(b_F,s - b_ref,s). Every crossing at n > 10 is therefore decided by the
  exponents fitted on n = 5-10, not by anything observed at that n.
* The extrapolated value here is not the plotted value of the source figure over
  the reference's: the source fits the mean r over seeds once, this divides the
  per-seed fits. ``<id>-ratio.json`` holds both (``value``, ``ratio_of_plotted``).
* Min-max bars of a ratio are not the min-max bars of the source figure divided:
  a seed that is hard for both formulas cancels here.

Absolute panels draw the source record's values unchanged. Raises if a per-seed
value does not reproduce the record's mean, min or max, if any plotted value or
bar falls outside the configured y-range, or if the panels do not list each
formula exactly once. Only ``xyz`` and ``twogroup-*`` records: optimised steps
keep their per-seed totals elsewhere, and are refused rather than approximated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from . import pfchilds
from .figdrop import DERIVED_TAG
from .pfcompare import formula_registry, provenance

DIVIDER_N = 11.4  # between the last measured size (10) and the first extrapolated one (13)
GREY = "black!22"

Points = dict[str, list[dict]]  # {"measured": [{n, value, min, max}, ...], "extrapolated": [...]}


# ------------------------------------------------------------------ the numbers

def measured_per_seed(rec: dict, runs: dict[int, dict]) -> dict[str, dict[int, dict[int, int]]]:
    """{scheme: {n: {seed: total}}} from the runs. Raises unless the totals
    reproduce the mean, min and max stored in the record."""
    if rec["kind"] == "optimised":
        raise ValueError("optimised records are not supported: their totals are not in these rows")
    rows = pfchilds._measured_rows(rec["kind"], runs)
    column = pfchilds.METRICS[rec["metric"]][0]
    out: dict[str, dict[int, dict[int, int]]] = {}
    for scheme in rec["shown"]:
        out[scheme] = {}
        for p in rec["series"][scheme]["measured"]:
            n = p["n"]
            totals = {seed: int(row[column]) for seed, row in rows[(scheme, n)].items()}
            values = list(totals.values())
            if (min(values), max(values)) != (p["min"], p["max"]) or abs(sum(values) / len(values) - p["mean"]) > 1e-6:
                raise ValueError(f"{scheme} n={n}: runs do not reproduce the record's mean, min, max")
            out[scheme][n] = totals
    return out


def extrapolated_per_seed(rec: dict, scheme: str, n: int) -> dict[int, int]:
    """{seed: c(n) ceil(a_s n^b_s)}, pfchilds's rule on each seed's own fit."""
    s = rec["series"][scheme]
    model = s["per_step_model"]
    c = pfchilds.affine_eval(Fraction(model["alpha"]), Fraction(model["beta"]), n)
    return {int(seed): c * pfchilds.r_fit_ceil(a, b, n) for seed, (a, b) in s["per_seed_fit"].items()}


def paired_summary(numerator: dict[int, int], denominator: dict[int, int]) -> dict[str, float]:
    """Geometric mean, min and max over seeds of numerator / denominator.
    Raises unless both cover the same seeds."""
    if sorted(numerator) != sorted(denominator):
        raise ValueError(f"seeds differ: {sorted(numerator)} and {sorted(denominator)}")
    ratios = []
    for seed in sorted(numerator):
        ratios.append(numerator[seed] / denominator[seed])
    return {"value": pfchilds.geomean(ratios), "min": min(ratios), "max": max(ratios)}


def ratio_points(rec: dict, runs: dict[int, dict], reference: str) -> dict[str, Points]:
    """{scheme: paired ratios to ``reference``} for every shown formula."""
    if reference not in rec["shown"]:
        raise ValueError(f"reference {reference!r} is not in the figure ({rec['shown']})")
    per_seed = measured_per_seed(rec, runs)
    ref = rec["series"][reference]
    ref_plotted = {p["n"]: p["mean"] for p in ref["measured"]} | {p["n"]: p["value"] for p in ref["extrapolated"]}
    out: dict[str, Points] = {}
    for scheme in rec["shown"]:
        s = rec["series"][scheme]
        pts: Points = {"measured": [], "extrapolated": []}
        for p in s["measured"]:
            n = p["n"]
            pts["measured"].append({"n": n, **paired_summary(per_seed[scheme][n], per_seed[reference][n]),
                                    "ratio_of_plotted": p["mean"] / ref_plotted[n]})
        for p in s["extrapolated"]:
            n = p["n"]
            mine = extrapolated_per_seed(rec, scheme, n)
            if (min(mine.values()), max(mine.values())) != (p["min"], p["max"]):
                raise ValueError(f"{scheme} n={n}: per-seed fits do not reproduce the record's min, max")
            pts["extrapolated"].append({"n": n, **paired_summary(mine, extrapolated_per_seed(rec, reference, n)),
                                        "ratio_of_plotted": p["value"] / ref_plotted[n]})
        out[scheme] = pts
    return out


def absolute_points(rec: dict) -> dict[str, Points]:
    """The record's own plotted values, in the same form as ``ratio_points``."""
    out: dict[str, Points] = {}
    for scheme in rec["shown"]:
        s = rec["series"][scheme]
        out[scheme] = {
            "measured": [{"n": p["n"], "value": p["mean"], "min": p["min"], "max": p["max"]} for p in s["measured"]],
            "extrapolated": [{"n": p["n"], "value": p["value"], "min": p["min"], "max": p["max"]}
                             for p in s["extrapolated"]],
        }
    return out


def check_range(points: dict[str, Points], ymin: float, ymax: float) -> None:
    """Raise if any plotted value or bar is outside the y-range: a clipped bar
    would understate the spread without saying so."""
    for scheme, pts in points.items():
        for p in pts["measured"] + pts["extrapolated"]:
            if p["min"] < ymin or p["max"] > ymax:
                raise ValueError(f"{scheme} n={p['n']}: {p['min']:.4g}..{p['max']:.4g} outside y-range {ymin:g}..{ymax:g}")


def check_panels(panels: list[dict], shown: list[str], reference: str | None) -> None:
    """Each shown formula in exactly one panel; the reference (flat at 1) in none."""
    listed = [k for panel in panels for k in panel["schemes"]]
    expected = [k for k in shown if k != reference]
    if sorted(listed) != sorted(expected):
        raise ValueError(f"panels list {sorted(listed)}, expected each of {sorted(expected)} once")


# ------------------------------------------------------------------ tikz

def _plot(colour: str, mark: str, fill: str, line: str, mark_size: str, forget: bool, bars: bool) -> str:
    opts = f"{colour}, {line}, mark={mark}, mark size={mark_size}, mark options={{solid, fill={fill}}}"
    if forget:
        opts += ", forget plot"
    if bars:
        opts += ", error bars/.cd, y dir=both, y explicit, error bar style={solid, line width=0.3pt}"
    return f"    \\addplot[{opts}] coordinates {{"


def _coordinate(p: dict, bars: bool) -> str:
    if bars:
        return (f"      ({p['n']},{p['value']:.6g}) += (0,{p['max'] - p['value']:.6g}) "
                f"-= (0,{p['value'] - p['min']:.6g})")
    return f"      ({p['n']},{p['value']:.6g})"


def series_lines(pts: Points, label: str, colour: str, mark: str, mark_size: str,
                 bar_sizes: list[int] | None = None) -> list[str]:
    """Filled markers + solid line = measured, then open markers + dashed line =
    extrapolated, joined at the last measured point (pfchilds's convention).
    Bars at every n, or only at ``bar_sizes`` if given."""
    def coord(p: dict) -> str:
        return _coordinate(p, bar_sizes is None or p["n"] in bar_sizes)

    out = [_plot(colour, mark, colour, "solid", mark_size, False, True)]
    out += [coord(p) for p in pts["measured"]]
    out += ["    };", f"    \\addlegendentry{{{label}}}"]
    out.append(_plot(colour, mark, "white", "dashed", mark_size, True, True))
    out.append(_coordinate(pts["measured"][-1], False))
    out += [coord(p) for p in pts["extrapolated"]]
    out.append("    };")
    return out


def grey_lines(pts: Points) -> list[str]:
    out = [f"    \\addplot[{GREY}, solid, no markers, forget plot] coordinates {{"]
    out += [_coordinate(p, False) for p in pts["measured"] + pts["extrapolated"]]
    out.append("    };")
    return out


def _ticks(values: list[float]) -> str:
    text = ",".join(f"{v:g}" for v in values)
    return f"ytick={{{text}}}, yticklabels={{{text}}}, yminorticks=false,"


def _divider(ymin: float, ymax: float, labels: bool) -> list[str]:
    out = [f"    \\draw[black!35] (axis cs:{DIVIDER_N},{ymin:g}) -- (axis cs:{DIVIDER_N},{ymax:g});"]
    if labels:
        out += [f"    \\node[anchor=north east, font=\\scriptsize, black!60] at (axis cs:{DIVIDER_N},{ymax:g}) {{measured}};",
                f"    \\node[anchor=north west, font=\\scriptsize, black!60] at (axis cs:{DIVIDER_N},{ymax:g}) {{extrapolated}};"]
    return out


def header_lines(rec: dict, source: str, sha: str, what: str, extra: list[str]) -> list[str]:
    prov = provenance()
    out = ["% Generated by pysimcount.figratio -- do not edit; regenerate instead.",
           f"% figure: {rec['id']} -- {what}",
           f"% seeds {rec['seeds']}; metric={rec['metric']} kind={rec['kind']} merged (cz, simcount template)",
           f"% shown: {', '.join(rec['shown'])}",
           f"% removed (see {rec['id']}-ranking.md): {', '.join(rec['removed']) or 'none'}"]
    out += [f"% {line}" for line in extra]
    out.append(f"{DERIVED_TAG} {source} by pysimcount.figratio; source_sha256={sha}")
    out.append(f"% figure made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}")
    if prov["git_dirty"]:
        out.append("% WARNING: made from uncommitted code; the commit above does not reproduce it.")
    return out


def _fit_note(rec: dict) -> str:
    return f"$n = {min(rec['fit_sizes'])}$--${max(rec['fit_sizes'])}$"


def ratio_tex(rec: dict, points: dict[str, Points], reference: str, cfg: dict, header: list[str], registry: dict) -> str:
    """One axis: every formula's ratio to the reference, which is the line at 1."""
    check_range(points, cfg["ymin"], cfg["ymax"])
    ref_label, _, _ = pfchilds._style(reference, registry)
    out = header + ["\\begin{tikzpicture}", "  \\begin{loglogaxis}[",
                    "    scale only axis,",
                    f"    width={cfg['width_cm']}cm, height={cfg['height_cm']}cm,",
                    "    xlabel={System size $n$},",
                    f"    ylabel={{{cfg['ylabel']}}},",
                    "    xtick={5,10,30,100,300}, xticklabels={5,10,30,100,300},",
                    f"    {_ticks(cfg['ytick'])}",
                    f"    xmin={cfg['xmin']:g}, xmax={cfg['xmax']:g}, ymin={cfg['ymin']:g}, ymax={cfg['ymax']:g},",
                    "    ymajorgrids=true, grid style={black!12},",
                    "    legend style={at={(0.5,-0.13)}, anchor=north, draw=none, font=\\footnotesize,",
                    f"      legend columns={cfg['legend_columns']}, /tikz/every even column/.append style={{column sep=8pt}}}},",
                    "    legend cell align=left,",
                    "  ]"]
    out += _divider(cfg["ymin"], cfg["ymax"], True)
    out += [f"    \\addplot[black, solid, line width=0.9pt, no markers] coordinates {{({cfg['xmin']:g},1) ({cfg['xmax']:g},1)}};",
            f"    \\addlegendentry{{{ref_label} (reference)}}"]
    for scheme in rec["shown"]:
        if scheme == reference:
            continue
        label, colour, mark = pfchilds._style(scheme, registry)
        out += series_lines(points[scheme], label, colour, mark, cfg["mark_size"], cfg["bar_sizes"])
    out += ["    \\addlegendimage{black, solid, mark=*, mark options={solid, fill=black}}",
            "    \\addlegendentry{measured $r$}",
            "    \\addlegendimage{black, dashed, mark=*, mark options={solid, fill=white}}",
            f"    \\addlegendentry{{extrapolated: per-seed power-law fits of $r$ ({_fit_note(rec)})}}",
            # a blank first cell puts the note in the second legend column, which is already this wide
            "    \\addlegendimage{empty legend}",
            "    \\addlegendentry{}",
            "    \\addlegendimage{empty legend}",
            f"    \\addlegendentry{{{cfg['bar_note']}}}",
            "  \\end{loglogaxis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def panels_tex(rec: dict, points: dict[str, Points], reference: str | None, cfg: dict, header: list[str], registry: dict) -> str:
    """One axis per family on identical scales. Each panel draws the formulas of
    the other panels in grey first, then its own in colour with their bars."""
    check_range(points, cfg["ymin"], cfg["ymax"])
    panels = cfg["panels"]
    check_panels(panels, rec["shown"], reference)
    columns = cfg["columns"]
    w, h = cfg["width_cm"], cfg["height_cm"]
    step_x, step_y = w + cfg["gap_x_cm"], h + cfg["gap_y_cm"]
    rows = (len(panels) + columns - 1) // columns
    out = header + ["\\begin{tikzpicture}"]
    for i, panel in enumerate(panels):
        col, row = i % columns, i // columns
        first_col, last_row = col == 0, i + columns >= len(panels)
        opts = ["    scale only axis,",
                f"    at={{({col * step_x}cm,{-row * step_y}cm)}}, anchor=south west,",
                f"    width={w}cm, height={h}cm,",
                f"    title={{{panel['title']}}}, title style={{font=\\small, yshift=-2pt}},",
                "    xtick={5,10,30,100,300}, xticklabels={5,10,30,100,300},",
                "    tick label style={font=\\footnotesize},",
                f"    xmin={cfg['xmin']:g}, xmax={cfg['xmax']:g}, ymin={cfg['ymin']:g}, ymax={cfg['ymax']:g},",
                "    ymajorgrids=true, grid style={black!12},",
                f"    legend pos={cfg['legend_pos']},",
                "    legend style={draw=none, fill=white, font=\\scriptsize, inner sep=1pt},",
                "    legend cell align=left,"]
        if cfg.get("ytick"):
            opts.append(f"    {_ticks(cfg['ytick'])}")
        if last_row:
            opts.append("    xlabel={System size $n$}, xlabel style={font=\\footnotesize},")
        else:
            opts.append("    xticklabels={},")
        if first_col:
            opts.append(f"    ylabel={{{cfg['ylabel']}}}, ylabel style={{font=\\footnotesize}},")
        else:
            opts.append("    yticklabels={},")
        out += ["  \\begin{loglogaxis}["] + opts + ["  ]"]
        out += _divider(cfg["ymin"], cfg["ymax"], False)
        for scheme in rec["shown"]:
            if scheme not in panel["schemes"] and scheme != reference:
                out += grey_lines(points[scheme])
        if reference is not None:
            out.append(f"    \\addplot[black, solid, line width=0.9pt, no markers, forget plot] "
                       f"coordinates {{({cfg['xmin']:g},1) ({cfg['xmax']:g},1)}};")
        for scheme in panel["schemes"]:
            label, colour, mark = pfchilds._style(scheme, registry)
            label = panel.get("labels", {}).get(scheme, label)  # a short key, so the legend clears the data
            out += series_lines(points[scheme], label, colour, mark, cfg["mark_size"])
        out.append("  \\end{loglogaxis}")
    width = columns * w + (columns - 1) * cfg["gap_x_cm"]
    y = -(rows - 1) * step_y - 1.2
    out.append(f"  \\node[anchor=north, text width={width}cm, align=flush center, font=\\footnotesize] "
               f"at ({width / 2}cm,{y}cm) {{{cfg['note'].replace('{fit}', _fit_note(rec))}}};")
    out += ["\\end{tikzpicture}", ""]
    return "\n".join(out)


# ------------------------------------------------------------------ driver

def run(cfg: dict, here: Path, render: bool) -> list[Path]:
    registry = formula_registry()
    written: list[Path] = []
    for fig in cfg["figures"]:
        src = here / fig["source"]
        text = src.read_text()
        rec = json.loads(text)
        sha = hashlib.sha256(text.encode()).hexdigest()
        runs = pfchilds.load_run_set(here, fig["runs"])  # the record's own run set, n = 9, 11 included
        if sorted(runs) != sorted(rec["seeds"]):
            raise ValueError(f"runs have seeds {sorted(runs)}, the record has {rec['seeds']}")
        reference = fig["reference"]
        ratios = ratio_points(rec, runs, reference)
        ref_label, _, _ = pfchilds._style(reference, registry)
        ratio_lines = [f"y = paired ratio to {reference}: geometric mean over seeds of each seed's ratio; bars min..max over seeds",
                       "extrapolated n: each seed's own power-law fit, r = ceil(a_s n^b_s); per-step counts exact",
                       "bars of the absolute figure are not these bars: the same seed is used in both formulas",
                       f"single-axis ratio figure draws bars only at n = {fig['ratio']['bar_sizes']}; the panel figure draws all"]
        abs_lines = ["y = the pfchilds record's plotted values: mean over seeds; extrapolated = power-law fit to mean r",
                     "bars min..max over seeds (per-seed fits when extrapolated)"]
        made = [
            (f"{rec['id']}-ratio", ratio_tex(rec, ratios, reference, fig["ratio"], header_lines(
                rec, src.name, sha, f"cost ratio to {ref_label} vs n", ratio_lines), registry)),
            (f"{rec['id']}-families", panels_tex(rec, absolute_points(rec), None, fig["families"], header_lines(
                rec, src.name, sha, "absolute cost vs n, one panel per family", abs_lines), registry)),
            (f"{rec['id']}-families-ratio", panels_tex(rec, ratios, reference, fig["families_ratio"], header_lines(
                rec, src.name, sha, f"cost ratio to {ref_label} vs n, one panel per family", ratio_lines), registry)),
        ]
        side = src.with_name(f"{rec['id']}-ratio.json")
        side.write_text(json.dumps({"source": src.name, "source_sha256": sha, "reference": reference,
                                    "seeds": rec["seeds"], "metric": rec["metric"], "ratios": ratios}, indent=1))
        written.append(side)
        for stem, tex in made:
            path = src.with_name(stem + ".tex")
            path.write_text(tex)
            written.append(path)
            print(f"  wrote {path}")
            if render:
                pdf, _ = pfchilds.render_tex(path, here / "rendered")
                print(f"  rendered {pdf}")
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args(argv)
    run(json.loads(args.config.read_text()), args.config.parent, args.render)


if __name__ == "__main__":
    main()
