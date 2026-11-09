"""Copies of existing figures with zoomed insets in the empty parts of the plot.

    python -m pysimcount.figinset results/np-childs-format/insets.json [--render]

For each figure in the config this writes ``<stem>-inset.tex`` beside the
source: one main axis plus one small axis per inset, each with a rectangle on
the main axis marking the zoomed window and two lines joining it to the inset.
No number is computed here: every coordinate is copied from the source figure,
and a ``% derived:`` line names the source and its SHA-256. Originals are never
modified.

Differences from the source, all set in the config and listed in the header:

* the plot area is larger and the legend sits below it, so that the figure
  fills the text width (``scale only axis``: width and height are the plot area);
* the y-axis is fixed, which leaves the empty corners the insets go in;
* with ``main_error_bars: false`` the main axis drops its error bars, and only
  the insets show them. The min-max bars of the sizes outside every inset are
  then not drawn at all, so the caption has to say so.

An inset is an ordinary axis on the same log-log scales, not a magnified image:
marks keep their size and only the distances between them grow. A window need
not contain every point; the header lists the points each inset does not show.
Raises if an inset would cover a plotted point or line, if it leaves the plot
area, or if its window contains no plotted point.

The key for open markers is reworded to start ``extrapolated:``, which also
keeps these figures out of figuniform's scope: its restyle assumes one axis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .figdrop import COORD, DERIVED_TAG
from .figuniform import ADDPLOT, _with_forget

SUFFIX = "-inset"
SOURCE_FIT_KEY = "\\addlegendentry{$r$ from power-law fit"
FIT_KEY = "\\addlegendentry{extrapolated: $r$ from power-law fit"
# Room kept clear for an inset's tick labels, left of and below its plot area.
LABEL_MARGIN_CM = (1.0, 0.5)
SEGMENT_SAMPLES = 40  # points checked along each line of the main axis


@dataclass
class Plot:
    """One ``\\addplot`` of the source figure: options, raw coordinate lines,
    the formula's comment line, and the legend lines that follow it."""
    comment: str
    opts: str
    coords: list[str]
    legend: list[str]

    def points(self) -> list[tuple[float, float]]:
        out = []
        for line in self.coords:
            m = COORD.search(line)
            if m is None:
                raise ValueError(f"not a coordinate: {line.strip()}")
            out.append((float(m.group(1)), float(m.group(2))))
        return out


def parse_figure(text: str) -> tuple[list[str], list[Plot], list[str]]:
    """(header comment lines, plots in source order, trailing legend lines)."""
    lines = text.splitlines()
    n_head = next(k for k, l in enumerate(lines) if not l.startswith("%"))
    header = lines[:n_head]
    end_axis = lines.index("  \\end{loglogaxis}")
    plots: list[Plot] = []
    trailing: list[str] = []
    comment = ""
    i = next(k for k, l in enumerate(lines) if l.strip() == "]") + 1
    while i < end_axis:
        s = lines[i].strip()
        m = ADDPLOT.match(lines[i])
        if m:
            end = next(k for k in range(i, end_axis) if lines[k].strip() == "};")
            plots.append(Plot(comment, m.group(2), lines[i + 1:end], []))
            i = end + 1
            continue
        if s.startswith("%"):
            comment = s
        elif s.startswith("\\addlegendentry") and plots and not trailing:
            plots[-1].legend.append(s)
        elif s.startswith("\\addlegendimage") or trailing:
            trailing.append(s)
        else:
            raise ValueError(f"line {i + 1}: unexpected inside the axis: {s}")
        i += 1
    if not plots:
        raise ValueError("no \\addplot found")
    return header, plots, trailing


def _log_frac(v: float, lo: float, hi: float) -> float:
    """Position of v on a log axis from lo to hi: 0 at lo, 1 at hi."""
    return math.log(v / lo) / math.log(hi / lo)


def main_position_cm(x: float, y: float, main: dict) -> tuple[float, float]:
    """(x, y) of a data point in cm from the main plot area's south-west corner."""
    return (_log_frac(x, main["xmin"], main["xmax"]) * main["width_cm"],
            _log_frac(y, main["ymin"], main["ymax"]) * main["height_cm"])


def check_placement(plots: list[Plot], main: dict, inset: dict) -> None:
    """Raise if the inset (with room for its tick labels) leaves the plot area
    or covers any plotted point or line of the main axis."""
    x0, y0 = inset["at_cm"]
    x1, y1 = x0 + inset["width_cm"], y0 + inset["height_cm"]
    x0 -= LABEL_MARGIN_CM[0]
    y0 -= LABEL_MARGIN_CM[1]
    if x0 < 0 or y0 < 0 or x1 > main["width_cm"] or y1 > main["height_cm"]:
        raise ValueError(f"inset {inset['name']!r} leaves the plot area")
    for p in plots:
        pts = p.points()
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            for k in range(SEGMENT_SAMPLES + 1):
                f = k / SEGMENT_SAMPLES
                # a straight segment on log-log axes is linear in log x, log y
                x = xa * (xb / xa) ** f
                y = ya * (yb / ya) ** f
                cx, cy = main_position_cm(x, y, main)
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    raise ValueError(f"inset {inset['name']!r} covers data near (n={x:.3g}, y={y:.3g})")


def in_window(p: Plot, inset: dict) -> bool:
    return any(inset["xmin"] <= x <= inset["xmax"] and inset["ymin"] <= y <= inset["ymax"] for x, y in p.points())


def _series_name(p: Plot) -> str:
    """Formula id from a source comment of the form ``% suzuki4: measured n, ...``."""
    name, sep, _ = p.comment.lstrip("% ").partition(":")
    if not sep:
        raise ValueError(f"plot comment does not name a formula: {p.comment!r}")
    return name


def points_outside(plots: list[Plot], inset: dict) -> list[str]:
    """``formula n=...`` for every plotted point whose n is in the window's
    x-range but whose value is above or below it: the inset's x-axis promises
    that size, so the reader needs to know the point is not shown. Error bars
    are not counted, only the plotted values."""
    out = []
    for p in plots:
        for x, y in p.points():
            if inset["xmin"] <= x <= inset["xmax"] and not inset["ymin"] <= y <= inset["ymax"]:
                out.append(f"{_series_name(p)} n={x:g}")
    return out


def _tick_label(v: float) -> str:
    """Tick label for a y value: $10^{4}$ or $5{\\times}10^{3}$."""
    exp = math.floor(math.log10(v) + 1e-9)
    mant = v / 10**exp
    if abs(mant - round(mant)) > 1e-9:
        raise ValueError(f"y tick {v} is not a single digit times a power of ten")
    return f"$10^{{{exp}}}$" if round(mant) == 1 else f"${round(mant)}{{\\times}}10^{{{exp}}}$"


def _plain_opts(opts: str, mark_size: str | None) -> str:
    head = opts.partition(", error bars/.cd")[0]
    return re.sub(r"mark size=[^,]+", f"mark size={mark_size}", head) if mark_size else head


def _plain_coord(line: str) -> str:
    return line.partition(" +=")[0]


def _main_axis(main: dict, source_xtick: str) -> list[str]:
    return ["  \\begin{loglogaxis}[",
            "    name=main, scale only axis,",
            f"    width={main['width_cm']}cm, height={main['height_cm']}cm,",
            "    xlabel={System size $n$},",
            f"    ylabel={{{main['ylabel']}}},",
            f"    {source_xtick}",
            f"    xmin={main['xmin']:g}, xmax={main['xmax']:g}, ymin={main['ymin']:g}, ymax={main['ymax']:g},",
            "    ymajorgrids=true, grid style={black!12},",
            "    legend style={at={(0.5,-0.13)}, anchor=north, draw=none, font=\\footnotesize,",
            f"      legend columns={main['legend_columns']}, /tikz/every even column/.append style={{column sep=8pt}}}},",
            "    legend cell align=left,",
            "  ]"]


def _inset_axis(inset: dict, plots: list[Plot]) -> list[str]:
    x0, y0 = inset["at_cm"]
    name = inset["name"]
    ticks = inset["ytick"]
    out = ["  \\begin{loglogaxis}[",
           f"    name=inset-{name}, scale only axis,",
           f"    at={{(main.south west)}}, anchor=south west, xshift={x0}cm, yshift={y0}cm,",
           f"    width={inset['width_cm']}cm, height={inset['height_cm']}cm,",
           f"    xmin={inset['xmin']:g}, xmax={inset['xmax']:g}, ymin={inset['ymin']:g}, ymax={inset['ymax']:g},",
           "    xtick={" + ",".join(f"{v:g}" for v in inset["xtick"]) + "}, xticklabels={"
           + ",".join(f"{v:g}" for v in inset["xtick"]) + "},",
           "    ytick={" + ",".join(f"{v:g}" for v in ticks) + "}, yticklabels={"
           + ",".join(_tick_label(v) for v in ticks) + "},",
           "    minor tick num=0, xminorticks=false, yminorticks=false,",
           "    ymajorgrids=true, grid style={black!12},",
           "    tick label style={font=\\scriptsize},",
           "    axis background/.style={fill=white},",
           f"    title={{{inset['title']}}}, title style={{font=\\scriptsize, yshift=-3pt}},",
           "  ]"]
    for p in plots:
        out.append(f"    \\addplot[{_with_forget(p.opts) if 'forget plot' not in p.opts else p.opts}] coordinates {{")
        out += p.coords
        out.append("    };")
    out.append("  \\end{loglogaxis}")
    return out


def _zoom_marks(inset: dict) -> list[str]:
    """Rectangle on the main axis, with named corners for the joining lines."""
    n, (a, b, c, d) = inset["name"], (inset["xmin"], inset["xmax"], inset["ymin"], inset["ymax"])
    return [f"    % zoom window of inset {n}",
            f"    \\draw[black!70] (axis cs:{a:g},{c:g}) rectangle (axis cs:{b:g},{d:g});",
            f"    \\coordinate (zoom-{n}-sw) at (axis cs:{a:g},{c:g});",
            f"    \\coordinate (zoom-{n}-nw) at (axis cs:{a:g},{d:g});",
            f"    \\coordinate (zoom-{n}-se) at (axis cs:{b:g},{c:g});",
            f"    \\coordinate (zoom-{n}-ne) at (axis cs:{b:g},{d:g});"]


def _joining_lines(inset: dict) -> list[str]:
    """Two lines from the window to the facing edge of the inset. An inset above
    its window joins the window's top corners to its bottom corners, and the
    reverse for one below."""
    n = inset["name"]
    if inset["side"] == "above":
        pairs = [("nw", "south west"), ("ne", "south east")]
    elif inset["side"] == "below":
        pairs = [("sw", "north west"), ("se", "north east")]
    else:
        raise ValueError(f"inset {n!r}: side must be 'above' or 'below', got {inset['side']!r}")
    return [f"  \\draw[black!45] (zoom-{n}-{a}) -- (inset-{n}.{b});" for a, b in pairs]


def inset_text(text: str, source_name: str, fig: dict) -> str:
    """The derived figure's full text for source text ``text``."""
    main, insets = fig["main"], fig["insets"]
    for key in ("xmin", "xmax", "ymin", "ymax"):
        if main[key] <= 0:
            raise ValueError(f"main {key} must be positive on a log axis, got {main[key]}")
    header, plots, trailing = parse_figure(text)
    xtick = next((l.strip() for l in text.splitlines() if l.strip().startswith("xtick=")), None)
    if xtick is None:
        raise ValueError(f"{source_name}: no xtick line in the source axis")

    notes = []
    for inset in insets:
        if not (inset["xmin"] < inset["xmax"] and inset["ymin"] < inset["ymax"]):
            raise ValueError(f"inset {inset['name']!r}: empty window")
        shown = [p for p in plots if in_window(p, inset)]
        if not shown:
            raise ValueError(f"inset {inset['name']!r}: no plotted point in its window")
        check_placement(plots, main, inset)
        notes.append(f"inset {inset['name']}: n = {inset['xmin']:g}..{inset['xmax']:g}, "
                     f"y = {inset['ymin']:g}..{inset['ymax']:g}; points outside the y-range, not shown in this inset: "
                     f"{', '.join(points_outside(plots, inset)) or 'none'}")

    sha = hashlib.sha256(text.encode()).hexdigest()
    bars = "kept" if main["main_error_bars"] else "removed on the main axis (insets keep them)"
    out = list(header)
    out.append(f"{DERIVED_TAG} {source_name} with zoomed insets by pysimcount.figinset; coordinates copied unchanged; "
               f"error bars {bars}; source_sha256={sha}")
    out += [f"% {note}" for note in notes]
    out += ["\\begin{tikzpicture}"] + _main_axis(main, xtick)
    for p in plots:
        opts = p.opts if main["main_error_bars"] else _plain_opts(p.opts, main.get("mark_size"))
        coords = p.coords if main["main_error_bars"] else [_plain_coord(c) for c in p.coords]
        out.append(f"    {p.comment}")
        out.append(f"    \\addplot[{opts}] coordinates {{")
        out += coords
        out.append("    };")
        out += [f"    {l}" for l in p.legend]
    for l in trailing:
        out.append("    " + (l.replace(SOURCE_FIT_KEY, FIT_KEY) if l.startswith(SOURCE_FIT_KEY) else l))
    for inset in insets:
        out += _zoom_marks(inset)
    out.append("  \\end{loglogaxis}")
    for inset in insets:
        out += _inset_axis(inset, plots)
        out += _joining_lines(inset)
    out += ["\\end{tikzpicture}", ""]
    return "\n".join(out)


def run(cfg: dict, here: Path, render: bool) -> list[Path]:
    written = []
    for fig in cfg["figures"]:
        src = here / fig["source"]
        out = src.with_name(src.stem + SUFFIX + ".tex")
        out.write_text(inset_text(src.read_text(), src.name, fig))
        written.append(out)
        print(f"  wrote {out}")
        if render:
            from .pfchilds import render_tex
            pdf, _ = render_tex(out, here / "rendered")
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
