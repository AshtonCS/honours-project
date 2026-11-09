"""Copies of existing figures with extrapolated points drawn like measured ones.

    python -m pysimcount.figuniform results [--dry-run]

pffigure, pfseeds, pfchilds and pftcount (and figdrop's copies of their
figures) draw a point whose r comes from the power-law fit as an open marker,
on a dashed or dotted line where the series has a line, and add legend keys
for the distinction. Childs et al. 2018 (arXiv:1711.10980) draw their
extrapolated sizes (Figs. 3, 10, 11: n = 13-500, r from Eq. 147) exactly like
every other point. For every figure under ``results/`` that says its open
markers are extrapolations, this writes ``<stem>-uniform.tex`` beside it:

* each open-marker ``forget plot`` continuation of a series is redrawn with
  the options of that series' own (legend) ``\\addplot``, plus ``forget plot``:
  same colour, mark, fill and line style as its measured points;
* the style-key legend entries (``measured $r$``, ``computed $r$``,
  ``fitted $r$ (extrapolated)``, ``$r$ from power-law fit ...``) are removed,
  and so is pftcount's vertical "fitted $r$" divider;
* every other line is copied byte for byte, header included, plus a
  ``% derived:`` line naming the source file, its SHA-256 and the
  extrapolated n.

The data are unchanged. The figure itself no longer shows which n are
extrapolated, so the caption has to say it; the ``% derived:`` line lists them.
Originals are never modified.

In scope only if the figure states that its open markers are extrapolations:
a legend key whose entry is ``fitted $r$ ...`` or ``$r$ from power-law fit
...``, or a header comment ``open markers: r from ... fit``. Figures that use
open markers for something else (pfgroupfigure: two-group grouping; pffig13 /
pffig13ext: before optimisation; pftcount's t-per-rotation) are not touched.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from .figdrop import COORD, DERIVED_TAG

SUFFIX = "-uniform"
ADDPLOT = re.compile(r"^(\s*)\\addplot\[(.*)\] coordinates \{\s*$")
# Legend keys that explain the open/filled distinction, and nothing else.
STYLE_KEY = re.compile(r"^\\addlegendentry\{(measured \$r\$|computed \$r\$|fitted \$r\$|\$r\$ from power-law fit)")
FIT_KEY = re.compile(r"^\\addlegendentry\{(fitted \$r\$|\$r\$ from power-law fit)")
FIT_HEADER = re.compile(r"open markers: r from .*fit")
DIVIDER_LABEL = re.compile(r"^\\node\[.*\] at \(axis cs:.*\) \{fitted \$r\$\};$")


def _is_open(opts: str) -> bool:
    return "fill=white" in opts


def _with_forget(opts: str) -> str:
    """``opts`` plus ``forget plot``, placed before ``error bars/.cd``: keys
    after it are read in the error-bars family, where forget plot is unknown."""
    head, sep, tail = opts.partition(", error bars/.cd")
    return f"{head}, forget plot{sep}{tail}"


def in_scope(text: str) -> bool:
    """True if the figure says its open markers are r from the power-law fit."""
    lines = [l.strip() for l in text.splitlines()]
    if any(l.startswith("%") and FIT_HEADER.search(l) for l in lines):
        return True
    return any(a.startswith("\\addlegendimage{") and _is_open(a) and FIT_KEY.match(b)
               for a, b in zip(lines, lines[1:]))


def restyle(text: str) -> tuple[str, list[int]]:
    """``text`` with every open-marker continuation drawn in its series' own
    style and the style keys removed. Returns the new text and the
    extrapolated n (x of open points not also plotted by the series head)."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    head_opts: str | None = None
    head_xs: set[int] = set()
    extrapolated: set[int] = set()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        m = ADDPLOT.match(lines[i].rstrip("\n"))
        if m:
            indent, opts = m.groups()
            end = next(k for k in range(i, len(lines)) if lines[k].strip() == "};")
            xs = {int(float(c.group(1))) for c in COORD.finditer("".join(lines[i:end]))}
            if "forget plot" not in opts:
                if _is_open(opts):
                    raise ValueError(f"line {i + 1}: open-marker series with its own legend entry; "
                                     "no measured style to copy")
                head_opts, head_xs = opts, xs
                out.append(lines[i])
            elif _is_open(opts):
                if head_opts is None:
                    raise ValueError(f"line {i + 1}: open-marker continuation before any series")
                if ("error bars" in opts) != ("error bars" in head_opts):
                    raise ValueError(f"line {i + 1}: continuation and series disagree on error bars")
                out.append(f"{indent}\\addplot[{_with_forget(head_opts)}] coordinates {{\n")
                extrapolated |= xs - head_xs
            else:
                out.append(lines[i])
            out += lines[i + 1:end + 1]
            i = end + 1
            continue
        if s.startswith("\\addlegendimage{") and i + 1 < len(lines) and STYLE_KEY.match(lines[i + 1].strip()):
            i += 2
            continue
        if s.startswith("\\draw[") and i + 1 < len(lines) and DIVIDER_LABEL.match(lines[i + 1].strip()):
            i += 2
            continue
        if DIVIDER_LABEL.match(s):
            raise ValueError(f"line {i + 1}: 'fitted $r$' label without its divider line")
        out.append(lines[i])
        i += 1
    new = "".join(out)
    left = [l for l in new.splitlines() if l.lstrip().startswith("\\addplot[") and _is_open(l)]
    if left:
        raise ValueError(f"open markers remain: {left[0].strip()[:80]}")
    if not extrapolated:
        raise ValueError("no extrapolated points found")
    return new, sorted(extrapolated)


def derived_text(src: Path) -> str:
    """The copy written beside ``src``: original header, a ``% derived:``
    line, then the restyled body."""
    text = src.read_text()
    body, ns = restyle(text)
    lines = body.splitlines(keepends=True)
    n_head = next(k for k, l in enumerate(lines) if not l.startswith("%"))
    sha = hashlib.sha256(text.encode()).hexdigest()
    note = (f"{DERIVED_TAG} {src.name} restyled by pysimcount.figuniform: extrapolated points "
            f"(r from power-law fit) drawn in their series' measured style, style-key legend entries removed; "
            f"extrapolated n = {ns}; source_sha256={sha}\n")
    return "".join(lines[:n_head]) + note + "".join(lines[n_head:])


def main(argv: list[str] | None = None) -> None:
    from .figgallery import is_figure

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    written = []
    for p in sorted(args.results.rglob("*.tex")):
        if p.stem.endswith(SUFFIX) or not is_figure(p):
            continue
        if not in_scope(p.read_text()):
            continue
        out = p.with_name(p.stem + SUFFIX + ".tex")
        text = derived_text(p)
        if not args.dry_run:
            out.write_text(text)
        written.append(out.relative_to(args.results).as_posix())
    print("\n".join(written))
    print(f"\n{len(written)} figure(s) {'would be restyled' if args.dry_run else 'restyled'}")


if __name__ == "__main__":
    main()
