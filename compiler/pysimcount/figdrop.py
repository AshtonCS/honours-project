"""Copies of existing figures with one formula's series removed.

    python -m pysimcount.figdrop results [--formula lie-trotter] [--min-share 0.3333] [--dry-run]

Some figures are hard to read because one formula (usually Lie-Trotter S1)
sits orders of magnitude above the rest and the log axis squeezes everything
else into a small band. For every figure under ``results/`` that plots the
formula alongside at least one other, this measures

    share = (decades of the y-range with the formula - decades without) / decades with

i.e. the fraction of the autoscaled log y-axis that exists only because of
that formula (y-range includes error-bar ends, as pgfplots' autoscaling
does). Where ``share >= --min-share`` it writes ``<stem>-no-<formula>.tex``
beside the original, with the formula's series removed and every other line
copied byte for byte. Originals are never modified. The copy keeps the
original header and adds a ``% derived:`` line naming the source file and its
SHA-256, which ``figgallery`` reads to show the copy in place of the original.

Figures that fix their y-axis (``ymin=`` / ``ymax=``, e.g. pffig13's, set to
match a published figure) get no copy: removing a series would not rescale
the axis, so the copy would only add empty space. They are listed as such.

Every candidate and its share is listed in ``results/gallery/<formula>-removed.md``.

Series model (what pffigure, pfseeds and pfoptimised write): an
``\\addplot[...] coordinates {...};`` followed by ``\\addlegendentry`` starts a
series; a following ``forget plot`` addplot with no legend entry continues it
(e.g. the open-marker extrapolated points). ``\\addlegendimage`` entries are
style keys, not data, and are kept. Anything else raises.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .pfcompare import formula_registry

# A coordinate, optionally with explicit asymmetric y error bars: (x,y) += (0,hi) -= (0,lo)
COORD = re.compile(r"\(([-+\d.eE]+),([-+\d.eE]+)\)(?:\s*\+=\s*\(0,([-+\d.eE]+)\)\s*-=\s*\(0,([-+\d.eE]+)\))?")
DERIVED_TAG = "% derived:"


@dataclass
class Series:
    """One legend entry's data: its line ranges [start, stop) in the file
    and the y-values that set the autoscaled axis (points and error-bar ends)."""
    legend: str
    spans: list[tuple[int, int]] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)


def _ys(block: str) -> list[float]:
    out = []
    for m in COORD.finditer(block):
        y = float(m.group(2))
        hi, lo = float(m.group(3) or 0), float(m.group(4) or 0)
        out += [y + hi, y - lo if y - lo > 0 else y]  # a bar to <= 0 is not drawn on a log axis
    return out


def parse_series(text: str) -> tuple[list[Series], int]:
    """The data series of a pgfplots figure, by line (see module docstring),
    and the number of one-line expression plots (``\\addplot[...] {expr};``,
    e.g. pfeps's theory-slope guides), which are not data and are skipped."""
    lines = text.splitlines(keepends=True)
    series: list[Series] = []
    guides = 0
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("\\addlegendimage"):
            i += 2  # the style key and its legend entry
            continue
        if not s.startswith("\\addplot"):
            i += 1
            continue
        if s.endswith("};") and "coordinates" not in s:
            guides += 1
            i += 1
            if i < len(lines) and lines[i].strip().startswith("\\addlegendentry{"):
                i += 1
            continue
        if not s.endswith("coordinates {"):
            raise ValueError(f"line {i + 1}: only '\\addplot[...] coordinates {{' blocks are supported: {s[:80]}")
        start = i
        while lines[i].strip() != "};":
            i += 1
            if i == len(lines):
                raise ValueError(f"line {start + 1}: unterminated \\addplot")
        i += 1
        block = "".join(lines[start:i])
        forget = "forget plot" in lines[start]
        legend = None
        if i < len(lines) and lines[i].strip().startswith("\\addlegendentry{"):
            legend = lines[i].strip()[len("\\addlegendentry{"):-1]
            i += 1
        if forget and legend is None:
            if not series:
                raise ValueError(f"line {start + 1}: 'forget plot' block before any series")
            series[-1].spans.append((start, i))
            series[-1].ys += _ys(block)
        elif not forget and legend is not None:
            series.append(Series(legend, [(start, i)], _ys(block)))
        else:
            raise ValueError(f"line {start + 1}: addplot with forget plot={forget} and legend={legend!r} "
                             "does not fit the series model")
    return series, guides


def _decades(ys: list[float]) -> float:
    return math.log10(max(ys) / min(ys))


def share(series: list[Series], name: str) -> float:
    """Fraction of the log y-axis that only the series named ``name`` needs."""
    mine = [y for s in series if s.legend.startswith(name) for y in s.ys]
    rest = [y for s in series if not s.legend.startswith(name) for y in s.ys]
    if not mine or not rest:
        raise ValueError(f"need series both starting and not starting with {name!r}")
    full = _decades(mine + rest)
    return (full - _decades(rest)) / full


def drop(text: str, name: str) -> tuple[str, list[str]]:
    """``text`` without every series whose legend starts with ``name``; all
    other lines unchanged. Returns the new text and the removed legends."""
    series, guides = parse_series(text)
    if guides:
        raise ValueError(f"{guides} expression plot(s) present; they may be drawn from the removed series")
    gone = [s for s in series if s.legend.startswith(name)]
    if not gone:
        raise ValueError(f"no series named {name!r}")
    if len(gone) == len(series):
        raise ValueError(f"every series is {name!r}; nothing would remain")
    cut = {k for s in gone for a, b in s.spans for k in range(a, b)}
    lines = text.splitlines(keepends=True)
    return "".join(l for k, l in enumerate(lines) if k not in cut), [s.legend for s in gone]


def derived_text(src: Path, name: str, frac: float) -> str:
    """The copy written beside ``src``: original header, a ``% derived:``
    line, then the body with ``name``'s series removed."""
    text = src.read_text()
    body, gone = drop(text, name)
    lines = body.splitlines(keepends=True)
    n_head = next(k for k, l in enumerate(lines) if not l.startswith("%"))
    sha = hashlib.sha256(text.encode()).hexdigest()
    note = (f"{DERIVED_TAG} {src.name} with {name} removed by pysimcount.figdrop "
            f"({name} alone spans {frac:.0%} of the original y-axis); "
            f"removed legends: {'; '.join(gone)}; source_sha256={sha}\n")
    return "".join(lines[:n_head]) + note + "".join(lines[n_head:])


FIXED_Y = re.compile(r"(^|[\s,\[])y(min|max)\s*=")


def has_fixed_y(text: str) -> bool:
    """True if the axis options set ymin or ymax (pgfplots then does not autoscale y)."""
    return any(FIXED_Y.search(l) for l in text.splitlines() if not l.lstrip().startswith("%"))


def is_derived(text: str) -> bool:
    return any(l.startswith(DERIVED_TAG) for l in text.splitlines() if l.startswith("%"))


def main(argv: list[str] | None = None) -> None:
    from .figgallery import is_figure

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--formula", default="lie-trotter", help="scheme id from the formula registry")
    ap.add_argument("--min-share", type=float, default=1 / 3,
                    help="write a copy when the formula alone needs at least this fraction of the y-axis")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not 0 < args.min_share < 1:
        raise SystemExit("--min-share must be in (0, 1)")
    name = formula_registry()[args.formula].name
    suffix = f"-no-{args.formula}"
    rows = []
    for p in sorted(args.results.rglob("*.tex")):
        if p.stem.endswith(suffix) or not is_figure(p):
            continue
        text = p.read_text()
        if is_derived(text):
            continue
        if f"\\addlegendentry{{{name}" not in text:
            continue
        series, _ = parse_series(text)
        named = [s for s in series if s.legend.startswith(name)]
        if not named or len(named) == len(series):
            continue  # formula absent, or a one-formula figure: nothing to compare
        frac = share(series, name)
        out = p.with_name(p.stem + suffix + ".tex")
        fixed = has_fixed_y(text)
        keep = frac >= args.min_share and not fixed
        if keep and not args.dry_run:
            out.write_text(derived_text(p, name, frac))
        rows.append((p.relative_to(args.results).as_posix(), frac, "yes" if keep else "no (fixed y-axis)" if fixed else "no"))
    rows.sort(key=lambda r: -r[1])
    md = [f"# Figures with {name} removed", "",
          f"Written by `python -m pysimcount.figdrop {args.results} --formula {args.formula} "
          f"--min-share {args.min_share:.4g}`.", "",
          f"share = fraction of the original log y-axis needed only by {name}. "
          f"A copy `<stem>{suffix}.tex` is written where share >= {args.min_share:.4g}; "
          "the gallery shows it in place of the original.", "",
          "| figure | share | copy |", "|---|---:|---|"]
    md += [f"| `{p}` | {f:.2f} | {k} |" for p, f, k in rows]
    if not args.dry_run:
        (args.results / "gallery").mkdir(exist_ok=True)
        (args.results / "gallery" / f"{args.formula}-removed.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n{sum(k == 'yes' for *_, k in rows)} of {len(rows)} candidate figures "
          f"{'would get' if args.dry_run else 'got'} a copy without {name}")


if __name__ == "__main__":
    main()
