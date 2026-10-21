"""Analysis and figures for an eps sweep made by ``pfcompare`` (config key ``eps_trotter``).

    python -m plsimcount.pfeps results/pf-eps/results.json --render

Theory: for an order-p product formula the Trotter error over time t with r
steps behaves as C t^(p+1) / r^p once r is large (the asymptotic regime), so
the smallest r with error <= eps scales as

    r ∝ eps^(-1/p),   i.e.  d log r / d log eps -> -1/p.

This is the leading-order error scaling of any order-p formula (e.g. Childs
et al., PRX 11, 011020 (2021), Sec. III -- I have not checked the exact
equation number). It says nothing about the constant, and it only holds
asymptotically: when r is a handful of steps the ceiling in "smallest
integer r" dominates and the measured slope can be far from -1/p. Such
points are flagged (r <= SMALL_R at any eps in the fit).

What is computed per (formula, n):

* the least-squares slope of log r against log eps over all eps values;
* the two pairwise slopes (adjacent eps values), to see whether the slope
  drifts towards -1/p as eps falls (r grows);
* slope / (-1/p): 1 = textbook scaling.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SMALL_R = 3  # r at or below this: discreteness dominates (flagged, not dropped)


@dataclass
class Slope:
    scheme: str
    family: str
    order: int
    n: int
    eps: list[float]            # ascending
    r: list[int]                # r at each eps
    fit_slope: float            # least squares d log r / d log eps
    pair_slopes: list[float]    # between adjacent eps values, ascending eps
    theory: float               # -1 / order
    small_r: bool               # some r <= SMALL_R

    @property
    def ratio(self) -> float:
        return self.fit_slope / self.theory


def loglog_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of log y against log x (needs >= 2 points)."""
    if len(xs) < 2:
        raise ValueError("a slope needs at least two points")
    b, _ = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(b)


def slopes(rows: list[dict], gatebase: str = "cz") -> list[Slope]:
    """One Slope per (scheme, n) from the measured rows (merged and unmerged
    rows carry the same r; they must agree or this raises)."""
    table: dict[tuple[str, int], dict[float, int]] = {}
    meta: dict[str, tuple[str, int]] = {}
    for row in rows:
        if row["instance"] != "simcount" or row["r_source"] != "measured" or row["gatebase"] != gatebase:
            continue
        per = table.setdefault((row["scheme"], int(row["n"])), {})
        e, r = float(row["eps_trotter"]), int(row["r"])
        if e in per and per[e] != r:
            raise ValueError(f"two r for {row['scheme']} n={row['n']} eps={e}: {per[e]}, {r}")
        per[e] = r
        meta[row["scheme"]] = (row["family"], int(row["order"]))
    out = []
    for (scheme, n), per in sorted(table.items()):
        eps = sorted(per)
        rs = [per[e] for e in eps]
        pairs = []
        for i in range(len(eps) - 1):
            pairs.append(math.log(rs[i + 1] / rs[i]) / math.log(eps[i + 1] / eps[i]))
        fam, order = meta[scheme]
        out.append(Slope(scheme, fam, order, n, eps, rs, loglog_slope(eps, rs), pairs, -1.0 / order,
                         min(rs) <= SMALL_R))
    return out


def slope_table(sl: list[Slope], scheme_order: list[str]) -> str:
    eps = sl[0].eps
    head = ("| scheme | p | n | " + " | ".join(f"r @ {e:g}" for e in eps)
            + " | fit slope | " + " | ".join(f"slope {eps[i]:g}->{eps[i + 1]:g}" for i in range(len(eps) - 1))
            + " | -1/p | fit / (-1/p) | r <= 3 |")
    lines = [head, "|---|---:|---:|" + "---:|" * (len(eps) + 1 + len(eps) - 1 + 2) + "---|"]
    rank = {s: i for i, s in enumerate(scheme_order)}
    for x in sorted(sl, key=lambda x: (rank[x.scheme], x.n)):
        lines.append(f"| {x.scheme} | {x.order} | {x.n} | " + " | ".join(f"{r:,}" for r in x.r)
                     + f" | {x.fit_slope:.3f} | " + " | ".join(f"{s:.3f}" for s in x.pair_slopes)
                     + f" | {x.theory:.3f} | {x.ratio:.2f} | {'yes' if x.small_r else ''} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ figures

def figure_tex(rows: list[dict], order: int, sizes: list[int], column: str, ylabel: str,
               header: list[str]) -> str:
    """CNOT (or other) total vs eps, every formula of one order, one series per
    (formula, n): solid line n = sizes[-1], dashed for the others. A dotted
    black line of slope -1/p through the first formula's middle point at the
    largest n shows the theoretical scaling."""
    from .pfcompare import formula_registry
    from .pffigure import COLOUR_OVERRIDE, FAMILY_COLOUR, ORDER_MARK, latex_escape
    reg = formula_registry()
    pts = [r for r in rows if r["instance"] == "simcount" and r["r_source"] == "measured" and r["merged"]
           and r["gatebase"] == "cz" and int(r["order"]) == order and int(r["n"]) in sizes]
    schemes = [s for s in reg if any(r["scheme"] == s for r in pts)]
    out = [f"% {h}" for h in header] + [
        "\\begin{tikzpicture}", "  \\begin{loglogaxis}[", "    width=10cm,",
        "    xlabel={Trotter error target $\\varepsilon$},", f"    ylabel={{{ylabel}}},",
        "    ymajorgrids=true,", "    legend pos=outer north east,", "    legend cell align=left,",
        "    legend style={font=\\footnotesize},", "  ]"]
    for s in schemes:
        f = reg[s]
        colour = COLOUR_OVERRIDE.get(s, FAMILY_COLOUR[f.family])
        mark = ORDER_MARK[f.order]
        for n in sizes:
            ser = sorted((float(r["eps_trotter"]), float(r[column])) for r in pts if r["scheme"] == s and int(r["n"]) == n)
            style = "solid" if n == sizes[-1] else "dashed"
            out.append(f"    \\addplot[{colour}, {style}, mark={mark}, mark size=2.2pt, mark options={{solid, fill={colour}}}] "
                       "coordinates {")
            out += [f"      ({e:g},{y:.0f})" for e, y in ser]
            out += ["    };", f"    \\addlegendentry{{{latex_escape(f.name)}, $n={n}$}}"]
    # reference slope through the first formula's middle-eps point at the largest n
    ref = sorted((float(r["eps_trotter"]), float(r[column])) for r in pts
                 if r["scheme"] == schemes[0] and int(r["n"]) == sizes[-1])
    e0, y0 = ref[len(ref) // 2]
    lo, hi = ref[0][0], ref[-1][0]
    out += ["    \\addplot[black, dotted, thick, domain=%g:%g, samples=2] {%g*(x/%g)^(%g)};" % (lo, hi, y0, e0, -1.0 / order),
            f"    \\addlegendentry{{slope $-1/{order}$ (theory)}}",
            "  \\end{loglogaxis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def write_figures(results: Path, sizes: list[int]) -> Path:
    from .pffigure import PREAMBLE
    data = json.loads(results.read_text())
    rows, meta = data["rows"], data["meta"]
    root = results.parent / "figures"
    root.mkdir(exist_ok=True)
    prov = meta["provenance"]
    pages = []
    orders = sorted({int(r["order"]) for r in rows})
    for order in orders:
        for column, ylabel, tag in (("cnot_total", "CNOT count", "cnot"),):
            header = ["Generated by plsimcount.pfeps -- do not edit; regenerate instead.",
                      f"order {order}, {tag} total (merged, cz) vs eps_trotter, n in {sizes}, measured r, seed "
                      f"{meta['config']['seed']}, t = n",
                      f"git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}"]
            rel = f"{tag}-order{order}-vs-eps.tex"
            (root / rel).write_text(figure_tex(rows, order, sizes, column, ylabel, header))
            pages.append(f"\\section*{{\\texttt{{{rel}}}}}\nOrder {order}: CNOT count vs $\\varepsilon$ "
                         f"(merged, measured $r$; solid $n={sizes[-1]}$, dashed $n={sizes[0]}$)\\par\\medskip\n"
                         f"\\input{{{rel}}}\n\\clearpage\n")
    (root / "all.tex").write_text(PREAMBLE + "\n".join(pages) + "\\end{document}\n")
    return root


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--figure-sizes", type=int, nargs="+", default=[8, 10])
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args(argv)
    data = json.loads(args.results.read_text())
    sl = slopes(data["rows"])
    order = data["meta"]["config"]["schemes"]
    md = ["# r against eps_trotter: log-log slopes vs -1/p", "",
          f"Source: `{args.results.name}` (cz, seed {data['meta']['config']['seed']}, t = n, measured r). "
          f"Rows with r <= {SMALL_R} at some eps are flagged: there the ceiling dominates.", "",
          slope_table(sl, order), ""]
    (args.results.parent / "slopes.md").write_text("\n".join(md))
    (args.results.parent / "slopes.json").write_text(json.dumps([x.__dict__ for x in sl], indent=1))
    root = write_figures(args.results, args.figure_sizes)
    print("\n".join(md))
    if args.render:
        from .pffigure import render
        print(f"rendered {render(root)}")


if __name__ == "__main__":
    main()
