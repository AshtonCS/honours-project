"""Who is ahead at one n: a ranked dot plot, the full figure beside it, and mean rank against n.

    python -m pysimcount.figleader results/np-childs-format/leaderboard.json [--render]

For one pfchilds record ``<id>.json`` (and its source ``<id>.tex``) and one size
``n`` this module writes

``<id>-n<n>-dots.tex``     one row per formula, cheapest at the top: the record's
                           plotted count at n, min-max bar over seeds, and two
                           columns of seed statistics (below)
``<id>-with-n<n>.tex``     the source figure's curves (error bars removed,
                           coordinates copied unchanged) with the dot plot beside it
``<id>-mean-rank.tex``     mean over seeds of each formula's rank, against n
``<id>-leader.json``       every number drawn

Ranks are per seed: on each seed's Hamiltonian instance the formulas are sorted
by cost, 1 = cheapest. Measured n use the runs' totals; extrapolated n use each
seed's own power-law fit, c(n) ceil(a_s n^b_s), as in figratio. The two seed
columns of the dot plot are

    cheapest on   number of seeds on which the formula has rank 1
    rank          min-max of its rank over seeds

The dot plot's position is the record's plotted value (mean over seeds when
measured; power-law fit to the mean r when extrapolated), so it matches the
source figure, while the rank columns come from the per-seed values. The two can
disagree in order when formulas are close; ``<id>-leader.json`` records both and
the header says if the order differs.

Ranking is ordinal: a 0.4% gap and a factor-of-2 gap both count as one rank. The
dot plot shows the size of the gap; the rank figure only the order. Equal costs
on a seed share the mean of their ranks (fractional ranking); an exact tie for
cheapest at the dot plot's n raises, since "cheapest on" would then need a rule.
Raises if the per-seed values do not reproduce the record's min and max, or if a
bar leaves the x-range.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from . import figinset, figratio, pfchilds
from .figdrop import DERIVED_TAG
from .pfcompare import formula_registry, provenance

LABEL_GAP = 0.42  # minimum spacing of the mean-rank figure's end labels, in ranks


# ------------------------------------------------------------------ the numbers

def per_seed_at(rec: dict, measured: dict[str, dict[int, dict[int, int]]], n: int) -> dict[str, dict[int, int]]:
    """{scheme: {seed: cost}} at n: the runs if n was measured, else each seed's own fit.
    A record with ``per_seed_stored`` (pftchilds) carries every per-seed value itself."""
    shown = rec["shown"]
    if rec.get("per_seed_stored"):
        out = {}
        for k in shown:
            s = rec["series"][k]
            pts = [p for p in s["measured"] + s["extrapolated"] if p["n"] == n]
            if len(pts) != 1:
                raise ValueError(f"n={n} is neither measured nor extrapolated in {rec['id']}")
            out[k] = {int(seed): v for seed, v in pts[0]["per_seed"].items()}
        return out
    if all(n in measured[k] for k in shown):
        return {k: measured[k][n] for k in shown}
    if n not in rec["extrapolate_sizes"]:
        raise ValueError(f"n={n} is neither measured nor extrapolated in {rec['id']}")
    return {k: figratio.extrapolated_per_seed(rec, k, n) for k in shown}


def seed_ranks(values: dict[str, dict[int, int]]) -> dict[str, dict[int, float]]:
    """{scheme: {seed: rank}}, rank 1 = cheapest on that seed. Equal costs share
    the mean of the ranks they span (fractional ranking)."""
    schemes = list(values)
    seeds = sorted(values[schemes[0]])
    for k in schemes:
        if sorted(values[k]) != seeds:
            raise ValueError(f"{k}: seeds {sorted(values[k])}, expected {seeds}")
    out: dict[str, dict[int, float]] = {k: {} for k in schemes}
    for seed in seeds:
        order = sorted(schemes, key=lambda k: values[k][seed])
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]][seed] == values[order[i]][seed]:
                j += 1
            for k in order[i:j + 1]:
                out[k][seed] = (i + 1 + j + 1) / 2  # ranks i+1 .. j+1 share their mean
            i = j + 1
    return out


def plotted_at(rec: dict, scheme: str, n: int) -> dict:
    """The record's plotted point at n: {value, min, max, measured}."""
    s = rec["series"][scheme]
    for p in s["measured"]:
        if p["n"] == n:
            return {"value": p["mean"], "min": p["min"], "max": p["max"], "measured": True}
    for p in s["extrapolated"]:
        if p["n"] == n:
            return {"value": p["value"], "min": p["min"], "max": p["max"], "measured": False}
    raise ValueError(f"{scheme}: no plotted point at n={n}")


def leaderboard(rec: dict, measured: dict[str, dict[int, dict[int, int]]], n: int) -> list[dict]:
    """One row per shown formula at n, sorted by the record's plotted value."""
    values = per_seed_at(rec, measured, n)
    ranks = seed_ranks(values)
    seeds = sorted(values[rec["shown"][0]])
    for seed in seeds:
        firsts = [k for k in rec["shown"] if ranks[k][seed] < 1.5]
        if len(firsts) != 1 or ranks[firsts[0]][seed] != 1:
            raise ValueError(f"n={n} seed {seed}: tie for cheapest among {firsts}")
    rows = []
    for k in rec["shown"]:
        p = plotted_at(rec, k, n)
        v = values[k]
        if (min(v.values()), max(v.values())) != (p["min"], p["max"]):
            raise ValueError(f"{k} n={n}: per-seed values do not reproduce the record's min, max")
        r = list(ranks[k].values())
        rows.append({"scheme": k, **p, "seed_mean": sum(v.values()) / len(v),
                     "wins": sum(1 for x in r if x == 1), "rank_min": min(r), "rank_max": max(r),
                     "mean_rank": sum(r) / len(r)})
    rows.sort(key=lambda row: row["value"])
    return rows


def mean_rank_series(rec: dict, measured: dict[str, dict[int, dict[int, int]]]) -> dict[str, list[dict]]:
    """{scheme: [{n, mean_rank, measured}, ...]} over every plotted n."""
    first = rec["series"][rec["shown"][0]]
    sizes = [(p["n"], True) for p in first["measured"]] + [(p["n"], False) for p in first["extrapolated"]]
    out: dict[str, list[dict]] = {k: [] for k in rec["shown"]}
    for n, is_measured in sizes:
        ranks = seed_ranks(per_seed_at(rec, measured, n))
        for k in rec["shown"]:
            r = list(ranks[k].values())
            out[k].append({"n": n, "mean_rank": sum(r) / len(r), "measured": is_measured})
    return out


def spread_labels(ys: dict[str, float], gap: float, lo: float, hi: float) -> dict[str, float]:
    """Label positions at least ``gap`` apart, each as close to its y as that allows.
    Pushes down from the top, then up from the bottom if the last one left [lo, hi]."""
    order = sorted(ys, key=lambda k: ys[k])
    pos = {}
    prev = None
    for k in order:
        y = max(ys[k], lo) if prev is None else max(ys[k], prev + gap)
        pos[k] = y
        prev = y
    nxt = None
    for k in reversed(order):
        y = min(pos[k], hi) if nxt is None else min(pos[k], nxt - gap)
        pos[k] = y
        nxt = y
    if pos[order[0]] < lo - 1e-9:
        raise ValueError(f"{len(ys)} labels {gap} apart do not fit in {lo}..{hi}")
    return pos


# ------------------------------------------------------------------ tikz

def _rank_text(row: dict) -> str:
    def fmt(x: float) -> str:
        return f"{x:g}"
    if row["rank_min"] == row["rank_max"]:
        return fmt(row["rank_min"])
    return f"{fmt(row['rank_min'])}--{fmt(row['rank_max'])}"


def _check_x(rows: list[dict], xmin: float, xmax: float) -> None:
    for row in rows:
        if row["min"] < xmin or row["max"] > xmax:
            raise ValueError(f"{row['scheme']}: {row['min']:.4g}..{row['max']:.4g} outside x-range {xmin:g}..{xmax:g}")


def dots_axis(rows: list[dict], n: int, n_seeds: int, cfg: dict, registry: dict, at_cm: tuple[float, float]) -> list[str]:
    """Rows top to bottom in the order given; x = count, log scale; bars = min-max over seeds."""
    _check_x(rows, cfg["xmin"], cfg["xmax"])
    labels = cfg.get("labels", {})
    names = [labels.get(r["scheme"], pfchilds._style(r["scheme"], registry)[0]) for r in rows]
    ticks = ",".join(f"{v:g}" for v in cfg["xtick"])
    tick_labels = ",".join(f"{v / cfg['x_unit']:g}" for v in cfg["xtick"])
    k = len(rows)
    out = ["  \\begin{semilogxaxis}[",
           "    scale only axis, clip=false,",
           f"    at={{({at_cm[0]}cm,{at_cm[1]}cm)}}, anchor=south west,",
           f"    width={cfg['width_cm']}cm, height={cfg['height_cm']}cm,",
           f"    title={{{cfg['title']}}}, title style={{font=\\small, yshift={cfg.get('title_yshift', '14pt')}}},",
           f"    xlabel={{{cfg['xlabel']}}}, xlabel style={{font=\\footnotesize}},",
           f"    xmin={cfg['xmin']:g}, xmax={cfg['xmax']:g}, xtick={{{ticks}}}, xticklabels={{{tick_labels}}},",
           "    xminorticks=false,",
           f"    y dir=reverse, ymin=0.4, ymax={k + 0.6}, ytick={{{','.join(str(i + 1) for i in range(k))}}},",
           f"    yticklabels={{{','.join('{' + s + '}' for s in names)}}},",
           "    tick label style={font=\\footnotesize}, ytick style={draw=none},",
           "    xmajorgrids=true, ymajorgrids=true, grid style={black!10},",
           "  ]"]
    for i, (row, name) in enumerate(zip(rows, names), start=1):
        _, colour, mark = pfchilds._style(row["scheme"], registry)
        fill = colour if row["measured"] else "white"
        out += [f"    % {row['scheme']}: plotted {row['value']:.6g}, seeds {row['min']:.6g}..{row['max']:.6g}, "
                f"cheapest on {row['wins']}/{n_seeds}, rank {row['rank_min']:g}..{row['rank_max']:g}",
                f"    \\addplot[{colour}, only marks, mark={mark}, mark size={cfg['mark_size']}, "
                f"mark options={{solid, fill={fill}, line width=0.6pt}}, error bars/.cd, x dir=both, x explicit, "
                f"error bar style={{solid, line width=0.6pt}}] coordinates {{",
                f"      ({row['value']:.6g},{i}) += ({row['max'] - row['value']:.6g},0) -= ({row['value'] - row['min']:.6g},0)",
                "    };"]
    cols = cfg["columns_cm"]
    head = f"{cfg['xmax']:g},0.4"
    out += [f"    \\node[anchor=south, font=\\scriptsize, align=center] at ([xshift={cols[0]}cm]axis cs:{head}) "
            f"{{cheapest\\\\on}};",
            f"    \\node[anchor=south, font=\\scriptsize, align=center] at ([xshift={cols[1]}cm]axis cs:{head}) "
            f"{{rank\\\\range}};"]
    for i, row in enumerate(rows, start=1):
        bold = "\\bfseries" if row["wins"] else ""
        out += [f"    \\node[font=\\footnotesize{bold}] at ([xshift={cols[0]}cm]axis cs:{cfg['xmax']:g},{i}) "
                f"{{{row['wins']}/{n_seeds}}};",
                f"    \\node[font=\\footnotesize] at ([xshift={cols[1]}cm]axis cs:{cfg['xmax']:g},{i}) {{{_rank_text(row)}}};"]
    out.append("  \\end{semilogxaxis}")
    return out


def _note(cfg: dict, rec: dict, n: int, width_cm: float, x_cm: float, y_cm: float) -> str:
    text = cfg["note"].replace("{fit}", figratio._fit_note(rec)).replace("{n}", str(n)).replace(
        "{seeds}", str(len(rec["seeds"])))
    return (f"  \\node[anchor=north, text width={width_cm}cm, align=flush center, font=\\footnotesize] "
            f"at ({x_cm}cm,{y_cm}cm) {{{text}}};")


def dots_tex(rec: dict, rows: list[dict], n: int, cfg: dict, header: list[str], registry: dict) -> str:
    out = header + ["\\begin{tikzpicture}"]
    out += dots_axis(rows, n, len(rec["seeds"]), cfg, registry, (0, 0))
    out.append(_note(cfg, rec, n, cfg["note_width_cm"], cfg["width_cm"] / 2 + cfg.get("note_shift_cm", 0), -1.1))
    out += ["\\end{tikzpicture}", ""]
    return "\n".join(out)


def _main_axis(main: dict, source_xtick: str) -> list[str]:
    """figinset's main axis, with the legend's distance below the axis configurable."""
    return ["  \\begin{loglogaxis}[",
            "    name=main, scale only axis,",
            f"    width={main['width_cm']}cm, height={main['height_cm']}cm,",
            "    xlabel={System size $n$},",
            f"    ylabel={{{main['ylabel']}}},",
            f"    {source_xtick}",
            f"    xmin={main['xmin']:g}, xmax={main['xmax']:g}, ymin={main['ymin']:g}, ymax={main['ymax']:g},",
            "    ymajorgrids=true, grid style={black!12},",
            f"    legend style={{at={{(0.5,{main['legend_y']})}}, anchor=north, draw=none, font=\\footnotesize,",
            f"      legend columns={main['legend_columns']}, /tikz/every even column/.append style={{column sep=8pt}}}},",
            "    legend cell align=left,",
            "  ]"]


def side_by_side_tex(source_text: str, rec: dict, rows: list[dict], n: int, cfg: dict, header: list[str],
                     registry: dict) -> str:
    """The source figure's curves on the left, without error bars; the dot plot on the right.
    Legend entries take the dot plot's short labels; the style key is rewritten (no
    figuniform key text, so figuniform leaves the figure alone)."""
    main, dots = cfg["main"], cfg["dots"]
    _, plots, _ = figinset.parse_figure(source_text)
    xtick = next(l.strip() for l in source_text.splitlines() if l.strip().startswith("xtick="))
    labels = dots.get("labels", {})
    out = header + ["\\begin{tikzpicture}"]
    out += _main_axis(main, xtick)
    out.append(f"    \\fill[black!7] (axis cs:{n / main['band_factor']:g},{main['ymin']:g}) rectangle "
               f"(axis cs:{n * main['band_factor']:g},{main['ymax']:g});")
    # inside the axis, left of the band: the axis clips nodes beyond its edge
    out.append(f"    \\node[anchor=north east, font=\\scriptsize, black!60] at "
               f"(axis cs:{n / main['band_factor']:g},{main['ymax']:g}) {{$n={n}$}};")
    for p in plots:
        scheme = p.comment.lstrip("% ").split(":")[0]
        if scheme not in rec["shown"]:
            raise ValueError(f"source plot comment {p.comment!r} names no shown formula")
        out.append(f"    {p.comment}")
        out.append(f"    \\addplot[{figinset._plain_opts(p.opts, main.get('mark_size'))}] coordinates {{")
        out += [figinset._plain_coord(c) for c in p.coords]
        out.append("    };")
        if p.legend:
            out.append(f"    \\addlegendentry{{{labels.get(scheme, pfchilds._style(scheme, registry)[0])}}}")
    out += ["    \\addlegendimage{black, solid, mark=*, mark options={solid, fill=black}}",
            "    \\addlegendentry{measured $r$}",
            "    \\addlegendimage{black, dashed, mark=*, mark options={solid, fill=white}}",
            f"    \\addlegendentry{{extrapolated (fit on {figratio._fit_note(rec)})}}"]
    out.append("  \\end{loglogaxis}")
    out += dots_axis(rows, n, len(rec["seeds"]), dots, registry, (main["width_cm"] + dots["gap_cm"], 0))
    out += ["\\end{tikzpicture}", ""]
    return "\n".join(out)


def side_by_side_zoom_tex(source_text: str, rec: dict, cfg: dict, header: list[str], registry: dict) -> str:
    """The source figure's curves on the left (no error bars) with the zoom window
    outlined; the same plots on the right in that window, coordinates copied
    unchanged, error bars kept, each formula labelled at its last point."""
    main, z = cfg["main"], cfg["panel"]
    _, plots, _ = figinset.parse_figure(source_text)
    xtick = next(l.strip() for l in source_text.splitlines() if l.strip().startswith("xtick="))
    labels = z.get("labels", {})
    shown = [p for p in plots if any(z["xmin"] <= x <= z["xmax"] for x, _ in p.points())]
    for p in shown:
        for x, y in p.points():
            if z["xmin"] <= x <= z["xmax"] and not z["ymin"] <= y <= z["ymax"]:
                raise ValueError(f"{p.comment}: point ({x:g}, {y:g}) is in the zoom's n-range but outside its y-range")
    out = header + ["\\begin{tikzpicture}"]
    out += _main_axis(main, xtick)
    out.append(f"    \\draw[black!55, line width=0.5pt] (axis cs:{z['xmin']:g},{z['ymin']:g}) rectangle "
               f"(axis cs:{z['xmax']:g},{z['ymax']:g});")
    for p in plots:
        scheme = p.comment.lstrip("% ").split(":")[0]
        if scheme not in rec["shown"]:
            raise ValueError(f"source plot comment {p.comment!r} names no shown formula")
        out.append(f"    {p.comment}")
        out.append(f"    \\addplot[{figinset._plain_opts(p.opts, main.get('mark_size'))}] coordinates {{")
        out += [figinset._plain_coord(c) for c in p.coords]
        out.append("    };")
        if p.legend:
            out.append(f"    \\addlegendentry{{{labels.get(scheme, pfchilds._style(scheme, registry)[0])}}}")
    out += ["    \\addlegendimage{black, solid, mark=*, mark options={solid, fill=black}}",
            "    \\addlegendentry{measured $r$}",
            "    \\addlegendimage{black, dashed, mark=*, mark options={solid, fill=white}}",
            f"    \\addlegendentry{{extrapolated (fit on {figratio._fit_note(rec)})}}",
            "  \\end{loglogaxis}"]
    yticks = ",".join(f"{v:g}" for v in z["ytick"])
    ylabels = ",".join("{" + figinset._tick_label(v) + "}" for v in z["ytick"])
    xticks = ",".join(f"{v:g}" for v in z["xtick"])
    out += ["  \\begin{loglogaxis}[",
            "    scale only axis,",
            f"    at={{({main['width_cm'] + z['gap_cm']}cm,0cm)}}, anchor=south west,",
            f"    width={z['width_cm']}cm, height={z['height_cm']}cm,",
            f"    title={{{z['title']}}}, title style={{font=\\small}},",
            "    xlabel={System size $n$},",
            f"    xtick={{{xticks}}}, xticklabels={{{xticks}}}, xminorticks=false,",
            f"    ytick={{{yticks}}}, yticklabels={{{ylabels}}}, yminorticks=false,",
            f"    xmin={z['xmin']:g}, xmax={z['xmax']:g}, ymin={z['ymin']:g}, ymax={z['ymax']:g},",
            "    xmajorgrids=true, ymajorgrids=true, grid style={black!12}, tick label style={font=\\footnotesize},",
            "    clip mode=individual,",  # plots clipped to the window; the end labels outside it are not
            "  ]"]
    ends: dict[str, float] = {}
    for p in shown:
        scheme = p.comment.lstrip("% ").split(":")[0]
        out.append(f"    {p.comment}")
        opts = p.opts if "forget plot" in p.opts else p.opts.replace(", error bars/.cd", ", forget plot, error bars/.cd")
        # bars of up to eight formulas meet at each n: thinner and lighter than the source's so the means stay readable
        opts = opts.replace("error bar style={solid}", "error bar style={solid, line width=0.35pt, opacity=0.55}")
        out.append(f"    \\addplot[{opts}] coordinates {{")
        out += p.coords
        out.append("    };")
        x_last, y_last = max(p.points())
        if x_last <= z["xmax"]:
            ends[scheme] = y_last
    # labels at the right edge, spread in log10(y) so none overlap
    lo, hi = math.log10(z["ymin"]), math.log10(z["ymax"])
    gap = z["label_gap_cm"] * (hi - lo) / z["height_cm"]
    pos = spread_labels({k: math.log10(v) for k, v in ends.items()}, gap, lo, hi)
    x_end = max(x for p in shown for x, _ in p.points() if x <= z["xmax"])
    edge = z["xmax"]
    for scheme, y in ends.items():
        label, colour, _ = pfchilds._style(scheme, registry)
        # leader: level from the point to the axis edge, then out to the label, so it never crosses the data
        out += [f"    \\draw[{colour}!70, line width=0.35pt] (axis cs:{x_end:g},{y:.6g}) -- (axis cs:{edge:g},{y:.6g}) -- "
                f"([xshift=0.25cm]axis cs:{edge:g},{10 ** pos[scheme]:.6g});",
                f"    \\node[anchor=west, font=\\footnotesize, text={colour}!80!black] at "
                f"([xshift=0.28cm]axis cs:{edge:g},{10 ** pos[scheme]:.6g}) {{{labels.get(scheme, label)}}};"]
    out += ["  \\end{loglogaxis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def mean_rank_tex(rec: dict, series: dict[str, list[dict]], cfg: dict, header: list[str], registry: dict) -> str:
    """Mean rank over seeds against n; each line labelled at its right end."""
    k = len(rec["shown"])
    labels = cfg.get("labels", {})
    out = header + ["\\begin{tikzpicture}", "  \\begin{semilogxaxis}[",
                    "    scale only axis, clip=false,",
                    f"    width={cfg['width_cm']}cm, height={cfg['height_cm']}cm,",
                    "    xlabel={System size $n$},",
                    f"    ylabel={{{cfg['ylabel']}}},",
                    "    xtick={5,10,30,100,300,500}, xticklabels={5,10,30,100,300,500}, xminorticks=false,",
                    f"    xmin={cfg['xmin']:g}, xmax={cfg['xmax']:g},",
                    f"    y dir=reverse, ymin=0.6, ymax={k + 0.4}, ytick={{{','.join(str(i + 1) for i in range(k))}}},",
                    "    ymajorgrids=true, grid style={black!10}, tick label style={font=\\footnotesize},",
                    "  ]"]
    # figratio._divider puts its labels at ymax, which y dir=reverse sends to the bottom; these sit above the axis
    divider = figratio.DIVIDER_N
    out += [f"    \\draw[black!35] (axis cs:{divider},0.6) -- (axis cs:{divider},{k + 0.4});",
            f"    \\node[anchor=south east, font=\\scriptsize, black!60] at (axis cs:{divider},0.6) {{measured}};",
            f"    \\node[anchor=south west, font=\\scriptsize, black!60] at (axis cs:{divider},0.6) {{extrapolated}};"]
    for scheme in rec["shown"]:
        _, colour, mark = pfchilds._style(scheme, registry)
        pts = series[scheme]
        meas = [p for p in pts if p["measured"]]
        extra = [p for p in pts if not p["measured"]]
        style = f"{colour}, mark={mark}, mark size={cfg['mark_size']}, line width=0.8pt"
        out += [f"    % {scheme}: mean rank over seeds",
                f"    \\addplot[{style}, solid, mark options={{solid, fill={colour}}}] coordinates {{",
                *[f"      ({p['n']},{p['mean_rank']:.4g})" for p in meas], "    };",
                # the dashed line starts at the last measured point but marks only the extrapolated ones,
                # so that point keeps its filled (measured) marker
                f"    \\addplot[{style}, dashed, mark options={{solid, fill=white}}, "
                f"mark indices={{2,...,{len(extra) + 1}}}] coordinates {{",
                *[f"      ({p['n']},{p['mean_rank']:.4g})" for p in meas[-1:] + extra], "    };"]
    ends = {s: series[s][-1]["mean_rank"] for s in rec["shown"]}
    pos = spread_labels(ends, LABEL_GAP, 0.7, k + 0.3)
    x_end = series[rec["shown"][0]][-1]["n"]
    for scheme in rec["shown"]:
        label, colour, _ = pfchilds._style(scheme, registry)
        label = labels.get(scheme, label)
        out += [f"    \\draw[{colour}, line width=0.4pt] (axis cs:{x_end},{ends[scheme]:.4g}) -- "
                f"([xshift=0.35cm]axis cs:{x_end},{pos[scheme]:.4g});",
                f"    \\node[anchor=west, font=\\footnotesize, text={colour}!80!black] at "
                f"([xshift=0.4cm]axis cs:{x_end},{pos[scheme]:.4g}) {{{label} ({ends[scheme]:.1f})}};"]
    out.append("  \\end{semilogxaxis}")
    out.append(_note(cfg, rec, x_end, cfg["note_width_cm"], cfg["width_cm"] / 2 + cfg.get("note_shift_cm", 0), -1.2))
    out += ["\\end{tikzpicture}", ""]
    return "\n".join(out)


def header_lines(rec: dict, source: str, sha: str, what: str, extra: list[str]) -> list[str]:
    prov = provenance()
    out = ["% Generated by pysimcount.figleader -- do not edit; regenerate instead.",
           f"% figure: {rec['id']} -- {what}",
           f"% seeds {rec['seeds']}; metric={rec['metric']} kind={rec['kind']} merged "
           f"({rec.get('gate_set', 'Clifford+Rz (cz)')}, simcount template)",
           f"% shown: {', '.join(rec['shown'])}",
           f"% removed (see {rec['id']}-ranking.md): {', '.join(rec['removed']) or 'none'}"]
    out += [f"% {line}" for line in extra]
    out.append(f"{DERIVED_TAG} {source} by pysimcount.figleader; source_sha256={sha}")
    out.append(f"% figure made at git_commit={prov['git_commit']} git_dirty={prov['git_dirty']}")
    if prov["git_dirty"]:
        out.append("% WARNING: made from uncommitted code; the commit above does not reproduce it.")
    return out


# ------------------------------------------------------------------ driver

def run(cfg: dict, here: Path, render: bool) -> list[Path]:
    registry = formula_registry()
    written: list[Path] = []
    for fig in cfg["figures"]:
        src = here / fig["source"]
        text = src.read_text()
        rec = json.loads(text)
        sha = hashlib.sha256(text.encode()).hexdigest()
        tex_src = src.with_suffix(".tex")
        tex_text = tex_src.read_text()
        tex_sha = hashlib.sha256(tex_text.encode()).hexdigest()
        if rec.get("per_seed_stored"):
            measured = {}  # per_seed_at reads the record
        else:
            runs = pfchilds.load_run_set(here, fig["runs"])  # the record's own run set, n = 9, 11 included
            if sorted(runs) != sorted(rec["seeds"]):
                raise ValueError(f"runs have seeds {sorted(runs)}, the record has {rec['seeds']}")
            measured = figratio.measured_per_seed(rec, runs)
        n = fig["n"]
        rows = leaderboard(rec, measured, n)
        ranks = mean_rank_series(rec, measured)
        by_rank = [r["scheme"] for r in sorted(rows, key=lambda r: r["mean_rank"])]
        by_value = [r["scheme"] for r in rows]
        order_note = ("row order (plotted value) and mean-rank order agree" if by_rank == by_value else
                      f"row order (plotted value) {by_value} differs from mean-rank order {by_rank}")
        dot_lines = [f"n = {n}; x = the source record's plotted value; bars min..max over seeds",
                     "cheapest on = seeds on which the formula is cheapest; rank range = min..max of its rank over seeds",
                     "per-seed values: runs' totals if measured, else each seed's own fit c(n) ceil(a_s n^b_s)",
                     order_note]
        made = [
            (f"{rec['id']}-n{n}-dots", dots_tex(rec, rows, n, fig["dots"], header_lines(
                rec, src.name, sha, f"cost at n = {n}, one row per formula", dot_lines), registry)),
            (f"{rec['id']}-with-n{n}", side_by_side_tex(tex_text, rec, rows, n, fig["side_by_side"], header_lines(
                rec, tex_src.name, tex_sha, f"source curves (no error bars) with cost at n = {n} beside them",
                dot_lines + ["left axis: coordinates copied unchanged from the source .tex"]), registry)),
            (f"{rec['id']}-mean-rank", mean_rank_tex(rec, ranks, fig["mean_rank"], header_lines(
                rec, src.name, sha, "mean rank over seeds vs n",
                ["rank 1 = cheapest on that seed; equal costs share the mean rank",
                 "per-seed values: runs' totals if measured, else each seed's own fit c(n) ceil(a_s n^b_s)"]), registry)),
        ]
        if "zoom" in fig:
            z = fig["zoom"]["panel"]
            made.append((f"{rec['id']}-with-zoom", side_by_side_zoom_tex(tex_text, rec, fig["zoom"], header_lines(
                rec, tex_src.name, tex_sha, f"source curves (no error bars) with n = {z['xmin']:g}..{z['xmax']:g} enlarged",
                ["left axis: coordinates copied unchanged from the source .tex, error bars removed",
                 "right axis: the same coordinates and error bars, clipped to the window"]), registry)))
        side = src.with_name(f"{rec['id']}-leader.json")
        side.write_text(json.dumps({"source": src.name, "source_sha256": sha, "n": n, "seeds": rec["seeds"],
                                    "metric": rec["metric"], "rows": rows, "mean_rank": ranks}, indent=1))
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
